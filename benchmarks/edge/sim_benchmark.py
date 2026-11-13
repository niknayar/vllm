# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurements for the edge guard that run on any machine (no GPU).

Part 1 - guard overhead (REAL code path): per-request CPU cost of
         EdgeGuardMiddleware vs the bare ASGI app.
Part 2 - contention model (SIMULATED engine): the real guard sits in front of
         a fake engine with N decode slots and a fixed service time, using
         either an FCFS queue (stock vLLM default) or a priority queue
         (--scheduling-policy priority). It shows the *scheduling effect*; it
         is NOT a vLLM throughput/latency measurement. Use
         `python -m vllm.edge.bench` against a real server for that.

    python benchmarks/edge/sim_benchmark.py [--duration 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

_EDGE = Path(__file__).resolve().parents[2] / "vllm" / "edge"
_spec = importlib.util.spec_from_file_location(
    "vllm_edge", _EDGE / "__init__.py", submodule_search_locations=[str(_EDGE)]
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["vllm_edge"] = _mod
_spec.loader.exec_module(_mod)
from vllm_edge import EdgeGuardMiddleware, parse_edge_config  # noqa: E402

KEYS = {"critical": "k-critical", "batch": "k-batch"}


def make_config(max_inflight: int | None = 8):
    def dg(k):
        return hashlib.sha256(k.encode()).hexdigest()

    return parse_edge_config(
        {
            "max_inflight": max_inflight,
            "tiers": {
                "critical": {"priority": 0, "shed_at": 1.0},
                "batch": {"priority": 10, "shed_at": 0.5},
            },
            "tenants": [
                {
                    "name": "crit",
                    "key_sha256": dg(KEYS["critical"]),
                    "tier": "critical",
                },
                {"name": "batch", "key_sha256": dg(KEYS["batch"]), "tier": "batch"},
            ],
        }
    )


# ------------------------------------------------------------ part 1
async def _call(app, scope_tpl, body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)

    async def send(_m):
        pass

    await app(dict(scope_tpl), receive, send)


async def overhead(n: int = 20000) -> dict:
    async def bare(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    body = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "hi " * 200}]}
    ).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "root_path": "",
        "client": ("127.0.0.1", 1),
        "headers": [
            (b"authorization", f"Bearer {KEYS['critical']}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    guarded = EdgeGuardMiddleware(bare, make_config(None), clock=time.monotonic)
    out = {}
    for name, app in (("bare_asgi", bare), ("edge_guard", guarded)):
        for _ in range(500):  # warm-up
            await _call(app, scope, body)
        t0 = time.perf_counter()
        for _ in range(n):
            await _call(app, scope, body)
        out[name] = (time.perf_counter() - t0) / n * 1e6
    out["added_us_per_request"] = out["edge_guard"] - out["bare_asgi"]
    out["body_bytes"] = len(body)
    out["requests"] = n
    return out


# ------------------------------------------------------------ part 2
class FakeEngine:
    """N slots, fixed service time. FCFS or priority-ordered waiting queue.
    Running requests are not preempted (a simplification)."""

    def __init__(self, slots: int, service_s: float, policy: str):
        self.free = slots
        self.service = service_s
        self.policy = policy
        self.waiting: list[tuple[float, float, int, asyncio.Future]] = []
        self.seq = 0

    def _pump(self):
        while self.free and self.waiting:
            key = (
                (lambda w: (w[0], w[1]))
                if self.policy == "priority"
                else (lambda w: (w[2], 0))
            )
            self.waiting.sort(key=key)
            _, _, _, fut = self.waiting.pop(0)
            self.free -= 1
            fut.set_result(None)

    async def serve(self, priority: int):
        fut = asyncio.get_running_loop().create_future()
        self.seq += 1
        self.waiting.append((priority, time.monotonic(), self.seq, fut))
        self._pump()
        await fut
        await asyncio.sleep(self.service)
        self.free += 1
        self._pump()


def make_app(engine: FakeEngine):
    async def completions(request: Request):
        body = await request.json()
        await engine.serve(int(body.get("priority", 0)))
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[Route("/v1/chat/completions", completions, methods=["POST"])]
    )


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def scenario(
    name: str,
    *,
    policy: str,
    guard: bool,
    batch_cheats: bool,
    duration: float,
    slots: int = 4,
    service_s: float = 0.05,
    crit_workers: int = 4,
    batch_workers: int = 32,
    crit_think_s: float = 0.2,
) -> dict:
    engine = FakeEngine(slots, service_s, policy)
    app = make_app(engine)
    if guard:
        app = EdgeGuardMiddleware(app, make_config(max_inflight=8))
    stats = {t: {"lat": [], "ok": 0, "rejected": 0} for t in ("critical", "batch")}
    deadline = time.perf_counter() + duration

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://edge"
    ) as c:

        async def worker(tier: str):
            payload = {"model": "m", "messages": []}
            if tier == "batch" and batch_cheats:
                payload["priority"] = -1000  # try to jump the queue
            hdr = {"Authorization": f"Bearer {KEYS[tier]}"}
            while time.perf_counter() < deadline:
                t0 = time.perf_counter()
                r = await c.post("/v1/chat/completions", json=payload, headers=hdr)
                if r.status_code == 200:
                    stats[tier]["lat"].append(time.perf_counter() - t0)
                    stats[tier]["ok"] += 1
                    if tier == "critical" and crit_think_s:
                        await asyncio.sleep(crit_think_s)  # interactive user
                else:
                    stats[tier]["rejected"] += 1
                    await asyncio.sleep(0.2)  # honour Retry-After-ish backoff

        await asyncio.gather(
            *[worker("critical") for _ in range(crit_workers)],
            *[worker("batch") for _ in range(batch_workers)],
        )

    row = {"scenario": name}
    for tier, s in stats.items():
        row[tier] = {
            "ok": s["ok"],
            "rejected": s["rejected"],
            "req_per_s": round(s["ok"] / duration, 1),
            "p50_ms": round(pct(s["lat"], 50) * 1e3, 1),
            "p95_ms": round(pct(s["lat"], 95) * 1e3, 1),
            "p99_ms": round(pct(s["lat"], 99) * 1e3, 1),
            "mean_ms": round(statistics.fmean(s["lat"]) * 1e3, 1) if s["lat"] else 0.0,
        }
    return row


async def main_async(args) -> dict:
    result = {"overhead": await overhead(), "scenarios": []}
    d = args.duration
    for name, kw in [
        (
            "A stock: FCFS, no guard",
            dict(policy="fcfs", guard=False, batch_cheats=False),
        ),
        (
            "B priority sched, no guard, batch cheats",
            dict(policy="priority", guard=False, batch_cheats=True),
        ),
        (
            "C priority sched + edge guard",
            dict(policy="priority", guard=True, batch_cheats=False),
        ),
        (
            "D priority sched + edge guard, batch cheats",
            dict(policy="priority", guard=True, batch_cheats=True),
        ),
        (
            "E guard, critical saturates node (no think)",
            dict(policy="priority", guard=True, batch_cheats=False, crit_think_s=0.0),
        ),
    ]:
        result["scenarios"].append(await scenario(name, duration=d, **kw))
    return result


def render(res: dict) -> str:
    o = res["overhead"]
    lines = [
        "PART 1 - guard overhead (real middleware, no engine)",
        f"  requests={o['requests']} body={o['body_bytes']}B",
        f"  bare ASGI : {o['bare_asgi']:.1f} us/req",
        f"  with guard: {o['edge_guard']:.1f} us/req",
        f"  added     : {o['added_us_per_request']:.1f} us/req",
        "",
        (
            "PART 2 - contention model (SIMULATED engine: 4 slots x 50 ms, "
            "4 interactive critical (200 ms think) + 32 batch workers)"
        ),
        (
            f"{'scenario':46s} {'tier':9s} {'ok':>5s} {'rej':>5s} {'req/s':>6s} "
            f"{'p50ms':>7s} {'p95ms':>7s} {'p99ms':>7s}"
        ),
    ]
    for s in res["scenarios"]:
        for tier in ("critical", "batch"):
            t = s[tier]
            lines.append(
                f"{s['scenario'] if tier == 'critical' else '':46s} {tier:9s} "
                f"{t['ok']:5d} {t['rejected']:5d} {t['req_per_s']:6.1f} "
                f"{t['p50_ms']:7.1f} {t['p95_ms']:7.1f} {t['p99_ms']:7.1f}"
            )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()
    res = asyncio.run(main_async(args))
    print(render(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
