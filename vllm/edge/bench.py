# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tier load generator: does priority scheduling protect critical traffic?

Runs closed-loop streaming chat requests for each tier concurrently and
reports TTFT / end-to-end latency percentiles, output tokens/s and the
admission outcomes (429 rate-limited, 503 shed) per tier.

    export KEY_CRIT=...  KEY_BATCH=...
    python -m vllm.edge.bench --base-url http://127.0.0.1:8000 --model <model> \\
        --tier critical:KEY_CRIT:4 --tier batch:KEY_BATCH:32 --duration 60

``--tier NAME:ENV_VAR:CONCURRENCY`` reads the API key from an environment
variable so it never appears in the process list or shell history.
Run it once against a plain server and once with the edge guard + priority
scheduling to quantify the difference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class TierStats:
    name: str
    ttft: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    tokens: int = 0
    status: Counter = field(default_factory=Counter)

    def summary(self, wall: float) -> dict:
        ok = self.status.get(200, 0)
        return {
            "tier": self.name,
            "requests": sum(self.status.values()),
            "ok": ok,
            "status": dict(self.status),
            "ttft_ms": {
                p: round(percentile(self.ttft, p) * 1e3, 1) for p in (50, 95, 99)
            },
            "latency_ms": {
                p: round(percentile(self.latency, p) * 1e3, 1) for p in (50, 95, 99)
            },
            "output_tokens_per_s": round(self.tokens / wall, 1) if wall else 0.0,
        }


async def _one_request(session, url, key, model, prompt, max_tokens, stats: TierStats):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    first: float | None = None
    chunks = 0
    try:
        async with session.post(
            url, json=payload, headers={"Authorization": f"Bearer {key}"}
        ) as resp:
            stats.status[resp.status] += 1
            if resp.status != 200:
                await resp.read()
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                    continue
                try:
                    delta = json.loads(line[5:])["choices"][0]["delta"]
                except (ValueError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    chunks += 1
                    if first is None:
                        first = time.perf_counter() - start
    except Exception:  # noqa: BLE001 - network errors count as failures
        stats.status["error"] += 1
        return
    stats.latency.append(time.perf_counter() - start)
    if first is not None:
        stats.ttft.append(first)
    stats.tokens += chunks  # one SSE content chunk ~= one token


async def _worker(session, url, key, model, prompt, max_tokens, stats, deadline):
    while time.perf_counter() < deadline:
        await _one_request(session, url, key, model, prompt, max_tokens, stats)


async def run(args: argparse.Namespace) -> list[dict]:
    import aiohttp  # imported lazily so --help works without it

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    stats: dict[str, TierStats] = {}
    tasks = []
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        deadline = time.perf_counter() + args.duration
        t0 = time.perf_counter()
        for spec in args.tier:
            name, env_var, conc = parse_tier_spec(spec)
            key = os.environ.get(env_var)
            if not key:
                raise SystemExit(f"environment variable {env_var} is not set")
            stats[name] = TierStats(name)
            for _ in range(conc):
                tasks.append(
                    _worker(
                        session,
                        url,
                        key,
                        args.model,
                        args.prompt,
                        args.max_tokens,
                        stats[name],
                        deadline,
                    )
                )
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return [s.summary(wall) for s in stats.values()]


def parse_tier_spec(spec: str) -> tuple[str, str, int]:
    parts = spec.split(":")
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise SystemExit(f"bad --tier {spec!r}; expected NAME:ENV_VAR:CONCURRENCY")
    try:
        conc = int(parts[2])
    except ValueError:
        raise SystemExit(f"bad concurrency in --tier {spec!r}") from None
    if conc < 1:
        raise SystemExit(f"concurrency must be >= 1 in --tier {spec!r}")
    return parts[0], parts[1], conc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--tier", action="append", required=True, metavar="NAME:ENV_VAR:CONCURRENCY"
    )
    p.add_argument("--duration", type=float, default=30.0, help="seconds")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--prompt", default="Summarize the benefits of edge inference in two sentences."
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = p.parse_args(argv)

    results = asyncio.run(run(args))
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    cols = ("tier", "reqs", "ok", "ttft p50", "ttft p95", "lat p95", "tok/s")
    print(
        f"{cols[0]:12s} {cols[1]:>6s} {cols[2]:>6s} {cols[3]:>9s} "
        f"{cols[4]:>9s} {cols[5]:>9s} {cols[6]:>8s}  status"
    )
    for r in results:
        print(
            f"{r['tier']:12s} {r['requests']:6d} {r['ok']:6d} "
            f"{r['ttft_ms'][50]:9.1f} {r['ttft_ms'][95]:9.1f} "
            f"{r['latency_ms'][95]:9.1f} {r['output_tokens_per_s']:8.1f}  {r['status']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
