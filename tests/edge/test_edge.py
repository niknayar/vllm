# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from vllm_edge import (
    EdgeConfigError,
    EdgeGuardMiddleware,
    TokenBucket,
    bench,
    keytool,
    load_edge_config,
    parse_edge_config,
    profiles,
)

REPO = Path(__file__).resolve().parents[2]
KEYS = {"crit": "key-critical", "batch": "key-batch", "ops": "key-ops"}


def _cfg(**overrides):
    raw = {
        "max_body_bytes": 1024,
        "tiers": {
            "critical": {"priority": 0, "rate_per_sec": 1000, "burst": 1000},
            "batch": {
                "priority": 10,
                "rate_per_sec": 1000,
                "burst": 1000,
                "shed_at": 0.5,
                "max_concurrency": 4,
            },
        },
        "tenants": [
            {
                "name": "crit",
                "key_sha256": keytool.digest(KEYS["crit"]),
                "tier": "critical",
            },
            {
                "name": "batch",
                "key_sha256": keytool.digest(KEYS["batch"]),
                "tier": "batch",
            },
            {
                "name": "ops",
                "key_sha256": keytool.digest(KEYS["ops"]),
                "tier": "critical",
                "admin": True,
            },
        ],
    }
    raw.update(overrides)
    return parse_edge_config(raw)


def _make_app(cfg, gate: asyncio.Event | None = None, clock=None):
    seen = {}

    async def echo(request: Request):
        seen["body"] = await request.json() if await request.body() else None
        if gate is not None:
            await gate.wait()
        return JSONResponse(seen["body"] or {})

    async def ok(request: Request):
        return PlainTextResponse("ok")

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", echo, methods=["POST"]),
            Route("/v1/embeddings", echo, methods=["POST"]),
            Route("/health", ok),
            Route("/sleep", ok, methods=["POST"]),
            Route("/v1/models", ok),
        ]
    )
    kwargs = {"clock": clock} if clock else {}
    return EdgeGuardMiddleware(app, cfg, **kwargs), seen


def _client(asgi):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=asgi), base_url="http://edge"
    )


def _auth(name):
    return {"Authorization": f"Bearer {KEYS[name]}"}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- auth
def test_missing_and_wrong_key_rejected():
    async def go():
        asgi, _ = _make_app(_cfg())
        async with _client(asgi) as c:
            r1 = await c.get("/v1/models")
            r2 = await c.get("/v1/models", headers={"Authorization": "Bearer nope"})
            r3 = await c.get("/v1/models", headers={"Authorization": "Basic abc"})
            return r1, r2, r3

    for r in run(go()):
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        assert "nope" not in r.text


def test_valid_key_and_open_health():
    async def go():
        asgi, _ = _make_app(_cfg())
        async with _client(asgi) as c:
            return (
                await c.get("/v1/models", headers=_auth("crit")),
                await c.get("/health"),
            )

    ok, health = run(go())
    assert ok.status_code == 200
    assert health.status_code == 200  # probes stay unauthenticated


def test_admin_path_requires_admin_tenant():
    async def go():
        asgi, _ = _make_app(_cfg())
        async with _client(asgi) as c:
            return (
                await c.post("/sleep", headers=_auth("crit")),
                await c.post("/sleep", headers=_auth("ops")),
                await c.post("/sleep"),
            )

    normal, admin, anon = run(go())
    assert (normal.status_code, admin.status_code, anon.status_code) == (403, 200, 401)


# ------------------------------------------------------------ priority
def test_priority_is_forced_by_tier_and_client_value_ignored():
    async def go():
        asgi, seen = _make_app(_cfg())
        async with _client(asgi) as c:
            # A batch tenant tries to jump the queue with priority -1000.
            r = await c.post(
                "/v1/chat/completions",
                json={"model": "m", "priority": -1000},
                headers=_auth("batch"),
            )
            return r, seen["body"]

    r, body = run(go())
    assert r.status_code == 200
    assert body["priority"] == 10 and body["model"] == "m"


def test_priority_not_injected_on_other_paths_or_when_disabled():
    async def go(cfg):
        asgi, seen = _make_app(cfg)
        async with _client(asgi) as c:
            await c.post("/v1/embeddings", json={"x": 1}, headers=_auth("batch"))
            return seen["body"]

    assert run(go(_cfg())) == {"x": 1}

    async def go2():
        asgi, seen = _make_app(_cfg(inject_priority=False))
        async with _client(asgi) as c:
            await c.post("/v1/chat/completions", json={"a": 1}, headers=_auth("batch"))
            return seen["body"]

    assert run(go2()) == {"a": 1}


def test_malformed_json_passes_through_untouched():
    async def go():
        asgi, _ = _make_app(_cfg())
        async with _client(asgi) as c:
            return await c.post(
                "/v1/chat/completions",
                content=b"{not json",
                headers={**_auth("batch"), "content-type": "application/json"},
            )

    # Echo handler fails parsing -> server error from the app, not the guard.
    with pytest.raises(json.JSONDecodeError):
        run(go())


# --------------------------------------------------------------- limits
def test_body_limit_413_by_content_length_and_stream():
    async def go():
        asgi, _ = _make_app(_cfg())
        async with _client(asgi) as c:
            big = await c.post(
                "/v1/chat/completions", content=b"x" * 2048, headers=_auth("crit")
            )

            async def gen():
                for _ in range(4):
                    yield b"y" * 600

            streamed = await c.post(
                "/v1/chat/completions", content=gen(), headers=_auth("crit")
            )
            return big, streamed

    big, streamed = run(go())
    assert big.status_code == 413 and streamed.status_code == 413


def test_rate_limit_429_with_retry_after_and_refill():
    now = [0.0]
    cfg = _cfg(
        tiers={
            "critical": {"priority": 0, "rate_per_sec": 1, "burst": 2},
            "batch": {"priority": 10},
        }
    )

    async def go():
        asgi, _ = _make_app(cfg, clock=lambda: now[0])
        async with _client(asgi) as c:
            codes = [
                (await c.get("/v1/models", headers=_auth("crit"))) for _ in range(3)
            ]
            now[0] += 1.0
            after = await c.get("/v1/models", headers=_auth("crit"))
            return codes, after

    codes, after = run(go())
    assert [r.status_code for r in codes] == [200, 200, 429]
    assert int(codes[2].headers["retry-after"]) >= 1
    assert after.status_code == 200


def test_load_shedding_drops_batch_before_critical():
    # max_inflight=4, batch sheds at 0.5 -> at 2 in flight batch gets 503
    # while critical (shed_at 1.0) is still admitted.
    cfg = _cfg(max_inflight=4)

    async def go():
        gate = asyncio.Event()
        asgi, _ = _make_app(cfg, gate=gate)
        async with _client(asgi) as c:
            hold = [
                asyncio.create_task(
                    c.post("/v1/chat/completions", json={}, headers=_auth("crit"))
                )
                for _ in range(2)
            ]
            await asyncio.sleep(0.05)
            batch = await c.post(
                "/v1/chat/completions", json={}, headers=_auth("batch")
            )
            crit = asyncio.create_task(
                c.post("/v1/chat/completions", json={}, headers=_auth("crit"))
            )
            await asyncio.sleep(0.05)
            gate.set()
            done = await asyncio.gather(*hold, crit)
            return batch, done

    batch, done = run(go())
    assert batch.status_code == 503 and batch.headers["retry-after"] == "1"
    assert all(r.status_code == 200 for r in done)


def test_per_tenant_concurrency_cap_and_counters_released():
    cfg = _cfg(
        tiers={
            "critical": {"priority": 0},
            "batch": {"priority": 10, "max_concurrency": 1},
        }
    )

    async def go():
        gate = asyncio.Event()
        asgi, _ = _make_app(cfg, gate=gate)
        async with _client(asgi) as c:
            first = asyncio.create_task(
                c.post("/v1/chat/completions", json={}, headers=_auth("batch"))
            )
            await asyncio.sleep(0.05)
            second = await c.post(
                "/v1/chat/completions", json={}, headers=_auth("batch")
            )
            gate.set()
            await first
            third = await c.post(
                "/v1/chat/completions", json={}, headers=_auth("batch")
            )
            return second, third, asgi

    second, third, asgi = run(go())
    assert second.status_code == 429
    assert third.status_code == 200
    assert asgi._inflight == 0 and asgi._tenant_inflight["batch"] == 0


# -------------------------------------------------------------- headers
def test_security_headers_present_and_hsts_opt_in():
    async def go(cfg):
        asgi, _ = _make_app(cfg)
        async with _client(asgi) as c:
            return await c.get("/v1/models", headers=_auth("crit"))

    r = run(go(_cfg()))
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in r.headers
    assert "strict-transport-security" in run(go(_cfg(hsts=True))).headers


def test_audit_log_never_contains_key(caplog):
    caplog.set_level("INFO", logger="vllm.edge.audit")

    async def go():
        asgi, _ = _make_app(_cfg())
        async with _client(asgi) as c:
            await c.get("/v1/models", headers=_auth("crit"))
            await c.get("/v1/models", headers={"Authorization": "Bearer secret-guess"})

    run(go())
    text = caplog.text
    assert "tenant=crit" in text and "outcome=auth_failed" in text
    assert KEYS["crit"] not in text and "secret-guess" not in text


# --------------------------------------------------------------- config
def test_token_bucket_math():
    t = [0.0]
    b = TokenBucket(rate=2.0, burst=1, clock=lambda: t[0])
    assert b.try_acquire() == 0.0
    assert b.try_acquire() == pytest.approx(0.5)
    t[0] = 0.5
    assert b.try_acquire() == 0.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(bogus=1),
        lambda r: r.update(tenants=[]),
        lambda r: r["tenants"][0].update(key_sha256="plaintext-key"),
        lambda r: r["tenants"][0].update(tier="missing"),
        lambda r: r["tenants"].append(dict(r["tenants"][0])),
        lambda r: r["tiers"]["batch"].update(shed_at=0),
        lambda r: r["tiers"]["batch"].update(priority=True),
        lambda r: r.update(max_body_bytes=0),
        lambda r: r.update(guarded_prefixes=["v1"]),
    ],
)
def test_invalid_configs_fail_closed(mutate):
    raw = {
        "tiers": {"critical": {"priority": 0}, "batch": {"priority": 10}},
        "tenants": [
            {"name": "a", "key_sha256": "a" * 64, "tier": "critical"},
        ],
    }
    mutate(raw)
    with pytest.raises(EdgeConfigError):
        parse_edge_config(raw)


def test_load_refuses_world_writable_file(tmp_path):
    p = tmp_path / "edge.json"
    p.write_text(
        json.dumps(
            {
                "tiers": {"t": {"priority": 0}},
                "tenants": [{"name": "a", "key_sha256": "b" * 64, "tier": "t"}],
            }
        )
    )
    os.chmod(p, 0o600)
    assert load_edge_config(str(p)).tenants[0].name == "a"
    os.chmod(p, 0o666)
    with pytest.raises(EdgeConfigError, match="world-writable"):
        load_edge_config(str(p))


def test_example_config_parses():
    path = REPO / "examples" / "features" / "edge" / "edge_config.example.json"
    cfg = parse_edge_config(json.loads(path.read_text()))
    assert {t.name for t in cfg.tenants} >= {"example-gateway", "example-ops"}


# ----------------------------------------------------- tools & profiles
def test_keytool_digest_roundtrip(capsys):
    assert keytool.main(["--name", "svc", "--tier", "critical"]) == 0
    out = capsys.readouterr().out
    key = re.search(r"(vk-[A-Za-z0-9_-]+)", out).group(1)
    assert keytool.digest(key) in out
    assert keytool.main(["--name", "bad name!", "--tier", "x"]) == 2


def test_profiles_only_use_real_vllm_flags():
    sources = "".join(
        p.read_text()
        for p in [
            REPO / "vllm" / "engine" / "arg_utils.py",
            REPO / "vllm" / "entrypoints" / "common" / "cli_args.py",
            REPO / "vllm" / "config" / "scheduler.py",
            REPO / "vllm" / "config" / "cache.py",
        ]
    )
    for prof in profiles.PROFILES.values():
        for flag, _ in profiles._COMMON + prof.args + profiles._SECURE:
            attr = flag.lstrip("-").replace("-", "_")
            assert attr in sources or attr == "scheduling_policy", flag
    assert profiles.main(["--show", "edge-balanced"]) == 0
    assert profiles.main(["--show", "nope"]) == 2


def test_bench_percentile_and_spec_parsing():
    assert bench.percentile([], 95) == 0.0
    assert bench.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert bench.parse_tier_spec("critical:KEY:4") == ("critical", "KEY", 4)
    with pytest.raises(SystemExit):
        bench.parse_tier_spec("critical:KEY")
    with pytest.raises(SystemExit):
        bench.parse_tier_spec("critical:KEY:0")
