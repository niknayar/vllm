# vLLM Edge: secure, priority-aware serving for edge nodes

Branch: `feature/edge-secure-priority-inference` (fork of `vllm-project/vllm`)

This branch adds an optional, self-contained toolkit under `vllm/edge/` for
running vLLM on shared edge nodes: a single GPU or Jetson-class box that
serves several callers with different urgency, on a network you do not fully
trust. Nothing changes unless you set `VLLM_EDGE_CONFIG`.

## What problem this solves

An edge node has a fixed, small amount of GPU memory and compute. Three
things go wrong when several callers share it:

1. **Head-of-line blocking.** A burst of low-value batch requests fills the
   scheduler queue, and the request a user is waiting on sits behind it.
2. **Overload collapse.** Past saturation, queues grow without bound and
   *every* tier sees timeouts, including the critical one.
3. **Weak perimeter.** Stock vLLM has one shared API key, no per-caller
   limits, no cap on request size, and several operational endpoints that are
   either open or only gated by that same key.

## What "acceleration" means here (and what it does not)

The toolkit does **not** make the model's forward pass faster. It improves
**latency and availability of the traffic you care about** under contention:

| Mechanism | Effect on the node |
|---|---|
| Per-tier `priority` injected server-side | vLLM's priority scheduler (`--scheduling-policy priority`) admits and preempts lower-priority work first, so critical TTFT stays low while batch work soaks up spare capacity. |
| Priority-aware load shedding (`shed_at`) | When in-flight requests approach `max_inflight`, batch is rejected with `503 + Retry-After` first. Critical requests keep headroom instead of joining a queue they would time out in. |
| Per-tenant token bucket + concurrency cap | One noisy caller cannot consume the batch budget of the whole node. |
| Bounded request bodies | Large prompts cannot force unbounded buffering on a memory-poor device. |
| Edge engine profiles | Starting values for `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--gpu-memory-utilization`, prefix caching, chunked prefill and optional fp8 KV cache sized for small devices. |
| `vllm.edge.bench` | Closed-loop, per-tier load generator so you can measure TTFT/latency/throughput and shed/limit counts with and without the guard. |

**No real-hardware vLLM numbers are claimed.** The profiles are starting
points, not measured optima. Expected benefit scales with how mixed your
traffic is: with one caller and no contention there is nothing to prioritise.
See [Simulation results](#simulation-results) for what has been measured.

## Security model

The guard (`EdgeGuardMiddleware`) is pure ASGI and is installed as the
**outermost** middleware, so rejected requests cost almost nothing.

- **Per-tenant API keys.** Config stores only SHA-256 digests of 256-bit
  random keys (`python -m vllm.edge.keytool`). Comparison is constant-time
  and visits every tenant, so timing does not reveal which entry matched.
- **No priority escalation.** The `priority` field in the JSON body is
  *overwritten* with the tenant's tier value on inference endpoints. Without
  this, any client could send `"priority": -1000` and jump the queue.
- **Admin separation.** Operational endpoints (`/sleep`, `/wake_up`,
  `/reset_prefix_cache`, `/collective_rpc`, `/server_info`, ...) need a tenant
  flagged `"admin": true`; ordinary tenants get `403`. `/metrics` and the
  tokenizer/pooling/score endpoints require a valid key. `/health` stays open
  for orchestrator probes.
- **Fail closed.** Unknown config keys, malformed digests, duplicate tenants,
  dangling tier references, a world-writable config file, or non-zero
  priorities without `--scheduling-policy priority` all abort start-up.
- **Abuse limits.** Token bucket (`429` + `Retry-After`), per-tenant
  concurrency (`429`), global shedding (`503`), body size (`413`, enforced on
  both `Content-Length` and streamed bytes).
- **Response hardening.** `X-Content-Type-Options`, `Referrer-Policy`,
  `X-Frame-Options`, `Cache-Control: no-store` on API responses; optional HSTS.
- **Audit log.** One line per guarded request (`vllm.edge.audit`): tenant,
  tier, method, path, status, outcome, duration, client address. Keys,
  `Authorization` values and bodies are never logged (covered by a test).
- **WebSocket** routes under guarded prefixes are authenticated and closed
  with code 1008 on failure.

### Not covered / known limits

- **TLS is not terminated by the guard.** Use `--ssl-keyfile/--ssl-certfile`
  or terminate at a reverse proxy / load balancer, then set `"hsts": true`.
- **State is per process.** Rate limits, concurrency and shedding counters live
  in memory. With `--api-server-count > 1` or several replicas, each process
  enforces its own limits; set them accordingly.
- **Client address is the socket peer.** Behind a proxy the audit log shows the
  proxy; the guard deliberately does not trust `X-Forwarded-For`.
- **Admission is by request count, not tokens.** A tier that sends few, very
  long requests is limited by concurrency and body size, not token budget.
- **Keys are unsalted SHA-256.** Appropriate for random 256-bit keys, not for
  human-chosen passwords. Rotate by adding a new tenant entry, then removing
  the old one and restarting.
- **No replacement for network controls.** Keep the default `--host 127.0.0.1`
  (or a private interface) and restrict ingress with firewall / security groups.

## Quick start

```bash
# 1. Mint a key per caller; put the printed entry in your config.
python -m vllm.edge.keytool --name gateway --tier critical
python -m vllm.edge.keytool --name nightly-job --tier batch
python -m vllm.edge.keytool --name ops --tier critical --admin

# 2. Start from the example config and lock the file down.
cp examples/features/edge/edge_config.example.json /etc/vllm/edge.json
chmod 600 /etc/vllm/edge.json

# 3. Pick an engine profile and serve.
python -m vllm.edge.profiles --list
python -m vllm.edge.profiles --show edge-balanced      # prints the command
VLLM_EDGE_CONFIG=/etc/vllm/edge.json vllm serve <model> \
  --scheduling-policy priority --host 127.0.0.1 ...

# 4. Measure. Keys come from environment variables, not argv.
KEY_CRIT=... KEY_BATCH=... python -m vllm.edge.bench \
  --model <model> --tier critical:KEY_CRIT:4 --tier batch:KEY_BATCH:32 --duration 60
```

Store real keys in a secrets manager (for example AWS Secrets Manager), never
in the repo. The digests in the example config are placeholders.

### Config reference

```jsonc
{
  "max_body_bytes": 4194304,   // 413 above this
  "max_inflight": 64,          // global in-flight cap used for shedding (optional)
  "inject_priority": true,     // overwrite body "priority" with the tier value
  "tiers": {
    "critical": {"priority": 0, "rate_per_sec": 50, "burst": 100,
                 "max_concurrency": 16, "shed_at": 1.0},
    "batch":    {"priority": 10, "rate_per_sec": 5, "burst": 10,
                 "max_concurrency": 8, "shed_at": 0.5}
  },
  "tenants": [
    {"name": "gateway", "key_sha256": "<64 hex>", "tier": "critical"},
    {"name": "ops", "key_sha256": "<64 hex>", "tier": "critical", "admin": true}
  ]
}
```

Lower `priority` is served first (vLLM semantics). `shed_at` is the fraction of
`max_inflight` at which that tier starts receiving `503`. Optional lists
`guarded_prefixes`, `admin_prefixes` and `priority_paths` override the defaults
in `vllm/edge/config.py`.

## Modification details

| File | Change |
|---|---|
| `vllm/edge/config.py` (new) | Strict, fail-closed config parsing and validation. |
| `vllm/edge/middleware.py` (new) | `EdgeGuardMiddleware`, `TokenBucket`. |
| `vllm/edge/profiles.py` (new) | Edge engine-argument profiles; `python -m vllm.edge.profiles`. |
| `vllm/edge/keytool.py` (new) | Key and digest generator. |
| `vllm/edge/bench.py` (new) | Per-tier streaming benchmark. |
| `vllm/entrypoints/serve/middleware/register.py` | Installs the guard last (outermost) when `VLLM_EDGE_CONFIG` is set; refuses to start on a priority-policy mismatch. |
| `vllm/envs.py` | Registers `VLLM_EDGE_CONFIG` (also excluded from compile-cache keys). |
| `examples/features/edge/edge_config.example.json` (new) | Example three-tier config. |
| `docs/features/edge.md` (new) | User documentation. |
| `tests/edge/` (new) | 27 tests; run without torch or a built vLLM. |

The existing `--api-key` middleware is untouched. If both are configured, a
request must satisfy both, and a warning is logged; prefer edge tenants alone.

## Testing

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install starlette httpx pytest aiohttp
pytest tests/edge --confcutdir=tests/edge -q
```

Covered: auth failures, admin gating, priority overwrite and non-injection
paths, 413 by header and by stream, rate-limit refill, shedding order,
concurrency release, security headers, audit-log secrecy, config rejection
cases, world-writable config, key tool round-trip, and that every flag in the
profiles exists in vLLM's argument sources.

## Simulation results

Harness: `benchmarks/edge/sim_benchmark.py`; raw output:
`benchmarks/edge/results/sim_results_apple_m5.json`. Run on an Apple M5,
Python 3.12, 8 s per scenario.

**Part 1: guard overhead (real middleware code, 20,000 requests, 661 B body).**
Bare ASGI app 1.8 us/request, with guard 25.6 us/request, so the guard adds
about 24 us per request.

**Part 2: contention model (simulated engine).** The real guard sits in front
of a fake engine with 4 slots at 50 ms per request (about 80 req/s capacity),
a first-come-first-served or priority waiting queue, 4 interactive critical
workers (200 ms think time) and 32 batch workers.

| Scenario | Tier | OK | Rejected | req/s | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| A: stock FCFS, no guard | critical | 52 | 0 | 6.5 | 442 | 452 |
| | batch | 572 | 0 | 71.5 | 482 | 489 |
| B: priority scheduling, no guard, batch sends `priority: -1000` | critical | 8 | 0 | 1.0 | 4116 | 8181 |
| | batch | 616 | 0 | 77.0 | 434 | 439 |
| C: priority scheduling + guard | critical | 120 | 0 | 15.0 | 55 | 103 |
| | batch | 431 | 1171 | 53.9 | 51 | 54 |
| D: guard, batch still sends `priority: -1000` | critical | 120 | 0 | 15.0 | 56 | 100 |
| | batch | 430 | 1172 | 53.8 | 52 | 55 |
| E: guard, critical saturates the node | critical | 616 | 0 | 77.0 | 52 | 53 |
| | batch | 0 | 1280 | 0.0 | - | - |

What it shows: critical p50 falls from about 442 ms (A) to about 55 ms (C);
without the guard a client can cheat its priority and push critical p95 to
about 8 s (B), and the guard prevents that (D). Costs: batch throughput drops
from 71.5 to 53.9 req/s because shed requests leave slots idle, and a critical
tenant that never pauses (E) starves batch completely, so cap critical tenants
with `rate_per_sec` or `max_concurrency` if that is not wanted.

**These are model results, not vLLM results.** The engine has a fixed service
time, does not preempt running requests, and has no KV cache or token-level
batching. Do not read the 8x figure as a prediction for vLLM.

**Not yet verified:** an end-to-end run against a live model on GPU/Jetson
hardware. The `bench` tool is provided for that; results should be added here
once collected.

## AI assistance

This change was developed with AI assistance (Kiro) and has not been proposed
upstream. If you submit it to `vllm-project/vllm`, follow `AGENTS.md`: a human
must review every line, run the test suite, and complete the duplicate-work
checks first.
