# Edge Serving: Security and Priority Tiers

The edge toolkit (`vllm.edge`) lets one vLLM node serve several callers with
different urgency on an untrusted network. It is disabled unless the
`VLLM_EDGE_CONFIG` environment variable points to a JSON config file.

It does not speed up the model itself. It protects the latency of
high-priority traffic by combining priority scheduling, load shedding and
per-tenant limits, and it adds per-tenant authentication.

## Enable

1. Generate a key per caller:

    ```bash
    python -m vllm.edge.keytool --name gateway --tier critical
    ```

2. Put the printed entry in a config file (see
   `examples/features/edge/edge_config.example.json`) and restrict the
   file permissions (`chmod 600`). World-writable files are rejected.

3. Start the server with priority scheduling:

    ```bash
    VLLM_EDGE_CONFIG=/etc/vllm/edge.json \
      vllm serve <model> --scheduling-policy priority --host 127.0.0.1
    ```

Start-up fails if tiers use non-zero priorities and the scheduling policy is
not `priority`, or if the config is invalid.

## Behavior

For guarded paths the middleware authenticates the bearer token, checks the
admin gate, applies a per-tenant token bucket (`429`), sheds load by tier when
`max_inflight` is approached (`503`), caps per-tenant concurrency (`429`),
bounds the request body (`413`), and overwrites the request `priority` with the
tenant's tier value so clients cannot escalate. `/health` is not guarded.

## Engine profiles and benchmark

```bash
python -m vllm.edge.profiles --list
python -m vllm.edge.profiles --show edge-memory-constrained
KEY_CRIT=... KEY_BATCH=... python -m vllm.edge.bench --model <model> \
  --tier critical:KEY_CRIT:4 --tier batch:KEY_BATCH:32
```

Profiles are starting points; validate them with the benchmark on your
hardware.

## Limitations

State is kept per process, TLS is not terminated by the guard, and the audit
log records the socket peer address rather than `X-Forwarded-For`.
