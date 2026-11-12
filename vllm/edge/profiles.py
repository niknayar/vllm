# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opinionated `vllm serve` starting points for edge hardware.

    python -m vllm.edge.profiles --list
    python -m vllm.edge.profiles --show edge-low-latency

These are *starting points*, not measured optima. Validate on your own
hardware with ``python -m vllm.edge.bench`` and tune from there. Only flags
that exist in vLLM's engine/server arguments are emitted.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    summary: str
    args: tuple[tuple[str, str | None], ...]  # (flag, value-or-None)
    notes: str


# Always applied: priority scheduling is what makes tiers meaningful, and the
# network defaults are the safe ones for a device on an untrusted network.
_COMMON: tuple[tuple[str, str | None], ...] = (
    ("--scheduling-policy", "priority"),
    ("--enable-prefix-caching", None),
    ("--enable-chunked-prefill", None),
)
_SECURE: tuple[tuple[str, str | None], ...] = (
    ("--host", "127.0.0.1"),
    ("--disable-fastapi-docs", None),
    ("--enable-request-id-headers", None),
)

PROFILES: dict[str, Profile] = {
    "edge-low-latency": Profile(
        "edge-low-latency",
        "Interactive traffic on a single 8-24 GB GPU; favour TTFT over throughput.",
        (
            ("--max-num-seqs", "16"),
            ("--max-num-batched-tokens", "1024"),
            ("--max-model-len", "4096"),
            ("--gpu-memory-utilization", "0.85"),
        ),
        "Small prefill chunks keep decode steps of high-priority requests from "
        "stalling behind long prompts.",
    ),
    "edge-balanced": Profile(
        "edge-balanced",
        "Mixed interactive + background traffic on a 16-48 GB GPU.",
        (
            ("--max-num-seqs", "32"),
            ("--max-num-batched-tokens", "2048"),
            ("--max-model-len", "8192"),
            ("--gpu-memory-utilization", "0.88"),
        ),
        "Default recommendation when tiers share one node.",
    ),
    "edge-throughput": Profile(
        "edge-throughput",
        "Batch-heavy nodes (document processing, offline enrichment).",
        (
            ("--max-num-seqs", "64"),
            ("--max-num-batched-tokens", "8192"),
            ("--max-model-len", "4096"),
            ("--gpu-memory-utilization", "0.90"),
        ),
        "Larger batches raise tokens/s at the cost of per-request latency.",
    ),
    "edge-memory-constrained": Profile(
        "edge-memory-constrained",
        "Unified-memory or small devices (Jetson-class, <=16 GB shared).",
        (
            ("--max-num-seqs", "8"),
            ("--max-num-batched-tokens", "512"),
            ("--max-model-len", "2048"),
            ("--gpu-memory-utilization", "0.70"),
            ("--kv-cache-dtype", "fp8"),
        ),
        "Leaves headroom for the OS and other processes. fp8 KV cache needs "
        "hardware/backend support; drop the flag if the engine rejects it.",
    ),
}


def render(profile: Profile, secure: bool = True) -> str:
    parts: list[str] = []
    flags = _COMMON + profile.args + (_SECURE if secure else ())
    for flag, value in flags:
        parts.append(flag if value is None else f"{flag} {shlex.quote(value)}")
    return " \\\n  ".join(["vllm serve <model>", *parts])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list profiles")
    g.add_argument("--show", metavar="NAME", help="print the serve command")
    p.add_argument(
        "--no-secure", action="store_true", help="omit network hardening flags"
    )
    args = p.parse_args(argv)

    if args.list:
        for prof in PROFILES.values():
            print(f"{prof.name:26s} {prof.summary}")
        return 0
    prof = PROFILES.get(args.show)
    if prof is None:
        print(
            f"unknown profile {args.show!r}; choose from {sorted(PROFILES)}",
            file=sys.stderr,
        )
        return 2
    print(f"# {prof.summary}\n# {prof.notes}")
    print(render(prof, secure=not args.no_secure))
    print("# Enable the guard with: VLLM_EDGE_CONFIG=/path/to/edge.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
