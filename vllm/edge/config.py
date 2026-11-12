# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the edge guard (security + priority tiers).

The config is a JSON document. It never contains plaintext API keys, only
their SHA-256 digests (see ``python -m vllm.edge.keytool``). Parsing is
strict and fail-closed: unknown keys, malformed digests, duplicate tenants
and dangling tier references abort server start-up instead of silently
weakening protection.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Any

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_PRIORITY_BOUND = 2**31 - 1
_MAX_CONFIG_BYTES = 1 << 20

DEFAULT_GUARDED_PREFIXES: tuple[str, ...] = (
    "/v1",
    "/v2",
    "/inference",
    "/cohere",
    "/tokenize",
    "/detokenize",
    "/pooling",
    "/score",
    "/rerank",
    "/metrics",
)
# Operational endpoints that must never be reachable by ordinary tenants.
DEFAULT_ADMIN_PREFIXES: tuple[str, ...] = (
    "/reset_prefix_cache",
    "/reset_mm_cache",
    "/sleep",
    "/wake_up",
    "/is_sleeping",
    "/collective_rpc",
    "/scale_elastic_ep",
    "/server_info",
    "/load",
)
DEFAULT_PRIORITY_PATHS: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
)


class EdgeConfigError(ValueError):
    """Raised when the edge configuration is invalid."""


@dataclass(frozen=True)
class TierConfig:
    """Per-tier QoS and admission limits.

    ``priority`` follows vLLM semantics: a *lower* value is scheduled first.
    ``shed_at`` is the fraction of ``max_inflight`` above which requests of
    this tier are rejected with 503, so low tiers are shed before critical
    traffic when the node is saturated.
    """

    name: str
    priority: int
    rate_per_sec: float | None = None
    burst: int = 1
    max_concurrency: int | None = None
    shed_at: float = 1.0


@dataclass(frozen=True)
class TenantConfig:
    name: str
    key_digest: bytes
    tier: str
    admin: bool = False


@dataclass(frozen=True)
class EdgeConfig:
    tenants: tuple[TenantConfig, ...]
    tiers: dict[str, TierConfig]
    max_body_bytes: int = 8 * 1024 * 1024
    max_inflight: int | None = None
    inject_priority: bool = True
    security_headers: bool = True
    hsts: bool = False
    audit_log: bool = True
    guarded_prefixes: tuple[str, ...] = DEFAULT_GUARDED_PREFIXES
    admin_prefixes: tuple[str, ...] = DEFAULT_ADMIN_PREFIXES
    priority_paths: tuple[str, ...] = DEFAULT_PRIORITY_PATHS


def _reject_unknown(d: dict[str, Any], allowed: set[str], ctx: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise EdgeConfigError(f"{ctx}: unknown keys {sorted(unknown)}")


def _int(v: Any, ctx: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise EdgeConfigError(f"{ctx}: expected integer in [{lo}, {hi}], got {v!r}")
    return v


def _str_tuple(v: Any, ctx: str) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(
        isinstance(x, str) and x.startswith("/") for x in v
    ):
        raise EdgeConfigError(f"{ctx}: expected a list of paths starting with '/'")
    return tuple(v)


def _parse_tier(name: str, raw: Any) -> TierConfig:
    ctx = f"tiers.{name}"
    if not _NAME_RE.match(name):
        raise EdgeConfigError(f"{ctx}: invalid tier name")
    if not isinstance(raw, dict):
        raise EdgeConfigError(f"{ctx}: expected an object")
    _reject_unknown(
        raw, {"priority", "rate_per_sec", "burst", "max_concurrency", "shed_at"}, ctx
    )
    if "priority" not in raw:
        raise EdgeConfigError(f"{ctx}: 'priority' is required")
    priority = _int(
        raw["priority"], f"{ctx}.priority", -_PRIORITY_BOUND, _PRIORITY_BOUND
    )

    rate = raw.get("rate_per_sec")
    if rate is not None and (
        isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0
    ):
        raise EdgeConfigError(f"{ctx}.rate_per_sec: must be a positive number or null")
    burst = _int(raw.get("burst", 1), f"{ctx}.burst", 1, 10**6)

    max_conc = raw.get("max_concurrency")
    if max_conc is not None:
        max_conc = _int(max_conc, f"{ctx}.max_concurrency", 1, 10**6)

    shed_at = raw.get("shed_at", 1.0)
    if (
        isinstance(shed_at, bool)
        or not isinstance(shed_at, (int, float))
        or not 0 < shed_at <= 1
    ):
        raise EdgeConfigError(f"{ctx}.shed_at: must be in (0, 1]")
    return TierConfig(
        name=name,
        priority=priority,
        rate_per_sec=float(rate) if rate is not None else None,
        burst=burst,
        max_concurrency=max_conc,
        shed_at=float(shed_at),
    )


def parse_edge_config(raw: Any) -> EdgeConfig:
    if not isinstance(raw, dict):
        raise EdgeConfigError("top level must be a JSON object")
    _reject_unknown(
        raw,
        {
            "tenants",
            "tiers",
            "max_body_bytes",
            "max_inflight",
            "inject_priority",
            "security_headers",
            "hsts",
            "audit_log",
            "guarded_prefixes",
            "admin_prefixes",
            "priority_paths",
        },
        "config",
    )

    raw_tiers = raw.get("tiers")
    if not isinstance(raw_tiers, dict) or not raw_tiers:
        raise EdgeConfigError("tiers: at least one tier is required")
    tiers = {n: _parse_tier(n, t) for n, t in raw_tiers.items()}

    raw_tenants = raw.get("tenants")
    if not isinstance(raw_tenants, list) or not raw_tenants:
        # Fail closed: an enabled guard with no tenants would reject everything
        # and almost certainly indicates a broken deployment.
        raise EdgeConfigError("tenants: at least one tenant is required")

    tenants: list[TenantConfig] = []
    names: set[str] = set()
    digests: set[str] = set()
    for i, t in enumerate(raw_tenants):
        ctx = f"tenants[{i}]"
        if not isinstance(t, dict):
            raise EdgeConfigError(f"{ctx}: expected an object")
        _reject_unknown(t, {"name", "key_sha256", "tier", "admin"}, ctx)
        name = t.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise EdgeConfigError(f"{ctx}.name: must match {_NAME_RE.pattern}")
        digest = t.get("key_sha256")
        if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
            raise EdgeConfigError(
                f"{ctx}.key_sha256: must be 64 lowercase hex chars "
                "(never store the plaintext key)"
            )
        tier = t.get("tier")
        if tier not in tiers:
            raise EdgeConfigError(f"{ctx}.tier: unknown tier {tier!r}")
        admin = t.get("admin", False)
        if not isinstance(admin, bool):
            raise EdgeConfigError(f"{ctx}.admin: must be a boolean")
        if name in names:
            raise EdgeConfigError(f"{ctx}.name: duplicate tenant {name!r}")
        if digest in digests:
            raise EdgeConfigError(f"{ctx}.key_sha256: duplicate key digest")
        names.add(name)
        digests.add(digest)
        tenants.append(TenantConfig(name, bytes.fromhex(digest), tier, admin))

    max_inflight = raw.get("max_inflight")
    if max_inflight is not None:
        max_inflight = _int(max_inflight, "max_inflight", 1, 10**6)

    kwargs: dict[str, Any] = {}
    for key in ("inject_priority", "security_headers", "hsts", "audit_log"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise EdgeConfigError(f"{key}: must be a boolean")
            kwargs[key] = raw[key]
    for key in ("guarded_prefixes", "admin_prefixes", "priority_paths"):
        if key in raw:
            kwargs[key] = _str_tuple(raw[key], key)

    return EdgeConfig(
        tenants=tuple(tenants),
        tiers=tiers,
        max_body_bytes=_int(
            raw.get("max_body_bytes", 8 * 1024 * 1024),
            "max_body_bytes",
            1,
            2**31 - 1,
        ),
        max_inflight=max_inflight,
        **kwargs,
    )


def load_edge_config(path: str) -> EdgeConfig:
    """Load and validate the config file at ``path``.

    Refuses world-writable files: anyone able to edit the file could mint
    keys or raise their own priority.
    """
    try:
        st = os.stat(path)
    except OSError as e:
        raise EdgeConfigError(f"cannot read edge config {path!r}: {e}") from e
    if not stat.S_ISREG(st.st_mode):
        raise EdgeConfigError(f"edge config {path!r} is not a regular file")
    if st.st_mode & stat.S_IWOTH:
        raise EdgeConfigError(
            f"edge config {path!r} is world-writable; run chmod o-w on it"
        )
    if st.st_size > _MAX_CONFIG_BYTES:
        raise EdgeConfigError(f"edge config {path!r} is larger than 1 MiB")
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise EdgeConfigError(f"edge config {path!r} is not valid JSON: {e}") from e
    return parse_edge_config(raw)
