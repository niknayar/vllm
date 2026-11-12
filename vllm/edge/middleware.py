# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-ASGI edge guard for the vLLM API server.

Request pipeline for guarded paths (outermost to innermost):

1. Authenticate the bearer token -> tenant (constant-time, no early exit).
2. Admin gate for operational endpoints.
3. Per-tenant token-bucket rate limit            -> 429
4. Priority-aware global load shedding           -> 503
5. Per-tenant concurrency cap                    -> 429
6. Bounded request-body read                     -> 413
7. Server-side ``priority`` injection (client-supplied value is overwritten,
   so a tenant cannot escalate its own scheduling priority).

Plaintext keys and request bodies are never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import secrets
import time
from collections.abc import Callable
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import EdgeConfig, TenantConfig, TierConfig

logger = logging.getLogger("vllm.edge")
audit_logger = logging.getLogger("vllm.edge.audit")

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cache-control", b"no-store"),
)
_HSTS: tuple[bytes, bytes] = (
    b"strict-transport-security",
    b"max-age=31536000; includeSubDomains",
)


class TokenBucket:
    """Classic token bucket. Not thread-safe; ASGI runs on one event loop."""

    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = rate
        self.capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._last = clock()

    def try_acquire(self) -> float:
        """Take one token. Returns 0.0 on success, else seconds to wait."""
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        return (1.0 - self._tokens) / self.rate


class _BodyTooLarge(Exception):
    pass


class _ClientDisconnected(Exception):
    pass


class EdgeGuardMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        config: EdgeConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.cfg = config
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}
        for t in config.tenants:
            tier = config.tiers[t.tier]
            if tier.rate_per_sec is not None:
                self._buckets[t.name] = TokenBucket(
                    tier.rate_per_sec, tier.burst, clock
                )
        self._tenant_inflight: dict[str, int] = {t.name: 0 for t in config.tenants}
        self._inflight = 0

    # ------------------------------------------------------------------ auth
    def _authenticate(self, headers: Headers) -> TenantConfig | None:
        value = headers.get("Authorization")
        if not value:
            return None
        scheme, _, param = value.partition(" ")
        if scheme.lower() != "bearer" or not param:
            return None
        digest = hashlib.sha256(param.strip().encode("utf-8")).digest()
        # Visit every tenant so timing does not reveal which entry matched.
        found: TenantConfig | None = None
        for tenant in self.cfg.tenants:
            if secrets.compare_digest(digest, tenant.key_digest):
                found = tenant
        return found

    # --------------------------------------------------------------- helpers
    def _shed_limit(self, tier: TierConfig) -> int | None:
        if self.cfg.max_inflight is None:
            return None
        return max(1, math.ceil(self.cfg.max_inflight * tier.shed_at))

    @staticmethod
    def _error(
        status: int, message: str, kind: str, headers: dict[str, str] | None = None
    ):
        return JSONResponse(
            {"error": {"message": message, "type": kind, "code": status}},
            status_code=status,
            headers=headers,
        )

    async def _read_body(self, receive: Receive, headers: Headers) -> bytes:
        limit = self.cfg.max_body_bytes
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    raise _BodyTooLarge
            except ValueError:
                raise _BodyTooLarge from None
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _ClientDisconnected
            if message["type"] != "http.request":
                continue
            part = message.get("body", b"")
            size += len(part)
            if size > limit:
                raise _BodyTooLarge
            chunks.append(part)
            if not message.get("more_body", False):
                return b"".join(chunks)

    def _inject_priority(
        self, path: str, headers: Headers, body: bytes, tier: TierConfig
    ) -> bytes:
        if not self.cfg.inject_priority or not body:
            return body
        if path.rstrip("/") not in self.cfg.priority_paths:
            return body
        if not headers.get("content-type", "").lower().startswith("application/json"):
            return body
        try:
            data: Any = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body  # let the API layer produce the 4xx
        if not isinstance(data, dict):
            return body
        data["priority"] = tier.priority
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    def _wrap_send(self, send: Send, guarded: bool, record: dict[str, Any]) -> Send:
        extra: list[tuple[bytes, bytes]] = []
        if self.cfg.security_headers:
            extra.extend(_SECURITY_HEADERS if guarded else _SECURITY_HEADERS[:3])
            if self.cfg.hsts:
                extra.append(_HSTS)

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                record["status"] = message["status"]
                if extra:
                    existing = {k.lower() for k, _ in message.get("headers", [])}
                    message = dict(message)
                    message["headers"] = list(message.get("headers", [])) + [
                        (k, v) for k, v in extra if k not in existing
                    ]
            await send(message)

        return wrapped

    def _audit(self, scope: Scope, record: dict[str, Any]) -> None:
        if not self.cfg.audit_log:
            return
        client = scope.get("client")
        audit_logger.info(
            "edge tenant=%s tier=%s method=%s path=%s status=%s outcome=%s "
            "duration_ms=%.1f client=%s",
            record.get("tenant", "-"),
            record.get("tier", "-"),
            scope.get("method", "WS"),
            scope.get("path", ""),
            record.get("status", "-"),
            record.get("outcome", "ok"),
            (self._clock() - record["start"]) * 1000.0,
            client[0] if client else "-",
        )

    # ------------------------------------------------------------------ ASGI
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path: str = scope["path"].removeprefix(scope.get("root_path", ""))
        is_admin_path = path.startswith(self.cfg.admin_prefixes)
        guarded = is_admin_path or path.startswith(self.cfg.guarded_prefixes)

        if scope["type"] == "websocket":
            await self._handle_websocket(scope, receive, send, guarded)
            return

        record: dict[str, Any] = {"start": self._clock()}
        send = self._wrap_send(send, guarded, record)

        # CORS preflight carries no credentials; leave it to the CORS layer.
        if not guarded or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        try:
            await self._handle_guarded(
                scope, receive, send, path, is_admin_path, record
            )
        finally:
            self._audit(scope, record)

    async def _handle_websocket(
        self, scope: Scope, receive: Receive, send: Send, guarded: bool
    ) -> None:
        if guarded and self._authenticate(Headers(scope=scope)) is None:
            audit_logger.info("edge ws auth_failed path=%s", scope.get("path", ""))
            # Consume the connect event, then refuse with policy-violation.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)

    async def _handle_guarded(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        path: str,
        is_admin_path: bool,
        record: dict[str, Any],
    ) -> None:
        headers = Headers(scope=scope)

        tenant = self._authenticate(headers)
        if tenant is None:
            record["outcome"] = "auth_failed"
            resp = self._error(
                401,
                "Unauthorized",
                "authentication_error",
                {"WWW-Authenticate": "Bearer"},
            )
            await resp(scope, receive, send)
            return
        tier = self.cfg.tiers[tenant.tier]
        record["tenant"], record["tier"] = tenant.name, tier.name

        if is_admin_path and not tenant.admin:
            record["outcome"] = "forbidden"
            await self._error(403, "Forbidden", "permission_error")(
                scope, receive, send
            )
            return

        bucket = self._buckets.get(tenant.name)
        if bucket is not None:
            wait = bucket.try_acquire()
            if wait > 0:
                record["outcome"] = "rate_limited"
                await self._error(
                    429,
                    "Rate limit exceeded",
                    "rate_limit_error",
                    {"Retry-After": str(max(1, math.ceil(wait)))},
                )(scope, receive, send)
                return

        shed_limit = self._shed_limit(tier)
        if shed_limit is not None and self._inflight >= shed_limit:
            record["outcome"] = "shed"
            await self._error(
                503,
                "Server busy, retry later",
                "overloaded_error",
                {"Retry-After": "1"},
            )(scope, receive, send)
            return

        in_flight = self._tenant_inflight[tenant.name]
        if tier.max_concurrency is not None and in_flight >= tier.max_concurrency:
            record["outcome"] = "concurrency_limited"
            await self._error(
                429,
                "Too many concurrent requests",
                "rate_limit_error",
                {"Retry-After": "1"},
            )(scope, receive, send)
            return

        self._inflight += 1
        self._tenant_inflight[tenant.name] += 1
        try:
            if scope.get("method") in _BODY_METHODS:
                try:
                    body = await self._read_body(receive, headers)
                except _BodyTooLarge:
                    record["outcome"] = "body_too_large"
                    await self._error(
                        413, "Request body too large", "invalid_request_error"
                    )(scope, receive, send)
                    return
                except _ClientDisconnected:
                    record["outcome"] = "client_disconnected"
                    return
                body = self._inject_priority(path, headers, body, tier)
                scope = dict(scope)
                scope["headers"] = [
                    (k, v) for k, v in scope["headers"] if k != b"content-length"
                ] + [(b"content-length", str(len(body)).encode("ascii"))]
                sent = False

                async def replay() -> Message:
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    return await receive()

                await self.app(scope, replay, send)
            else:
                await self.app(scope, receive, send)
        finally:
            self._inflight -= 1
            self._tenant_inflight[tenant.name] -= 1
