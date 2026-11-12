# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import inspect
from argparse import Namespace

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from vllm import envs
from vllm.logger import init_logger
from vllm.tasks import SupportedTask

from .log_response import log_response

logger = init_logger(__name__)


def init_entrypoints_middleware(
    args: Namespace,
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        from .authenticate import AuthenticationMiddleware

        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    if args.enable_request_id_headers:
        from .x_request_id import XRequestIdMiddleware

        app.add_middleware(XRequestIdMiddleware)

    if "generate" in supported_tasks:
        # Add scaling middleware to check for scaling state
        from vllm.entrypoints.serve.elastic_ep.middleware import ScalingMiddleware

        app.add_middleware(ScalingMiddleware)

    if "realtime" in supported_tasks:
        # Add WebSocket metrics middleware
        from vllm.entrypoints.speech_to_text.realtime.metrics import (
            WebSocketMetricsMiddleware,
        )

        app.add_middleware(WebSocketMetricsMiddleware)

    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning(
            "CAUTION: Enabling log response in the API Server. "
            "This can include sensitive information and should be "
            "avoided in production."
        )
        app.middleware("http")(log_response)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    # Edge guard is added last so it is the outermost layer: unauthenticated or
    # over-limit requests are rejected before any other middleware runs.
    # An invalid config raises here and aborts start-up (fail closed).
    if envs.VLLM_EDGE_CONFIG:
        from vllm.edge import EdgeGuardMiddleware, load_edge_config

        edge_config = load_edge_config(envs.VLLM_EDGE_CONFIG)
        if tokens:
            logger.warning(
                "Both --api-key/VLLM_API_KEY and VLLM_EDGE_CONFIG are set. "
                "Requests must satisfy both; prefer edge tenants only."
            )
        policy = getattr(args, "scheduling_policy", None)
        if (
            edge_config.inject_priority
            and policy is not None
            and policy != "priority"
            and any(t.priority != 0 for t in edge_config.tiers.values())
        ):
            # vLLM rejects non-zero priorities unless priority scheduling is on.
            raise ValueError(
                "Edge config assigns non-zero tier priorities but the server "
                f"uses --scheduling-policy {policy}; start with "
                "--scheduling-policy priority or set inject_priority=false."
            )
        app.add_middleware(EdgeGuardMiddleware, config=edge_config)
        logger.info(
            "Edge guard enabled: %d tenants, %d tiers",
            len(edge_config.tenants),
            len(edge_config.tiers),
        )
