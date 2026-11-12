# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load ``vllm.edge`` without importing the heavy ``vllm`` package.

The edge toolkit only depends on the standard library and Starlette, so its
tests can run on a machine without torch / a built vLLM:

    pytest tests/edge --confcutdir=tests/edge
"""

import importlib.util
import sys
from pathlib import Path

_EDGE_DIR = Path(__file__).resolve().parents[2] / "vllm" / "edge"


def _load() -> None:
    if "vllm_edge" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "vllm_edge",
        _EDGE_DIR / "__init__.py",
        submodule_search_locations=[str(_EDGE_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm_edge"] = module
    spec.loader.exec_module(module)


_load()
