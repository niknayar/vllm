# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate an API key and the config entry that authorizes it.

    python -m vllm.edge.keytool --name gateway --tier critical

The plaintext key is printed once to stdout; only its SHA-256 digest belongs
in the edge config. Keys are 256-bit random values, so an unsalted SHA-256
digest is sufficient (there is no low-entropy password to brute-force).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def new_key() -> str:
    return "vk-" + secrets.token_urlsafe(32)


def digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True, help="tenant name")
    p.add_argument("--tier", required=True, help="tier defined in the edge config")
    p.add_argument("--admin", action="store_true", help="allow admin endpoints")
    args = p.parse_args(argv)
    if not _NAME_RE.match(args.name) or not _NAME_RE.match(args.tier):
        print("name/tier must match [A-Za-z0-9_.-]{1,64}", file=sys.stderr)
        return 2

    key = new_key()
    entry = {"name": args.name, "key_sha256": digest(key), "tier": args.tier}
    if args.admin:
        entry["admin"] = True
    print(f"API key (shown once, store it in a secrets manager): {key}")
    print("Add this entry to the 'tenants' list of your edge config:")
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
