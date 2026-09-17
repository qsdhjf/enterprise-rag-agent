"""Create a scoped bootstrap identity; only the SHA-256 token hash is persisted."""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import get_args

from enterprise_agent.agent.contracts import PrincipalContext, RetrievalPolicy, SourceName
from enterprise_agent.web.security import Identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a NEW live identity file and print its one-time bearer token")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--source", action="append", choices=[s for s in get_args(SourceName) if s != "uploaded_file"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tenant == "local-demo-tenant":
        parser.error("the legacy demo tenant is not allowed in live mode")
    token = secrets.token_urlsafe(32)
    identity = Identity(
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        principal=PrincipalContext(tenant_id=args.tenant, subject_id=args.subject,
                                   roles=frozenset({"employee"}), groups=frozenset(args.group), authz_version="v1"),
        retrieval_policy=RetrievalPolicy(policy_version="v1", allowed_sources=frozenset(args.source),
                                         allow_cross_source=len(set(args.source)) > 1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump([identity.model_dump(mode="json")], target, ensure_ascii=False, indent=2)
    print("Store this token in your password manager. It is shown once and not saved in the identity file:")
    print(token)
    print("Restart the live service after identity changes. Protect this file with OS permissions.")


if __name__ == "__main__":
    main()
