"""Server-owned bearer identities. No identity/ACL claims from request bodies."""

import hashlib
import hmac
import json
from pathlib import Path

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext, PrincipalContext, RetrievalPolicy, RunBudget,
)


class Identity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    principal: PrincipalContext
    retrieval_policy: RetrievalPolicy
    budget: RunBudget = Field(default_factory=RunBudget)


def load_identities(path: str) -> tuple[Identity, ...]:
    """Configuration contains token hashes, never plaintext API credentials."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Identity file must be a non-empty JSON array")
    identities = tuple(Identity.model_validate(entry) for entry in entries)
    if len({item.token_sha256 for item in identities}) != len(identities):
        raise ValueError("Duplicate token hashes")
    if any(item.principal.tenant_id == "local-demo-tenant" for item in identities):
        raise ValueError("Live identities must not use the legacy demo tenant")
    return identities


def authenticate(request: Request) -> AgentRuntimeContext:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token or len(token) > 4096:
        raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})
    digest = hashlib.sha256(token.encode()).hexdigest()
    identities = getattr(request.app.state, "identities", ())
    matched = None
    for identity in identities:
        if hmac.compare_digest(identity.token_sha256, digest):
            matched = identity
    if matched is None:
        raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
    return AgentRuntimeContext(
        request_id=request.state.request_id, principal=matched.principal,
        retrieval_policy=matched.retrieval_policy, budget=matched.budget,
    )
