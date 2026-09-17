"""Versioned, unambiguous IDs shared by chunking and both index writers."""

import hashlib
import json
from typing import Any


def chunk_identity(tenant_id: str | None, source: str, document_id: str,
                   sequence: int, text: str) -> str:
    """Namespace content by tenant AND connector; delimiters cannot collide."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be non-empty")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be non-empty")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be non-empty")
    payload = json.dumps(["chunk-v2", tenant_id, source, document_id, sequence, text],
                         ensure_ascii=False, separators=(",", ":"))
    return "chunk_v2_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_chunk_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Never trust an imported _id: re-key after ACL normalization."""
    result = dict(record)
    result["chunk_id"] = chunk_identity(
        result["tenant_id"], result.get("source"), result.get("document_id"),
        result.get("sequence"), result.get("text"),
    )
    return result
