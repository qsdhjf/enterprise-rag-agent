"""Publish a complete tenant snapshot without deleting the live generation.

Single-writer operation: serialize publishers for each alias. Old generations
are retained for explicit rollback and retention review.
"""

import hashlib
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from enterprise_agent.ingestion.opensearch.hybrid_index import build_hybrid_index_body
from enterprise_agent.ingestion.opensearch.hybrid_ingestion import bulk_index_hybrid_jsonl, iter_chunk_batches
from enterprise_agent.ingestion.opensearch.index import CHUNK_INDEX_BODY
from enterprise_agent.ingestion.opensearch.ingestion import bulk_index_jsonl


@dataclass(frozen=True)
class SnapshotResult:
    alias: str
    index: str
    previous_index: str | None
    chunks: int
    input_sha256: str


def _current_index(client, alias: str, tenant_id: str) -> str | None:
    if not client.indices.exists_alias(name=alias):
        if client.indices.exists(index=alias):
            raise ValueError("Target is a physical index; choose a NEW managed alias")
        return None
    targets = client.indices.get_alias(name=alias)
    if len(targets) != 1:
        raise ValueError("Snapshot alias must resolve to exactly one index")
    name = next(iter(targets))
    metadata = client.indices.get_mapping(index=name)[name]["mappings"].get("_meta", {})
    if metadata.get("snapshot_tenant") != tenant_id or metadata.get("snapshot_alias") != alias:
        raise ValueError("Refusing to replace an unmanaged or different-tenant alias")
    return name


def publish_snapshot(client, alias: str, jsonl_path: Path, tenant_id: str, *,
                     confirm_full_snapshot: bool, allow_empty: bool = False,
                     embedding_client=None, embedding_model: str | None = None,
                     dimensions: int | None = None, batch_size: int = 8) -> SnapshotResult:
    """Preflight -> stage -> verify -> write-block -> atomic alias switch.

    Input MUST contain every source/document intended to remain in this alias.
    Omitted documents disappear from the new generation. On staging failure
    the live alias is untouched. Alias timeouts have unknown outcomes: inspect
    the alias before retrying. This function never deletes an index.
    """
    if not confirm_full_snapshot:
        raise ValueError("Full tenant snapshot acknowledgement required")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{1,99}", alias):
        raise ValueError("Alias must be 2-100 lowercase letters/digits/_/- and start with a letter")
    if not tenant_id.strip() or batch_size < 1:
        raise ValueError("Tenant and positive batch size required")
    previous = _current_index(client, alias, tenant_id)
    with TemporaryDirectory(prefix="enterprise-snapshot-") as directory:
        frozen = Path(directory) / "input.jsonl"
        shutil.copyfile(jsonl_path, frozen)
        with frozen.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        count = 0
        positions = set()
        for batch in iter_chunk_batches(frozen, batch_size, expected_tenant_id=tenant_id):
            for record in batch:
                position = (record["source"], record["document_id"], record["sequence"])
                if position in positions:
                    raise ValueError("Duplicate document/sequence in complete snapshot")
                positions.add(position)
                count += 1
        if not count and not allow_empty:
            raise ValueError("Empty snapshot rejected; explicitly allow only for intentional removal")
        body = (build_hybrid_index_body(embedding_model or "", dimensions or 0)
                if embedding_client is not None else deepcopy(CHUNK_INDEX_BODY))
        body["mappings"].setdefault("_meta", {}).update({
            "snapshot_tenant": tenant_id, "snapshot_alias": alias,
            "snapshot_sha256": digest, "chunk_identity_version": 2,
        })
        staged = f"{alias}--{uuid4().hex}"
        client.indices.create(index=staged, body=body)
        try:
            if embedding_client is not None:
                result = bulk_index_hybrid_jsonl(client, staged, frozen, embedding_client,
                                                 batch_size, expected_tenant_id=tenant_id)
            else:
                result = bulk_index_jsonl(client, staged, frozen, batch_size,
                                          expected_tenant_id=tenant_id)
            if result.failed or result.succeeded != count:
                raise RuntimeError("Staged writes are incomplete")
            client.indices.put_settings(index=staged, body={"index.blocks.write": True})
            if client.count(index=staged)["count"] != count:
                raise RuntimeError("Staged index count mismatch")
            if _current_index(client, alias, tenant_id) != previous:
                raise RuntimeError("Alias changed concurrently; serialize publishers")
            actions = []
            if previous is not None:
                actions.append({"remove": {"index": previous, "alias": alias, "must_exist": True}})
            actions.append({"add": {"index": staged, "alias": alias, "is_write_index": False}})
            response = client.indices.update_aliases(body={"actions": actions})
            if not response.get("acknowledged"):
                raise RuntimeError("Alias not acknowledged; inspect before retry")
        except Exception as error:
            raise RuntimeError(f"Snapshot incomplete: staged={staged}. Inspect alias; no indexes deleted.") from error
    return SnapshotResult(alias, staged, previous, count, digest)
