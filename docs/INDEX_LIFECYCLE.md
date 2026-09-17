# Safe index lifecycle / 安全更新索引

## Why plain upsert was insufficient

Content-derived chunk IDs change when text changes. Writing new chunks without
removing old ones makes obsolete text searchable. Hashing only document ID and
content also lets identically named documents across tenants/sources collide.

New chunk IDs hash an unambiguous JSON tuple of **version, tenant, source,
document ID, sequence, content**, using the full SHA-256 digest. Both keyword
and hybrid importers recalculate IDs after ACL normalization, including old JSONL.

**Do not upsert v2 IDs into an old index and expect stale IDs to disappear.**
Use a fresh managed alias. Legacy `enterprise-agent-ingest` remains an explicit
upsert utility for development; it does not synchronize updates/deletions.

## Publish a complete tenant snapshot

Prepare one JSONL containing ALL sources/documents that should remain in ONE
tenant's alias. A single source export is not a complete multi-source snapshot.

```bash
uv run enterprise-agent-snapshot --jsonl .local/all-chunks.jsonl --alias research-team-current --tenant research-team --confirm-full-snapshot
```

This creates a hybrid index with the configured embedding model. For a standalone
BM25 experiment only, add `--keyword-only` (not suitable for the live hybrid Agent).

The publisher:

1. Verifies alias ownership (one managed physical index, matching tenant).
2. Copies the input into a private temporary file; validates every row and ACL;
   rejects duplicate source/document/sequence positions and accidental empty input.
3. Creates a new versioned physical index and imports the immutable copy.
4. Requires all writes to succeed and the indexed count to match. Write-blocks it.
5. Rechecks the previous alias target and atomically swaps the read alias.

Readers see the old generation or the new generation, not a half-imported mixture.
Omitted documents and stale chunks no longer occur in the active snapshot.
No index is automatically deleted. Old and failed staging generations consume
storage until an operator reviews retention. A complete empty snapshot requires
the additional `--allow-empty` flag: it intentionally removes all active content.

Atomic alias operation follows [OpenSearch's aliases API](https://docs.opensearch.org/latest/api-reference/alias/aliases-api/).
This is **application reindexing**, not the OpenSearch backup Snapshot API.

## Concurrency and uncertain results

Run one publisher per alias at a time, across ALL hosts. Use your deployment
scheduler's lock. This alpha is not a distributed ingestion coordinator.
Existing-target removal requires `must_exist`; the publisher checks for changes
again before switching. The first creation still needs external serialization.

If the alias request times out or lacks acknowledgement, its outcome is unknown.
Inspect the current alias target before retrying. Do not assume failure implies
no switch. Errors report the staging index name; existing generations are retained.

## Migration and rollback

Existing physical indexes and unmanaged aliases are never replaced in-place.
Publish to a NEW alias, test it with the intended identity, then change
`OPENSEARCH_HYBRID_INDEX` and restart the app. Old indexes remain untouched.

Record the returned `index`, `previous_index`, `input_sha256` and model metadata.
To roll back, an operator can atomically move the alias to the recorded previous
generation after verifying tenant ownership, schema/model compatibility and ACLs.
**A rollback can restore revoked permissions**: do not restore an old snapshot
without reviewing its authorization state. No automatic rollback is performed.

Do not switch embedding models/dimensions during an alias update while readers
still use the old encoder. Deploy a matching application configuration together
with that migration. Use separate tenant aliases and externally serialize updates.
