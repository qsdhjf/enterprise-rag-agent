# Changelog

## 0.5.0 — 2026-09-17 (incremental indexing and resource controls)

- Build-plan previews show reusable/pending chunks, characters and batches before
  consent. Plan fingerprints reject intervening file/model/cache changes.
- Same-library unchanged-file vectors and completed batches survive explicit
  retries; partial indexes remain unavailable. Forced rebuild and owned index-only
  clearing are separate, explicit operations. No cross-owner/library cache.
- Successful batches commit individually, task UUIDs protect against stale
  workers, and v1/v2 migration is backed up before atomic schema-3 changes.
- Per-owner/global logical vector-byte quotas include atomic queued reservations;
  cancellation/failure/restart release reservations. Disk free-space checks guard
  uploads/indexing. These are not hard filesystem/WAL/backup quotas.
- Two upload slots, sixteen other body-handler slots, 15-second total read deadline,
  early Content-Length checks, and independent file/embedding worker threads.
- Same-name/same-bytes replacement of a ready file is a no-op, preserving its index.
- New fault/resource tests and browser checks verify no-call rebuild, incremental
  addition, scoped clearing and fail-before-model behavior. No paid-provider claims.

## 0.4.0 — 2026-09-17 (opt-in personal Agent)

- Uploaded-file adapters reuse the existing LangGraph planner, scoped policy,
  evidence judge, bounded follow-up retrieval and final citation response.
- Explicit, persisted semantic indexing jobs; BM25, exact cosine vectors, RRF
  and optional TEI reranking. File/model changes invalidate vectors; no fallback
  quality claims, no automatic paid replays after restart.
- Loopback-only, opt-in provider wizard, masked secrets and synthetic connection
  tests; saved provider configuration is private but not encrypted.
- Reproducible upload-fixture evaluation with labelled IDs, real parser, metrics,
  raw case results and an unfilled human answer-review template.
- Automatic private backup before SQLite schema 1 → 2 migration; v1 data retained.
- Regression tests and real-browser full-graph checks using loopback fake providers.
  Paid-provider quality, large-corpus performance and production security remain unverified.

## 0.3.0 — 2026-09-17 (personal-document workspace)

- Browser knowledge libraries, raw-body uploads, status polling, retries,
  replacement/deletion, source-location preview and per-library questions.
- UTF-8 TXT/MD, text PDF and DOCX extraction in a bounded subprocess; explicit
  warnings for missing OCR/layout support, format checks and storage quotas.
- Durable SQLite jobs with crash recovery, single-worker locking, revision guards
  and owner isolation. This is not conversation memory.
- Local BM25 excerpt answers; explicitly opt-in model generation with citation-ID
  validation and fresh source checks. No semantic fact-verification claim.
- Authenticated `workspace` mode without OpenSearch; local demo cookies restricted
  to loopback. Public deployments still require security hardening.
- Online SQLite backups, Windows background start/stop scripts and authenticated
  workspace Compose recipe. Remote Docker/image/CI verification remains separate.
- Real parser, ownership, lifecycle, browser, mobile and installed-wheel checks.
- Existing enterprise graph, hybrid retrieval and original indexes are unchanged.

Not added: OCR/Excel, shared-library roles/SSO, personal vector indexing, true
token streaming, provider-side cancellation, distributed workers or paid-model
quality claims. See [knowledge guide](docs/KNOWLEDGE.md).

## 0.2.0 — 2026-09-16 (local release candidate)

### Added

- Browser evidence workspace with mobile layout, citation excerpts, fact/gap
  assessments, retrieval summaries and JSON export.
- No-key/no-GPU offline BM25 sample mode, separate from the live Agent.
- Server-owned scoped bearer identities, initialization and diagnostics commands.
- Complete tenant snapshot publisher with preflight, index-count checks,
  write-blocking and atomic read-alias switching; old generations retained.
- Read-only folder and GitHub issue exports with explicit ACLs and atomic outputs.
- Reproducible document-level Hit/Recall/MRR/nDCG and four-method live ablation.
- Unit, integration and browser checks; Windows/Linux CI; release allowlist tooling.
- English/Chinese quickstart, deployment, security, attribution and contributor docs.

### Changed / migration required

- Chunk IDs now include tenant and source namespaces and use full SHA-256.
  Importers recalculate legacy IDs. Rebuild into a NEW managed alias rather than
  mixing old/new IDs in an existing index.
- Live HTTP startup requires an identity file. Existing unauthenticated clients
  need bearer tokens. The Docker image defaults to offline demo; live Compose
  explicitly requires a protected identity-file mount.
- Live API exports evidence only after a fresh index ACL/content recheck.
- API adds evidence, facts, activity and mode fields; old answer fields remain.
- Removed unused PostgreSQL/checkpointer dependencies; this release has no persistent memory/checkpoint implementation.

### Not claimed

No GitHub publication, public CI run, independent security audit, paid-model
end-to-end benchmark, production reliability claim or adoption/Star milestone.
See release verification notes for measurements actually performed locally.
