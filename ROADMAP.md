# Roadmap: adoption earned through usefulness

A million Stars is an aspiration, not a promise or an engineering acceptance
criterion. Never buy stars, fabricate users, copy competitor claims or present
teaching-set scores as a benchmark.

## 0.2 — make the core independently testable

- [x] Offline first-run experience and an evidence-focused workspace.
- [x] Explicit demo/live boundary and scoped service-token identities.
- [x] Tenant/source-scoped IDs and fail-closed snapshot publication.
- [x] Two bounded read-only import paths and transparent limitations.
- [x] Ranking metrics, reproducible reports and contributor/CI scaffolding.
- [ ] Public repository, verified release, monitored security/community channels.
- [ ] Independent fresh-machine feedback from at least three outside users.

## Next: validate a narrow real problem

### 0.3 — bring personal files into the evidence workspace

- [x] Browser upload, private libraries, local retrieval and source preview.
- [x] TXT/MD/text-PDF/DOCX parsing with explicit limits, durable status and retries.
- [x] Replace/delete race protection, owner isolation and backup/restore instructions.
- [x] Optional model answer mode; no silent model/data-provider calls.
- [ ] OCR/Excel and measured large-corpus quality.
- [ ] Real externally labelled model-answer evaluation and independent users.

### 0.4 — optional personal Agent and verifiable configuration

- [x] Reuse the real Agent graph with per-library owned retrieval/resolution.
- [x] Explicit semantic index jobs, exact cosine + BM25/RRF and TEI reranking.
- [x] Loopback provider setup, masked keys and consent-based connection checks.
- [x] Isolated labelled-fixture retrieval evaluator; no production-score claims.
- [ ] Hard filesystem/container quotas and large-library ANN.
- [ ] Independently labelled answer-faithfulness/cost evaluation with actual providers.

### 0.5 — bounded resources and incremental indexing

- [x] Explicit build plans, unchanged-file reuse and successful-batch retry recovery.
- [x] Model/input fingerprints, stale-plan rejection and generation-safe cancellation.
- [x] Per-owner/global logical vector quotas, atomic reservations and free-space checks.
- [x] Upload concurrency/read deadlines; file parsing independent of embedding calls.
- [x] Owned usage display and clear-vector operation that preserves source documents.
- [ ] Provider-side exactly-once billing is not promised; monitor uncertain failures.

### Product validation

Pick one user group: teams investigating cross-tool project decisions and handoffs.
Observe where they fail today. Measure time to first useful answer, citation
verification time, repeat usage, setup failure rate and cost per successful task.
Prioritize those measures over adding another model framework.

- Native connector permission mapping and change/delete cursors.
- Larger independently labelled evaluation set with confidence intervals.
- End-to-end answer faithfulness, abstention, conflict and authorization tests.
- Explicit model/index deployment compatibility checks and rollback tooling.
- Token streaming/cancellation and accessible citation navigation.

## Later: reliability and community

- SSO/OIDC, revocation propagation, distributed quotas and operational audit logs.
- Durable run recovery only when real workflows require it; no automatic return
  of the previously removed multi-turn memory module.
- Measured performance at corpus scale, deployment recipes and observability.
- Connector SDK and contribution review process, reproducible public case studies.

## Release and growth loop

Publish one truthful short demo and a minimal installation guide. Invite users
with the chosen problem, reproduce their blockers, ship a small improvement and
publish measured before/after results. Keep issues triaged and ownership clear.
Milestones should be active deployments and trusted answers; Stars may follow,
but cannot substitute for either.
