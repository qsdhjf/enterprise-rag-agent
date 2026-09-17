# Local release verification

## 0.5.0 — 2026-09-17 incremental indexing and bounded resources

Local Windows / Python 3.12.13 checks; no production performance or paid-provider
quality claims. This release retains the previous Agent/retrieval capabilities.

| Check | Observed result |
|---|---|
| Full tests | **243 passed, 2 skipped**, one existing Starlette TestClient deprecation warning |
| New tests | 29 incremental/resource cases, including actual simultaneous quota reservations, stale-worker protection, migration rollback, slow/trickling bodies and independent parsing during a blocked embedding call |
| Incremental invariant | Synthetic three-chunk index → unchanged rebuild: zero new embedding requests; adding two chunks embeds only those two; deleting the original file reuses remaining vectors without another model request |
| Interrupted job | Synthetic 20-chunk run: first 16 persisted, second batch of 4 fails; explicit retry reuses 16 and only resends 4. The failed call may still have been billed; no exactly-once claim |
| Ownership/resource checks | No cross-library/owner reuse; read-only plan endpoint; stale-plan refusal; low-disk and quota rejection before network; cancellation/restart releases reservations |
| Migration | v1/v2 → v3 with private pre-migration backup; ready v2 vectors reused without reembedding. Injected migration failure rolls back DDL and preserves original vectors/version |
| Browser | All three headless Edge suites passed; the Agent suite additionally checks plan/consent, unchanged rebuild with no new calls, one-file incremental build, clearing vectors while keeping documents, and missing-index rejection before model calls |
| Packaged installation | Wheel/sdist verified (127 members each); new non-editable environment, pinned dependencies, upload/parser/no-op replacement/usage endpoint smoke test and bundled evaluator passed |
| Installed-wheel browser | Full Agent browser suite also run using the fresh wheel interpreter and loopback synthetic chat/embedding/TEI server, not editable source |

Browser model traffic used a local synthetic HTTP server; no user documents were
sent to paid providers. Desktop/mobile screenshots were inspected and horizontal
overflow assertions passed. The bundled teaching fixture keeps the historical
BM25 smoke score (0.875 on eight answerable questions); this is not an independent
retrieval benchmark or an answer-faithfulness measurement.

Limits are intentionally scoped: vector quotas count logical JSON payload plus
pending reservations, not physical SQLite/WAL/backups; free-space checks are not
a filesystem guarantee. Upload limits are per process, not per-user fair quotas.
The two skipped OpenSearch tests, Linux/remote CI, container build, large-corpus
ANN, independent user trials and actual-provider quality/cost remain unverified
in this turn. Existing enterprise Docker services were not restarted.

## Historical: 0.4.0 — 2026-09-17 personal Agent, setup and retrieval

Local Windows / Python 3.12.13 verification, not a production certification.

| Check | Observed result |
|---|---|
| Full tests | **214 passed, 2 skipped**, one existing Starlette TestClient deprecation warning |
| New tests | 30 cases cover owned-library graph, two-round missing-fact retrieval, enterprise-source denial, deleted evidence, vector invalidation/cancellation/restart, secret masking, endpoint changes, fixture evaluation and schema migration |
| Browser | All three headless Edge scripts passed: original workspace, uploaded-file lifecycle, and setup → explicit index build → hybrid + TEI → actual LangGraph → cited answer |
| Model calls | Browser used an actual loopback HTTP stub implementing compatible chat/embeddings/TEI responses; **no paid-provider or semantic-quality validation claimed** |
| UI | Desktop/mobile screenshots inspected; mobile horizontal-overflow assertions passed. Model consent resets after reload; saving settings/uploading do not call providers |
| Packaging | Wheel + sdist built and checked: 126 members each; setup script and evaluation fixture included |
| Clean install | New non-editable wheel environment with lock-constrained dependencies; sample answer, upload, real parser subprocess and bundled retrieval evaluator passed |
| Offline evaluation | 6 synthetic documents / 10 questions, top-3 chunks then document deduplication: BM25 Hit/Recall/MRR/nDCG 0.875 on 8 answerable questions; empty-result rate 0.5 on 2 no-gold questions. Teaching-set plumbing only, not enterprise quality or refusal quality |

The existing enterprise graph now provides the planner with the runtime's allowed
sources; personal retrieval is restricted to `uploaded_file`. Enterprise entry
points still use the original nine sources. Existing enterprise Docker services
were not restarted. The skipped OpenSearch tests remain opt-in; no fresh Linux,
Docker build, public GitHub CI, security audit or paid model benchmark is claimed.

SQLite schema 1 → 2 is backed up automatically before migration. Configuration
is private but not encrypted. Model configuration/connection tests/index creation
all have distinct explicit actions; no real user files were sent to providers
during this validation. The evaluation's answer-review fields intentionally remain
empty: answer faithfulness requires independent human labels and actual outputs.

## Historical: 0.3.0 — 2026-09-17 personal-document workspace

Validated locally on Windows / Python 3.12.13. This section supersedes only the
release version/counts; historical enterprise integration results below remain
dated evidence, not a claim that those services were re-tested today.

| Check | Observed result |
|---|---|
| Full test suite | **184 passed, 2 skipped**, one existing Starlette TestClient deprecation warning |
| New knowledge tests | 35 cases: ownership, origin/host/peer checks, uploads, limits, replacement/deletion, parser failures/retry, restart recovery, worker lock, backup roundtrip, model opt-in and citation contracts |
| Real parsing | Actual subprocess TXT and PDF extraction; DOCX text/locations, archive bomb/entity rejection; blank PDF failure |
| Model route | Controlled fake responses test opt-in, invalid references, deleted evidence and unknown usage; **no paid provider quality test** |
| Browser | Headless Microsoft Edge: sample questions plus personal library creation, file upload, background readiness, ask, citation preview, refresh persistence, replacement, deletion, retry and separate-browser isolation |
| UI safety/layout | Literal malicious HTML displayed without execution; original URL safety tests; desktop/mobile screenshots inspected, no horizontal overflow |
| Distribution | Wheel and sdist built; 120 entries each; required knowledge assets present |
| Clean install | New non-editable wheel environment; 61 dependency packages resolved; sample ask, uploaded-file ask and actual parser subprocess passed |
| Compose | Demo and authenticated workspace Compose configurations parsed successfully; new image build/run **not** claimed |
| Background launch | Windows start/stop scripts verified against this project's demo; 8010 returned version 0.3.0 after restart |

The two skipped tests require disposable OpenSearch; this change did not modify
the enterprise graph/search implementation or restart existing enterprise Docker
services. Local knowledge uses BM25, not embedding/reranker semantics. Browser
fixtures are synthetic and stored in disposable directories, not user libraries.

Remaining: real model quality/cost/latency evaluation, real-world PDF/Word corpus
coverage, OCR, Linux runtime verification, hostile-input isolation/security audit,
large-scale performance, public GitHub CI/release and independent external users.
No production readiness or complete feature parity with mature RAG products is
claimed. Details and operational limitations: [KNOWLEDGE.md](KNOWLEDGE.md).

## Historical: 0.2.0 — 2026-09-16

Version: 0.2.0 alpha. Environment: Windows, Python 3.12.13, uv 0.12.2,
Playwright 1.62.1, headless Microsoft Edge (Chromium), Docker Desktop,
disposable OpenSearch 3.7.0.

## Passed

| Check | Observed result |
|---|---|
| Full pytest run with disposable OpenSearch enabled | **151 passed**, 1 deprecation warning |
| Unit/offline coverage | Original 114 tests retained; new auth, limits, evidence, connector, metric and lifecycle tests added |
| Real search integration | Keyword AND vector snapshot imports, atomic alias replacement, removed-document exclusion, old-generation retention, alias reads and cross-tenant denial |
| Browser | Desktop/mobile layout, successful evidence answer, no-match query, JSON export, literal rendering of malicious HTML and rejection of JavaScript links |
| Browser review | Generated desktop homepage and full mobile result screenshots inspected; no horizontal overflow or overlapping content found |
| Python build | Wheel and source distribution produced; 111 entries each; required UI/sample assets present |
| Clean wheel install | Separate virtual environment, installed artifact rather than editable source; UI/assets and HTTP ask passed; 55 installed packages compatible |
| Offline diagnostics/evaluation | Doctor passed; six explicitly synthetic teaching cases ran; fingerprinted per-case JSON report produced |
| Docker Compose | Demo Compose parsed successfully |
| Release preflight | Allowlist/credential-pattern scan completed; private env/identity files and parent corpus excluded |

The real vector integration test uses a **fixed test embedding**, not BGE/model
inference. It verifies OpenSearch ingestion/filter plumbing, not semantic quality.
Demo ranking results are intentionally easy smoke tests and must not be quoted as
enterprise benchmark performance.

The warning is Starlette's notice that its httpx-backed TestClient is deprecated.
It did not cause a test failure; migration should be tracked with dependency updates.
The original parent workspace also produces a uv warning about its parent
pyproject metadata; this is outside the standalone Agent package.

## Not completed or not claimed

- Docker image build reached the base-image pull but failed because the host's
  configured registry mirror/proxy was unavailable (connection refused to the
  local proxy). Global Docker/proxy settings were NOT modified. Therefore a
  successful container-image build/run is **not** claimed. Python installation
  and the running demo are independent of this Docker issue.
- Real LLM/embedding/reranker end-to-end quality/cost evaluation has not been run
  for this release. Live ablation code exists, but there is no measured live
  Recall/faithfulness improvement to report.
- GitHub issue connector is covered with mocked HTTP/pagination failures; no
  private repository, source permission sync or live GitHub export was tested.
- GitHub Actions is configured, but no remote repository/CI run/release was created.
- No Linux fresh install, penetration test, large-corpus load test, independent
  annotation study or external-user adoption study is claimed.

## Isolation and handoff

Existing running enterprise services and the original knowledge index were not
restarted or modified. Search tests used a separate disposable container and
UUID-owned indexes. The previous PostgreSQL/checkpointer dependencies were removed
from this project's environment only after confirming no source/test imports;
this release does not bring back the removed memory feature.

The release ZIP includes source, tests, documentation, sample data, CI and a
file-hash manifest. Wheel/sdist are separate installable artifacts. Hash scans
are not a substitute for a manual security and license review before publication.
Use [RELEASE.md](RELEASE.md) for the remaining owner/publication checklist.
