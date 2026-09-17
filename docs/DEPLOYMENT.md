# Deployment / 部署指南

For browser file uploads and a simpler authenticated deployment without OpenSearch,
see [personal knowledge setup](KNOWLEDGE.md). `--mode workspace` is distinct from
the enterprise `live` graph; it uses private local BM25, with opt-in model answers.

## 1. Offline first

Use Python 3.12+, run `uv sync --locked --no-dev`, then:

```bash
uv run --no-sync enterprise-agent-serve --mode demo --port 8010
```

Open http://127.0.0.1:8010. Installation needs downloaded dependencies; runtime
needs no network, account, API key, GPU or search cluster. Port 8010 avoids the
historical live service on 8000. Demo never reads imported enterprise indexes.

## 2. Configure real providers

Copy `.env.example` to your private `.env` and configure:

| Variables | Purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | OpenAI-compatible scope/planner/judge/answer service |
| `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_DIMENSIONS` | Embedding endpoint; model and dimensions must match the index |
| `OPENSEARCH_URL`, `OPENSEARCH_HYBRID_INDEX` | Search cluster and managed read alias |
| `RERANKER_BASE_URL`, `RERANKER_MODEL`, `RERANKER_REQUIRED` | TEI-compatible reranker; `true` fails closed if unavailable |
| `AGENT_MAX_CONCURRENT` | Per-process in-flight request cap, default 2; not distributed rate limiting |

LLM and embedding calls can send questions and document content to the configured
provider. Confirm your data processing rules before using enterprise material.
Do not put keys in source code, screenshots, issues or evaluation reports.

The existing `infra/opensearch/compose.yaml` is a **local development** topology:
OpenSearch security disabled, GPU reranker and a host embedding service. It is
not a production deployment template. Do not expose its search ports publicly.

## 3. Create a trusted identity

Example: Alice may query Confluence/GitHub documents that ACLs grant to the
engineering group. The token is generated randomly and printed once; only its
SHA-256 hash is saved. Run this yourself in a private terminal:

```bash
uv run enterprise-agent-identity --tenant research-team --subject alice --group engineering --source confluence --source github --output .local/identities.json
```

Save the printed token in a password manager. Protect `.local/identities.json`
with OS permissions (especially on Windows). Never commit or publish it.
To add identities, edit the JSON array on the server; each entry contains
`token_sha256`, `principal`, `retrieval_policy` and optional `budget`.
Use distinct tokens per principal. Restart after identity/permission changes;
this alpha does not hot-reload tokens or synchronize an identity provider.

Production identities cannot use `local-demo-tenant`. Client request bodies
cannot supply tenants, roles, groups or source permissions. This is scoped
bearer authentication, **not SSO/OIDC** or a complete user-management system.

## 4. Import and publish

Use [connectors](CONNECTORS.md) to export permitted files with explicit ACLs.
Combine all sources for ONE tenant alias, review the export, then follow
[index lifecycle](INDEX_LIFECYCLE.md). Point `OPENSEARCH_HYBRID_INDEX` at the
published alias. Do not point a live hybrid Agent at a keyword-only index.

## 5. Start and verify

```bash
uv run enterprise-agent-doctor --mode live --identity-file .local/identities.json
uv run enterprise-agent-serve --mode live --identity-file .local/identities.json --port 8000
```

Doctor validates local configuration; it does not prove remote availability or
call paid models. Open the workspace, enter your token, and ask a question.
The UI keeps the token only in page memory, never localStorage or exports.

HTTP contract:

```text
GET  /                    Workspace
GET  /api/config          Mode/version/examples, no secrets
GET  /health/live         Public process liveness
GET  /health/ready        Dependency readiness; bearer required in live mode
POST /ask                 {"question":"..."}; bearer required in live mode
GET  /docs                OpenAPI documentation
```

Send `Authorization: Bearer YOUR_PRIVATE_TOKEN` on authenticated endpoints.
Readiness checks search/index/reranker, not paid LLM or embedding inference.
`401` means missing/invalid token, `413` oversized body, `422` invalid request,
`429` all request slots occupied, `503` dependency/evidence verification failure.
Use the request ID to correlate logs. The server intentionally withholds internal
error details. Responses and evidence use `Cache-Control: no-store`.

## Containers and public exposure

The image defaults to offline demo. Live containers must explicitly run
`enterprise-agent-serve --mode live --host 0.0.0.0 --identity-file /run/agent/identities.json`
and mount a protected identity file read-only. The historical Compose file now
expects `.local/identities.json`; it does not create tokens for you.

Before any public deployment: TLS reverse proxy, header/body/connection timeouts,
per-user quotas, private authenticated OpenSearch, monitoring, backup/restore,
credential rotation, prompt-injection testing, and an independent security review.
Process-local concurrency is not a denial-of-service defence. Do not share the
same unrestricted service token among all users.

## Troubleshooting

- Port in use: choose `--port 8011`; Docker demo accepts `AGENT_DEMO_PORT`.
- Old API worked without a token: this is an intentional security-breaking change.
- Breakpoints lead to budget failure: wall-clock budget still runs while paused.
  Use fake-client unit/tutorial scripts, not production identity budgets, to learn.
- Live evidence disappears during a request: final fresh ACL/content checks fail
  closed. Inspect the request ID and retry against the new index generation.
- Missing source data: benchmark corpus is not in the release; demo data is separate.
