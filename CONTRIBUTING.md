# Contributing

Small, verifiable improvements are welcome. Start with a user problem, a minimal
reproduction and an observable acceptance criterion. Do not open a broad rewrite
before discussing its scope.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run enterprise-agent-doctor --mode demo
uv run enterprise-agent-eval --mode demo --output artifacts/my-demo-eval.json
```

Offline unit tests use fakes for external services. The opt-in integration test
uses only a randomly named namespace in a disposable OpenSearch cluster:

```bash
# Set EA_TEST_OPENSEARCH_URL to YOUR disposable cluster, never a production cluster.
uv run pytest tests/test_opensearch_integration.py -q
```

Optional browser QA (Node 22+, not needed to run the app):

```bash
npm ci
npx playwright install chromium
npm run test:browser
```

The browser script starts its own loopback demo server and stops it afterward.
Set `EA_PYTHON` to choose an interpreter; Windows can use `EA_BROWSER_CHANNEL=msedge`.
Screenshots are written under `artifacts/ui`. It tests desktop/mobile, evidence,
no-match queries, exports, and HTML/URL injection. It does not test every browser.

## Code boundaries

- `web/` handles HTTP/auth/presentation, not retrieval algorithms.
- `agent/` handles state, planning, policy, evidence and routing.
- `retrieval/`, `generation/`, `ingestion/` own their domain logic.
- `infrastructure/` owns external clients; inject fakes in tests.
- Demo data is clearly synthetic and independent of private corpus/model keys.

Keep imports side-effect free. Close owned clients. Treat model/provider payloads
and source documents as untrusted. Never infer permissions from retrieved text.
Use failure-closed contracts for security and evidence integrity. Add a regression
test before or alongside a bug fix. Explain any API/schema/ID migration.

## Good first contributions

1. A sanitized failure case with independent relevance labels.
2. Keyboard/screen-reader or mobile accessibility improvements.
3. A read-only connector with pagination, permission mapping and incomplete-fetch tests.
4. Missing test coverage for provider failures or permission revocation.

Please submit code and data you have the right to share under compatible terms.
Preserve upstream notices. Never attach customer records, keys, logs with tokens,
or manufactured benchmark improvements. Use [SECURITY.md](SECURITY.md) for
vulnerabilities, and follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

CI runs tests/package checks on Windows and Linux, a disposable OpenSearch test,
and a Chromium browser smoke test. GitHub CI results exist only after publishing
the repository and running the workflow; local tests are reported separately.
