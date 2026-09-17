# Release checklist

## Local candidate

```bash
uv sync --locked
uv run pytest -q
uv run enterprise-agent-doctor --mode demo
uv run enterprise-agent-eval --mode demo --output artifacts/release-demo-eval.json
npm ci
npx playwright install chromium
npm run test:browser
uv build
uv run python scripts/verify_release.py --output artifacts/enterprise-agent-0.3.0-source.zip
```

The source packager uses an explicit directory/file allowlist, rejects common
credential patterns, excludes `.env`, `.local`, `.git`, caches, model/index files,
personal notes and build output, and includes a file-hash manifest plus archive
SHA-256. It does not publish, create a repository or alter Git staging.
Pattern checks are not a guarantee of no secrets; manually review all outputs.

Verify wheel assets from a fresh environment, not only an editable checkout.
Run the opt-in integration test against a disposable OpenSearch. Check both
desktop and mobile screenshots and the failure paths. Record what was NOT tested.
The legacy parent-workspace corpus is not bundled.

## Before publication (requires owner decisions)

- Choose the actual GitHub owner/repository and confirm project name availability.
- Review license/provenance and model/dataset terms; keep upstream attribution.
- Review staged/untracked files; do not blindly commit the entire old workspace.
- Enable private vulnerability reporting and supply monitored security/community contacts.
- Run GitHub CI on the proposed public branch and pin reviewed workflow revisions.
- Inspect source archive, wheel and sdist for credentials/private data.
- Create a release with checksums and accurate limitations; no inflated benchmark claims.
- Publish a short reproducible demo and invite independent first-run testers.

No public registry or GitHub release is created automatically by this repository.
