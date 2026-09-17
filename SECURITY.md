# Security policy and deployment boundary

This is an alpha release, not a security certification. No independent penetration
test or production compliance assessment is claimed.

## Implemented controls

- Live/workspace startup requires server-owned hashed bearer identities; raw request bodies
  cannot choose identity, tenant or roles. Tokens are compared by digest.
- Tenant/ACL filters run during retrieval, and returned document metadata is
  rechecked. Live API evidence is freshly read and checked before response export.
- Scoped source policies, per-run budgets, per-process concurrency limits,
  16 KiB JSON body limit, 8 MiB raw-body document-upload exception,
  2,000-character question limit and sanitized service errors.
- Request bodies have a 15-second total read deadline, two concurrent upload
  slots and sixteen other mutating-request slots per process. Upload slots remain
  occupied until the handler releases the body. These are not per-user rate limits
  or a replacement for reverse-proxy connection/size/rate limits.
- UI uses text nodes, not untrusted HTML; source links allow HTTP(S) only;
  tokens are not written to localStorage or exported reports.
- Responses are non-cacheable. The workspace uses a same-origin content security
  policy. No analytics or telemetry is added.
- Snapshot IDs are tenant/source-scoped; failed staging does not replace the
  active generation. Old indexes are not automatically deleted.
- Personal libraries use owner-scoped SQLite queries and fresh revision/hash
  checks. Replacing/deleting files removes old chunks and their vectors.
- Parsing runs in a subprocess with time/resource limits, restricted inherited
  environment and format/expansion checks. This is containment, NOT a secure sandbox.
- Provider setup requires explicit `--enable-setup`, real loopback peer, loopback
  Host, same-origin checks and a valid knowledge session/identity. Forwarded
  headers cannot make a remote connection local. Never reverse-proxy this setup
  surface as a public administration interface.
- Saving configuration makes no provider call. Connection tests require consent
  and use synthetic text. Uploading never automatically embeds files; indexing
  requires consent, persists progress and is not automatically replayed after a crash.
- Remote provider URLs require HTTPS. API keys remain server-side; setup errors
  and responses do not echo them. A blank key is reused only for the same endpoint.
  Provider redirects are disabled for personal-workspace HTTP clients.

## Known limitations / threat model

Assume index administrators, identity-file owners and application operators are
trusted. An attacker controlling those can change data or grants. Imported ACLs
are administrator-assigned and are not native SaaS permission synchronization.
Identity changes take effect on process restart; index ACL changes take effect
when the new index/metadata is visible. In-flight requests cannot offer instantaneous
revocation guarantees across every external model call.

Model prompts cannot guarantee protection from prompt injection or hallucination.
Read-only tools limit side effects, but a compromised/malicious model can still
produce misleading text. Human verification is required for consequential decisions.

There is no OIDC/SSO, per-user billing, distributed quota system, encrypted evidence
store, durable audit log, automated key rotation or production incident response
service. Per-process concurrency/body limits are not complete DoS protection.
Use TLS, a hardened reverse proxy, authentication at all data services, narrow
network rules, input/connection timeouts, quotas, backups and monitoring.

The bundled demo answers use only synthetic samples. The same application ALSO
offers persistent personal-file libraries to loopback users. Those files can be
sensitive: do not describe the whole server as sample-only. Cookie sessions are
not accounts; deleting a cookie loses access without deleting server data. Use
authenticated workspace mode for stable identities; do not publicly proxy demo.
Neither sample mode nor uploaded libraries can search enterprise indexes.
Development OpenSearch Compose
disables its security plugin and must remain private. Provider keys, identity
files and questions/answers must not appear in public diagnostics.

SQLite files, automatic pre-migration backups and provider-settings.private.json
are sensitive. The provider file is NOT encrypted; OS permissions and encrypted
backups are operator responsibilities (Windows chmod is not a full ACL policy).
Deletion is not forensic erasure: backups, WAL and interrupted parser temporary
files can retain copies. Vector jobs are bounded to 2,000 chunks per library but
only a small-library storage model is supported. Logical UTF-8 vector payloads
have default limits of 128 MiB per owner and 512 MiB globally, including pending
reservations. Transactions reserve space before external calls; cancellation,
failure, clearing and restart release reservations. Free-space checks preserve a
default 256 MiB floor plus write headroom. These are NOT hard filesystem quotas:
SQLite pages, indexes, WAL, migration backups and other processes use extra space.
Monitor disk use and retain filesystem/container quotas. Exact cosine search in
SQLite/Python is intended for small libraries, not a multi-tenant hosted service.

Opt-in Agent queries send questions and selected excerpts to the LLM; hybrid
queries send questions to embeddings, and reranking sends candidate excerpts to
TEI. Index construction sends only selected-library chunks needing new vectors
(including filenames), unless a forced rebuild is selected. Reuse requires the
same endpoint/model/dimensions, document revision and exact embedding-input hash;
there is no cross-library/owner embedding cache. After provider/model changes,
the user must review and confirm a new plan. Read provider data handling policies first.
In-flight requests cannot be recalled after they leave this process. UI Stop
only stops waiting; index Cancel stops later batches, not provider billing.
Successful batches persist for explicit retry, but a paid response lost before
commit (including an ambiguous timeout) can be billed again. No exactly-once
billing guarantee is claimed. Partial indexes cannot be queried as ready indexes.

## Reporting

Do not post exploitable details, tokens or private data in public issues. Once
the repository is public, the owner must enable GitHub private vulnerability
reporting and publish a monitored security contact **before public service
exposure**. If that channel is not configured, contact the repository owner
through an established private channel; do not assume an unmonitored address.

Reports should include affected version, sanitized reproduction, impact and
suggested mitigation. No response SLA is promised until a maintenance owner has
explicitly committed to one. For a suspected exposure: isolate the service,
revoke/rotate affected credentials, restart identity consumers, preserve private
logs, verify index ACLs and assess provider-side data handling.
