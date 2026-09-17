# Read-only data connectors

Only import data you are authorized to use. Export is separate from publishing:
no connector automatically changes the live search index.

## Local UTF-8 Markdown / text

```bash
uv run enterprise-agent-connect --folder ./knowledge --source confluence --tenant research-team --acl-principal group:engineering --output .local/folder-chunks.jsonl
```

`--source` selects the existing source category for these exported documents; it
does not claim a live Confluence connection. Only `.md` and `.txt` are read.
Hidden directories/files are skipped; symlink files are rejected. Defaults cap
files at 10,000 and each file at 2 MB. Exceeding limits aborts, rather than silently
producing a truncated snapshot. Use a dedicated directory per connector scope:
relative paths form document IDs, so separate folders with identical paths need
distinct categories or must be combined under one common root.

## GitHub issues

```bash
uv run enterprise-agent-connect --github your-org/your-repository --tenant research-team --acl-principal group:engineering --output .local/github-chunks.jsonl
```

Set `GITHUB_TOKEN` in your terminal/secret manager if the repository requires it.
Use the minimum read-only issues permission. The fixed `api.github.com` endpoint
is used; arbitrary hosts and redirects are not followed. Each page requests
100 entries, at most 50 pages. HTTP errors/rate limits/page-limit exhaustion
abort export; the final JSONL is not published partially. Retry later explicitly.

Includes issue title/body and timestamps. Excludes pull requests, comments,
attachments, code and Discussions. The API can change during pagination; this
is not a source-side point-in-time snapshot or exactly-once incremental sync.
For a critical import, freeze the scope and review an export before publishing.
See the [official issue API](https://docs.github.com/en/rest/issues/issues#list-repository-issues).

## ACL and publication contract

Both connectors require a tenant and one or more `--acl-principal` values.
Examples: `user:alice`, `group:engineering`, `role:auditor`. Visibility is
`restricted`, never implicitly public. `--acl-version` defaults to `acl-v1`;
increment it when changing access policy and publish a new snapshot.

These are **administrator-assigned export ACLs**, not synchronized permissions
from GitHub/Confluence. Do not map a mixed-permission corpus to one broad group.
Native permissions, refresh scheduling and incremental change cursors are roadmap
work, not silently inferred capabilities.

Exports refuse to overwrite files. A same-filesystem temporary file is hard-linked
to the destination only after full success (NTFS/ext4 supported). Filesystems
without hard-link support fail safely; use a local disk for the export.
Review/merge the complete set for one tenant alias before [snapshot publication](INDEX_LIFECYCLE.md).
