"""Bounded read-only exports with explicit administrator-assigned ACLs.

These adapters do not infer native SaaS permissions. Only import a scope that
the configured principal is authorized to access, and assign a narrow ACL.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from enterprise_agent.ingestion.chunker import chunk_document
from enterprise_agent.ingestion.data_reader import EnterpriseDocument
from enterprise_agent.ingestion.jsonl_writer import write_chunks_jsonl
from enterprise_agent.retrieval.access_control import normalize_document_acl_metadata


@dataclass(frozen=True)
class ConnectorACL:
    tenant_id: str
    principals: tuple[str, ...]
    version: str = "acl-v1"

    def fields(self) -> dict:
        metadata = normalize_document_acl_metadata({
            "tenant_id": self.tenant_id, "visibility": "restricted",
            "acl_principals": self.principals, "acl_version": self.version,
        })
        return dict(tenant_id=metadata.tenant_id, visibility=metadata.visibility,
                    acl_principals=metadata.acl_principals, acl_version=metadata.acl_version)


def folder_documents(root: Path, source: str, acl: ConnectorACL, *,
                     max_files: int = 10000, max_bytes: int = 2_000_000):
    """UTF-8 .md/.txt only; ignore hidden dirs, reject symlinks and oversize input."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Folder connector requires a directory")
    fields = acl.fields()
    count = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith(".")
                         and not (Path(current) / name).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            if name.startswith(".") or path.suffix.lower() not in {".md", ".txt"}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Symlink/outside-root documents are not allowed")
            count += 1
            if count > max_files or path.stat().st_size > max_bytes:
                raise ValueError("Connector safety limit exceeded; export aborted, not truncated")
            with path.open("rb") as stream:
                raw = stream.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise ValueError("Document grew beyond connector size limit")
            content = raw.decode("utf-8-sig")
            relative = path.relative_to(root).as_posix()
            yield EnterpriseDocument(document_id=f"file:{relative}", source=source,
                                     title=path.stem, content=content, created_at=None,
                                     updated_at=None, original_location=None, **fields)


def github_issue_documents(repository: str, acl: ConnectorACL, *, token: str | None = None,
                           client: httpx.Client | None = None, max_pages: int = 50):
    """Export issue title/body (not PRs/comments/attachments); fail on incomplete fetch."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must be owner/name, not an arbitrary URL")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    fields = acl.fields()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    owned = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=False)
    try:
        for page in range(1, max_pages + 1):
            response = client.get(f"https://api.github.com/repos/{repository}/issues",
                                  params={"state": "all", "per_page": 100, "page": page,
                                          "sort": "created", "direction": "asc"}, headers=headers)
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                raise ValueError("GitHub response must be an array")
            for item in items:
                if "pull_request" in item:
                    continue
                number = item["number"]
                title = item["title"]
                if type(number) is not int or not isinstance(title, str):
                    raise ValueError("Invalid issue metadata")
                yield EnterpriseDocument(
                    document_id=f"github:{repository}:issue:{number}", source="github",
                    title=title, content=title + "\n\n" + (item.get("body") or ""),
                    created_at=item.get("created_at"), updated_at=item.get("updated_at"),
                    original_location=f"https://github.com/{repository}/issues/{number}", **fields,
                )
            if len(items) < 100:
                return
        raise ValueError("Pagination limit reached; export aborted rather than publishing a partial snapshot")
    finally:
        if owned:
            client.close()


def export_chunks(documents, output: Path, tenant_id: str) -> int:
    """Only expose a finished export; never overwrite an existing destination."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=".export-", suffix=".jsonl", dir=output.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        chunks = (chunk for document in documents for chunk in chunk_document(document))
        count = write_chunks_jsonl(chunks, temporary, overwrite=True, expected_tenant_id=tenant_id)
        # Same-filesystem link makes publication atomic AND refuses a competing destination.
        os.link(temporary, output)
        return count
    finally:
        temporary.unlink(missing_ok=True)
