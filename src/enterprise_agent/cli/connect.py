"""Read-only connectors. Export first, review ACLs, then explicitly publish."""

import argparse
import os
from pathlib import Path
from typing import get_args

from enterprise_agent.agent.contracts import SourceName
from enterprise_agent.ingestion.connectors import ConnectorACL, export_chunks, folder_documents, github_issue_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Export authorized data into ACL-tagged chunks; never changes an index")
    connector = parser.add_mutually_exclusive_group(required=True)
    connector.add_argument("--folder", type=Path, help="UTF-8 Markdown/text directory")
    connector.add_argument("--github", help="owner/repository; issue title/body only")
    parser.add_argument("--source", choices=[s for s in get_args(SourceName) if s != "uploaded_file"], default="confluence",
                        help="Folder import category; this does not create a native SaaS integration")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--acl-principal", action="append", required=True, help="Repeat: user:alice / group:engineering")
    parser.add_argument("--acl-version", default="acl-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acl = ConnectorACL(args.tenant, tuple(args.acl_principal), args.acl_version)
    docs = (folder_documents(args.folder, args.source, acl) if args.folder else
            github_issue_documents(args.github, acl, token=os.environ.get("GITHUB_TOKEN")))
    count = export_chunks(docs, args.output, args.tenant)
    print(f"Exported {count} chunks. Review the complete export and ACLs before snapshot publication.")


if __name__ == "__main__":
    main()
