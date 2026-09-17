"""Explicit, tenant-scoped full snapshot publication. Never run implicitly."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from enterprise_agent.ingestion.opensearch.snapshot import publish_snapshot
from enterprise_agent.infrastructure.config import load_embedding_settings
from enterprise_agent.infrastructure.embedding_client import OpenAICompatibleEmbeddingClient
from enterprise_agent.infrastructure.opensearch_client import create_opensearch_client, load_opensearch_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a complete tenant snapshot into a NEW index; retain old indexes")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--confirm-full-snapshot", action="store_true", required=True,
                        help="Input includes ALL sources/documents for this alias; omitted documents disappear")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--keyword-only", action="store_true", help="BM25 only; not suitable for the live hybrid Agent")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    with ExitStack() as stack:
        client = create_opensearch_client(load_opensearch_settings())
        stack.callback(client.close)
        options = {}
        if not args.keyword_only:
            settings = load_embedding_settings()
            embedding = OpenAICompatibleEmbeddingClient(settings)
            stack.callback(embedding.close)
            options = dict(embedding_client=embedding, embedding_model=settings.model, dimensions=settings.dimensions)
        result = publish_snapshot(client, args.alias, args.jsonl, args.tenant,
                                  confirm_full_snapshot=args.confirm_full_snapshot,
                                  allow_empty=args.allow_empty, batch_size=args.batch_size, **options)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
