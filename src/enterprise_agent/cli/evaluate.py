"""Reproducible offline smoke evaluation or four-method live retrieval ablation."""

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from statistics import mean
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enterprise_agent.agent.contracts import SourceName
from enterprise_agent.evaluation.metrics import percentile, retrieval_metrics


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=2000)
    source: SourceName | None = None
    relevant_document_ids: list[str]

    @field_validator("relevant_document_ids")
    @classmethod
    def nonempty_ids(cls, value):
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError("Gold IDs must be non-empty and unique")
        return value


def load_cases_text(text: str) -> list[Case]:
    cases = [Case.model_validate_json(line) for line in text.splitlines() if line.strip()]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Cases must be non-empty with unique case_id values")
    for case in cases:
        qualified = []
        for item in case.relevant_document_ids:
            if "::" in item:
                qualified.append(item)
            elif case.source:
                qualified.append(f"{case.source}::{item}")
            else:
                raise ValueError("Gold IDs need source::document_id when case.source is absent")
        case.relevant_document_ids = qualified
    return cases


def evaluate(cases: list[Case], methods: dict, k: int) -> dict:
    rows = []
    for index, case in enumerate(cases):
        # Rotate execution order to reduce systematic warm-cache ordering bias.
        names = list(methods)
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            start = perf_counter()
            ranked = list(dict.fromkeys(methods[name](case)))[:k]
            latency = (perf_counter() - start) * 1000
            rows.append({"case_id": case.case_id, "method": name, "ranked_ids": ranked,
                         "relevant_ids": case.relevant_document_ids,
                         "latency_ms": round(latency, 3),
                         **retrieval_metrics(ranked, set(case.relevant_document_ids), k)})
    summaries = {}
    for name in methods:
        selected = [row for row in rows if row["method"] == name]
        summary = {"cases": len(selected)}
        for metric in ("hit", "recall", "mrr", "ndcg"):
            values = [row[metric] for row in selected if row[metric] is not None]
            summary[f"{metric}@{k}"] = mean(values) if values else None
        empty = [row["no_result"] for row in selected if not row["relevant_ids"]]
        summary["no_gold_cases"] = len(empty)
        summary["no_result_rate_on_no_gold"] = mean(empty) if empty else None
        summary["latency_p50_ms"] = percentile([row["latency_ms"] for row in selected], .5)
        summary["latency_p95_ms"] = percentile([row["latency_ms"] for row in selected], .95)
        summaries[name] = summary
    return {"summary": summaries, "cases": rows}


def live_methods(stack, principal_path: Path, k: int, cases: list[Case]):
    from enterprise_agent.agent.contracts import PrincipalContext
    from enterprise_agent.infrastructure.config import load_embedding_settings, load_reranker_settings
    from enterprise_agent.infrastructure.embedding_client import OpenAICompatibleEmbeddingClient
    from enterprise_agent.infrastructure.opensearch_client import create_opensearch_client, load_opensearch_settings
    from enterprise_agent.infrastructure.reranker_client import TEIRerankerClient
    from enterprise_agent.retrieval.access_control import RetrievalAccessContext, build_opensearch_search_filter
    from enterprise_agent.retrieval.keyword_search import search_chunks
    from enterprise_agent.retrieval.vector_search import vector_search_chunks
    from enterprise_agent.retrieval.hybrid_search import hybrid_search_chunks
    principal = PrincipalContext.model_validate_json(principal_path.read_text(encoding="utf-8"))
    access = RetrievalAccessContext.from_principal(principal)
    settings = load_opensearch_settings()
    client = create_opensearch_client(settings)
    stack.callback(client.close)
    # Pin a physical index so an alias update cannot change the corpus mid-eval.
    mappings = client.indices.get_mapping(index=settings.hybrid_index_name)
    if len(mappings) != 1:
        raise ValueError("Evaluation requires a single physical index/generation")
    index = next(iter(mappings))
    metadata = mappings[index]["mappings"].get("_meta", {})
    embedding_settings = load_embedding_settings()
    if (metadata.get("embedding_model") != embedding_settings.model
            or metadata.get("embedding_dimensions") != embedding_settings.dimensions):
        raise ValueError("Index embedding metadata does not match configured model/dimensions")
    embedding = OpenAICompatibleEmbeddingClient(embedding_settings)
    stack.callback(embedding.close)
    reranker = TEIRerankerClient(load_reranker_settings())
    stack.callback(reranker.close)
    if not reranker.required or not reranker.is_ready():
        raise ValueError("Live ablation requires a ready reranker and RERANKER_REQUIRED=true; no fallback scores")
    if k > reranker.max_candidates:
        raise ValueError("--k exceeds the configured reranker candidate limit")
    for case in cases:
        for gold in case.relevant_document_ids:
            source, document_id = gold.split("::", 1)
            if case.source and source != case.source:
                raise ValueError("Gold source conflicts with query source filter")
            filters = build_opensearch_search_filter(access, source)
            count = client.count(index=index, body={"query": {"bool": {"filter": [filters, {"term": {"document_id": document_id}}]}}})["count"]
            if not count:
                raise ValueError(f"Gold document missing or inaccessible: {gold}")
    def keys(hits):
        return [f"{hit.source}::{hit.document_id}" for hit in hits]
    common = dict(client=client, index_name=index, top_k=k, max_chunks_per_document=1, access_context=access)
    def hybrid(case, use_reranker):
        return keys([item.hit for item in hybrid_search_chunks(
            **common, embedding_client=embedding, lexical_query=case.query,
            semantic_query=case.query, source=case.source,
            reranker_client=reranker if use_reranker else None)])
    methods = {
        "bm25": lambda case: keys(search_chunks(**common, query=case.query, source=case.source)),
        "vector": lambda case: keys(vector_search_chunks(**common, embedding_client=embedding, query=case.query, source=case.source)),
        "rrf": lambda case: hybrid(case, False),
        "rrf_reranker": lambda case: hybrid(case, True),
    }
    return methods, {"physical_index": index, "index_metadata": metadata,
                     "embedding_model": embedding_settings.model, "reranker_model": reranker.model,
                     "query_rewrite": False, "max_chunks_per_document": 1,
                     "reranker_max_candidates": reranker.max_candidates,
                     "candidate_multiplier": 4, "rrf_k": 60}


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest document-level retrieval evaluation; no external services in demo mode")
    parser.add_argument("--mode", choices=("demo", "live"), default="demo")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--principal", type=Path, help="Live only: trusted evaluation identity JSON, not a token file")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="New report file (existing files are not overwritten)")
    args = parser.parse_args()
    if not 1 <= args.k <= 100:
        parser.error("--k must be between 1 and 100")
    if args.mode == "live" and (not args.cases or not args.principal):
        parser.error("live mode requires --cases and --principal")
    if args.output and args.output.exists():
        parser.error("output already exists; use a new report filename")
    text = (args.cases.read_text(encoding="utf-8") if args.cases else
            files("enterprise_agent.demo").joinpath("cases.jsonl").read_text(encoding="utf-8"))
    cases = load_cases_text(text)
    with ExitStack() as stack:
        if args.mode == "demo":
            from enterprise_agent.demo.runner import DemoRunner
            runner = DemoRunner()
            methods = {"demo_bm25": lambda case: [f"{doc['source']}::{doc['id']}" for _, doc in runner.search(case.query, 100)
                                                 if case.source is None or doc["source"] == case.source][:args.k]}
            metadata = {"corpus": "9 original synthetic demo documents",
                        "corpus_sha256": hashlib.sha256(files("enterprise_agent.demo").joinpath("corpus.json").read_bytes()).hexdigest()}
        else:
            methods, metadata = live_methods(stack, args.principal, args.k, cases)
        report = {"schema_version": 1, "mode": args.mode, "package_version": "0.5.0",
                  "timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                  "dataset_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  "k": args.k, "metadata": metadata,
                  "limitations": "Demo scores are smoke checks, not production quality. Recall uses labelled positives; no-gold cases are excluded from ranking means. This is retrieval, not answer faithfulness evaluation.",
                  **evaluate(cases, methods, args.k)}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
