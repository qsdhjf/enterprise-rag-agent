"""Reproducible upload-library retrieval evaluation on a labelled text fixture.

Uses the production parser/store/retriever in an isolated temporary directory.
Never touches a user's live library. External methods require explicit consent.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, model_validator

from enterprise_agent.cli.evaluate import Case, evaluate
from enterprise_agent.knowledge.profiles import ProfileConfig
from enterprise_agent.knowledge.service import KnowledgeService


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=180)
    text: str = Field(min_length=1, max_length=100000)


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=2000)
    relevant_document_ids: list[str]
    expected_facts: list[str] = Field(default_factory=list)


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    provenance: str
    synthetic: bool
    documents: list[Document] = Field(min_length=1, max_length=200)
    cases: list[Label] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def identifiers(self):
        ids = {d.id for d in self.documents}
        if len(ids) != len(self.documents) or len({d.name for d in self.documents}) != len(self.documents):
            raise ValueError("Document IDs and names must be unique")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        for c in self.cases:
            if len(set(c.relevant_document_ids)) != len(c.relevant_document_ids) or not set(c.relevant_document_ids) <= ids:
                raise ValueError("Gold labels must refer to unique existing document IDs")
        return self


def run_fixture(fixture, k=3, methods=("bm25",), providers=None):
    if not methods or not set(methods) <= {"bm25", "vector", "hybrid", "reranked"}:
        raise ValueError("Unknown/empty methods")
    if not 1 <= k <= 20:
        raise ValueError("k must be 1..20")
    with TemporaryDirectory(prefix="ea-evaluation-") as directory:
        service = KnowledgeService(Path(directory), start_worker=False)
        try:
            if providers:
                for kind in ("llm", "embedding", "reranker"):
                    service.update_profile(kind, getattr(providers, kind), service.profiles.snapshot().revision)
            library = service.store.create_library("evaluation", fixture.name)["id"]
            ids = {}
            for doc in fixture.documents:
                if not doc.name.endswith((".txt", ".md")):
                    raise ValueError("Evaluation fixture embeds UTF-8 .txt/.md text only")
                raw = doc.text.encode()
                item = service.store.put_document("evaluation", library, doc.name, raw)
                ids[item["id"]] = doc.id
                parsed = service.parse(service.store.claim())
                if "error" in parsed:
                    raise ValueError("Fixture could not be parsed")
                service.store.finish(item["id"], 1, parsed)
            if any(method != "bm25" for method in methods):
                profile = service.profiles.snapshot()
                if not profile.embedding:
                    raise ValueError("Embedding configuration required")
                if "reranked" in methods and not profile.reranker:
                    raise ValueError("Reranker configuration required")
                service.store.queue_vectors("evaluation", library, profile.embedding.fingerprint, profile.embedding.dimensions)
                service.build_vectors(service.store.claim_vectors())
                status = service.store.vector_status("evaluation", library, profile.embedding.fingerprint)
                if status["status"] != "ready":
                    raise ValueError(status["error"] or "Vector indexing failed")
            cases = [Case(case_id=c.case_id, query=c.query, relevant_document_ids=c.relevant_document_ids) for c in fixture.cases]
            def method(strategy):
                def search(case):
                    with service.retriever("evaluation", library, strategy) as retriever:
                        # Measure the actual top-k CHUNK budget used by the product,
                        # then deduplicate document IDs; do not backfill extra hits.
                        return [ids[r["document_id"]] for r in retriever.search(case.query, k, strategy)]
                return search
            result = evaluate(cases, {name: method(name) for name in methods}, k)
            return {"schema_version": 1, "package_version": "0.5.0", "fixture": fixture.name,
                "synthetic": fixture.synthetic, "provenance": fixture.provenance,
                "fixture_sha256": hashlib.sha256(fixture.model_dump_json().encode()).hexdigest(),
                "timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "documents": len(fixture.documents), "cases_count": len(cases), "k": k,
                "configuration": {"ranking": "top-k chunks -> unique document IDs; no backfill", "candidate_pool": 20,
                    "rrf_constant": 60, "chunk_characters": 900, "overlap_characters": 200,
                    "embedding_model": providers.embedding.model if providers and providers.embedding else None,
                    "embedding_dimensions": providers.embedding.dimensions if providers and providers.embedding else None,
                    "reranker_model": providers.reranker.model if providers and providers.reranker else None},
                "limitations": "Retrieval only; not answer correctness. Synthetic scores are plumbing checks, not production quality. No-gold cases are excluded from ranking averages; no_result_rate is NOT a refusal-quality metric. Timings exclude indexing and include per-query client setup. Labels require independent human review.",
                "answer_review_template": [{"case_id": c.case_id, "question": c.query, "expected_facts": c.expected_facts,
                    "answer": None, "citations_supported": None, "facts_correct": None, "should_refuse": not c.relevant_document_ids,
                    "reviewer": None} for c in fixture.cases], **result}
        finally:
            service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Labelled JSON fixture; defaults to bundled synthetic examples")
    parser.add_argument("--methods", nargs="+", choices=("bm25", "vector", "hybrid", "reranked"), default=["bm25"])
    parser.add_argument("--providers", type=Path, help="Private provider-settings.private.json; never exported")
    parser.add_argument("--allow-external", action="store_true", help="Consent to sending fixture texts/questions to paid embedding/reranking services")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help="New private report file; never overwrites")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    external = any(method != "bm25" for method in args.methods)
    if external and (not args.allow_external or not args.providers):
        parser.error("External methods require --providers and --allow-external (may incur charges)")
    raw = args.fixture.read_text(encoding="utf-8") if args.fixture else files("enterprise_agent.knowledge").joinpath("eval_fixture.json").read_text(encoding="utf-8")
    try:
        fixture = Fixture.model_validate_json(raw)
        providers = ProfileConfig.model_validate_json(args.providers.read_text(encoding="utf-8")) if external else None
        report = run_fixture(fixture, args.k, tuple(dict.fromkeys(args.methods)), providers)
    except Exception as error:
        # Avoid provider error bodies or validation inputs containing secrets.
        parser.error(f"Evaluation failed ({type(error).__name__}); check fixture/configuration privately. No partial scores reported.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"synthetic": report["synthetic"], "summary": report["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
