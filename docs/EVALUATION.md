# Reproducible evaluation / 如何测召回

## Smoke test is not a benchmark claim

```bash
uv run enterprise-agent-eval --mode demo --k 5 --output artifacts/demo-eval.json
```

This runs local lexical retrieval over **9 synthetic documents and 6 teaching
cases**. Easy known-positive questions deliberately verify plumbing. High scores
here do not establish enterprise retrieval quality. The sixth query has no gold
document and is excluded from ranking-metric means.

## Evaluate the real retrieval stack

Prepare JSONL using independently labelled relevant documents:

```json
{"case_id":"archive-001","query":"How should Atlas be archived?","relevant_document_ids":["confluence::file:archive.md","jira::ATLAS-42"]}
```

IDs use `source::document_id`, preventing cross-source name collisions. Existing
cases with `source` may supply bare document IDs; the loader qualifies them.
Case IDs and gold IDs must be unique. An empty gold list is allowed but must be
reviewed by a human against the full **authorized** corpus, not guessed from a miss.

Create a private JSON file containing the evaluation principal (tenant_id,
subject_id, roles, groups, authz_version). It contains no bearer token.

```bash
uv run enterprise-agent-eval --mode live --cases eval/retrieval_cases.jsonl --principal .local/eval-principal.json --k 5 --output artifacts/live-eval.json
```

Requirements: populated compatible hybrid index, embedding service, and ready
reranker with `RERANKER_REQUIRED=true`. This command calls configured embedding
and reranking providers and may incur cost; it does not call an answer LLM.

The evaluator pins one physical index generation, checks embedding metadata,
and verifies each labelled positive is present and authorized before measuring.
It compares the actual **BM25, vector, RRF, RRF + reranker** functions. There is
no LLM query rewrite in this ablation; it measures retrieval, not the entire
Agent/planner chain. Failure does not silently become a score or fallback rank.
Method order rotates across cases. Reports contain model/index configuration,
dataset SHA-256, UTC time, environment, per-case ranked IDs and latency.

## Metric definitions

Results are unique documents, not repeated chunks. At cutoff K:

| Metric | Meaning |
|---|---|
| Hit@K | At least one labelled relevant document appears |
| Recall@K | Number of retrieved labelled positives / all labelled positives |
| MRR@K | Reciprocal rank of the first relevant document; 0 for a miss |
| nDCG@K | Discounted binary relevance normalized by ideal ordering |
| p50 / p95 | Interpolated per-query retrieval latency; includes model/service calls in that method |

Means are macro-averages across cases with nonempty gold sets. Empty-gold cases
have null ranking metrics and a separate `no_result_rate_on_no_gold`; this is
**not answer rejection accuracy**. Vector search may return neighbors even when
none are relevant; the evidence judge, not a nonempty result set, decides answerability.

Recall depends on label completeness. K=5 cannot retrieve all 20 positives.
Latency on 12 cases is not a reliable production p95, and this harness does not
yet report statistical confidence intervals. Record hardware, concurrency, cold/
warm cache state, model revision and corpus/version outside the report as needed.

## Before reporting results publicly

Build a larger independent test set across the nine source categories, multi-hop,
time-sensitive, conflicting, unanswerable, prompt-injection and permission-denied
questions. Freeze train/tune/test splits and publish annotation rules, label
agreement, per-case results and bootstrap confidence intervals where appropriate.
Do not tune on the test questions. Compare against the same authorized corpus.

Separately evaluate answer citation precision, faithfulness, information coverage,
conflict detection, abstention, cost and end-to-end latency. Existing
`evaluation/answerability.py` and unit tests are starting points, not evidence of
production quality. Upstream benchmark IDs require the separately obtained
upstream corpus. This repository deliberately makes no invented Recall claim.
