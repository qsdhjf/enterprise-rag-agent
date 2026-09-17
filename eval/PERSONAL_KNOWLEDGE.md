# Personal-document evaluation protocol (not measured quality results)

The new personal workspace must earn its own evaluation. Do not reuse the nine
sample-document scores as evidence for arbitrary uploaded files, or describe
an offline BM25 result as an Agent answer. Since 0.4, personal libraries can opt
into the shared graph and hybrid retrieval, but still need their own evaluation.

## Runnable baseline and comparison

`uv run enterprise-agent-eval-knowledge --output artifacts/personal-eval.json`
uses 6 original synthetic documents and 10 manually authored teaching questions.
It exercises the actual parser subprocess, isolated SQLite store and retriever.
The packaged `knowledge/eval_fixture.json` defines the input format. Gold IDs must
exist and be unique; duplicate case IDs are rejected. It does not touch live files.

For an authorized independently labelled text snapshot, supply `--fixture`,
`--methods bm25 vector hybrid reranked`, `--providers` pointing at the private
provider configuration and `--allow-external`. All non-BM25 methods may send text
and incur charges; failure aborts instead of silently reporting fallback scores.
No external model comparison has been claimed from the synthetic unit tests.

Results include fingerprint, model names/dimensions (no keys or endpoint URLs),
rotating method order, per-case lists, metrics, latency and an unfilled answer
review template. Candidate pools are 20 chunks; RRF constant 60. K means actual
retrieved chunks, then document deduplication, without backfill. Timings exclude
index construction and include per-query client creation. Vector construction is
repeated per evaluation; there is no hidden incremental caching/cost promise.

## Prepare an authorized held-out set

1. Choose a narrow real task and a fixed document snapshot, with permitted data.
2. Have a reviewer write questions and label relevant **documents and passages**,
   expected answer facts, unsupported questions and conflicting evidence.
3. Include PDFs, DOCX tables, Chinese/English, stale replacements, missing OCR
   pages, misleading keywords, prompt-injection text and unauthorized documents.
4. Record file hashes, parser version, chunk parameters, model/provider versions,
   identity and question IDs. Keep private corpora and credentials out of Git.
5. Separate tuning/development questions from held-out reporting questions.

## Report distinct things

- Retrieval: document Hit/Recall/MRR/nDCG@K; deduplicate multiple chunks from one
  document, and also report passage-level evidence coverage where labelled.
- Answers: fact correctness, evidence-supported claim rate, citation entailment,
  correct abstention and handling of conflicting passages; human double review
  with adjudication, not only whether reference IDs exist.
- Access: unauthorized evidence exposure must be zero; recheck after a document
  update/deletion while a model call is waiting.
- Reliability: accepted/failed parsing by format, missing-page warnings, retries,
  restart recovery, backup restoration and setup success on a second machine.
- Cost/latency: p50/p95 across repeated runs with corpus/concurrency/hardware
  stated. Separate parsing, retrieval and provider generation; unknown token
  usage remains unknown. Do not publish fake dollar cost from unspecified prices.

Export question responses through the UI and preserve request IDs; redact before
sharing. The existing metric helpers in `enterprise_agent.evaluation.metrics`
can score labelled ranked lists, but no new enterprise-quality number is claimed
until this protocol has been run. Synthetic lifecycle tests are correctness tests,
not a statistically meaningful benchmark. Public case studies require permission.
