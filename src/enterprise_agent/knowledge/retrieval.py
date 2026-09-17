"""Private library retrieval: BM25, exact cosine, RRF, optional TEI reranking.

SQLite vectors suit bounded small libraries, not a distributed ANN service.
Embedding construction is an explicit job, never a side effect of an upload.
"""

import hashlib
import json
import math
from collections import Counter

from enterprise_agent.demo.runner import tokens
from enterprise_agent.knowledge.store import KnowledgeError


def row_key(row):
    return f'{row["document_id"]}:{row["revision"]}:{row["ordinal"]}'


def embedding_text(row):
    return row["name"] + "\n" + row["text"]


def embedding_hash(row):
    # Include the filename because it is part of the exact provider input.
    return hashlib.sha256(embedding_text(row).encode()).hexdigest()


def corpus_hash(rows):
    canonical = sorted((row_key(row), row["name"], row["location"], row["text"]) for row in rows)
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False).encode()).hexdigest()


def bm25(rows, query, top_k):
    if not rows:
        return []
    terms = [Counter(tokens(row["name"] + " " + row["text"])) for row in rows]
    query = set(tokens(query))
    lengths = [sum(candidate.values()) for candidate in terms]
    average = sum(lengths) / len(lengths) or 1
    df = Counter(term for candidate in terms for term in query if term in candidate)
    ranked = []
    for row, candidate, length in zip(rows, terms, lengths, strict=True):
        score = sum(math.log(1 + (len(rows) - df[t] + .5) / (df[t] + .5)) * candidate[t] * 2.2 /
                    (candidate[t] + 1.2 * (.25 + .75 * length / average)) for t in query if candidate[t])
        if score:
            ranked.append(dict(row, score=score))
    return sorted(ranked, key=lambda row: (-row["score"], row_key(row)))[:top_k]


def normalized(vector, dimensions):
    if len(vector) != dimensions or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in vector):
        raise KnowledgeError(502, "向量模型返回维度或数值无效")
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise KnowledgeError(502, "向量模型返回零向量或溢出数值")
    return [v / norm for v in vector]


def fuse(lexical, semantic, top_k):
    scores, candidates = {}, {}
    for ranked in (lexical, semantic):
        for rank, row in enumerate(ranked, 1):
            key = row_key(row)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            candidates[key] = row
    return [dict(candidates[key], score=scores[key]) for key in sorted(scores, key=lambda k: (-scores[k], k))[:top_k]]


class LibraryRetriever:
    def __init__(self, store, owner, library_id, profile, *, embedding_client=None, reranker_client=None):
        self.store, self.owner, self.library_id, self.profile = store, owner, library_id, profile
        self.embedding_client, self.reranker_client = embedding_client, reranker_client
        self.query_embeddings = 0
        self.rerank_calls = 0

    def search(self, query, top_k=5, strategy="bm25"):
        rows = self.store.corpus(self.owner, self.library_id)
        lexical = bm25(rows, query, 20)
        if strategy == "bm25":
            return lexical[:top_k]
        if strategy not in {"vector", "hybrid", "reranked"}:
            raise KnowledgeError(422, "未知检索策略")
        if not self.profile.embedding or self.embedding_client is None:
            raise KnowledgeError(409, "请先配置向量模型并构建该知识库的语义索引")
        vectors = self.store.read_vectors(self.owner, self.library_id, self.profile.embedding.fingerprint, corpus_hash(rows))
        if not rows:
            return []
        vector = normalized(self.embedding_client.embed_text(query), self.profile.embedding.dimensions)
        self.query_embeddings += 1
        semantic = []
        for row in rows:
            stored = vectors.get(row_key(row))
            if stored is None:
                raise KnowledgeError(409, "语义索引不完整，请重建")
            # Validate stored floats as well; corrupted disk content must fail closed.
            stored = normalized(stored, len(vector))
            score = sum(a*b for a, b in zip(vector, stored, strict=True))
            semantic.append(dict(row, score=score))
        semantic.sort(key=lambda row: (-row["score"], row_key(row)))
        ranked = semantic[:20] if strategy == "vector" else fuse(lexical, semantic[:20], 20)
        if strategy == "reranked":
            if not self.profile.reranker or self.reranker_client is None:
                raise KnowledgeError(409, "尚未配置重排服务")
            scores = self.reranker_client.rerank(query, [row["text"] for row in ranked])
            self.rerank_calls += 1
            ranked = [dict(ranked[s.index], score=s.score) for s in sorted(scores, key=lambda s: (-s.score, s.index))]
        # Vector-only semantic candidates can be irrelevant. The evidence judge,
        # not a magic cosine cutoff, determines whether they support an answer.
        self.store.verify_evidence(self.owner, self.library_id, ranked[:top_k])
        return ranked[:top_k]
