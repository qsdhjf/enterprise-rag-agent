"""Small BM25 baseline over original, clearly labelled synthetic sample data."""

import json
import math
import re
from collections import Counter
from importlib.resources import files

from enterprise_agent.agent.runner import AgentReadiness, AgentRunResult
from enterprise_agent.agent.presentation import Activity, EvidenceCard, FactCard


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "a", "an", "of", "to", "in", "and", "is", "are", "how", "what", "do", "for"}
    result = [word for word in words if word not in stop]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(run[i:i+2] for i in range(len(run)-1))
    return result


class DemoRunner:
    def __init__(self) -> None:
        self.documents = json.loads(files("enterprise_agent.demo").joinpath("corpus.json").read_text(encoding="utf-8"))
        self.terms = [Counter(tokens(doc["title"] + " " + doc["text"])) for doc in self.documents]
        self.average_length = sum(sum(terms.values()) for terms in self.terms) / len(self.terms)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        """A real BM25 ranking, shared by the demo and its reproducible smoke eval."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        ranked = []
        for doc, terms in zip(self.documents, self.terms, strict=True):
            score = 0.0
            for term in set(tokens(query)):
                df = sum(term in candidate for candidate in self.terms)
                tf = terms[term]
                if not tf:
                    continue
                idf = math.log(1 + (len(self.terms) - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sum(terms.values()) / self.average_length))
            if score > 0:
                ranked.append((score, doc))
        return sorted(ranked, key=lambda item: (-item[0], item[1]["id"]))[:top_k]

    def check_readiness(self) -> AgentReadiness:
        return AgentReadiness(True, False, False, len(self.documents), "Offline samples ready; OpenSearch and models are not used")

    def ask(self, question, runtime_context) -> AgentRunResult:
        ranked = self.search(question)
        cards = tuple(EvidenceCard(
            id=f"E{i}", source=doc["source"], document_id=doc["id"],
            chunk_id=doc["id"], title=doc["title"], excerpt=doc["text"],
        ) for i, (_, doc) in enumerate(ranked, 1))
        if cards:
            answer = "离线检索结果（原文摘录，不是大模型生成或事实审核结论）：\n\n" + "\n\n".join(f"[{card.id}] {card.excerpt}" for card in cards)
        else:
            answer = "离线样本没有匹配的证据。请尝试 Atlas 归档、发布回滚或新员工入职。"
        return AgentRunResult(
            status="completed", request_id=runtime_context.request_id, task_id=None,
            question=question, scope_category="enterprise_knowledge",
            outcome="partially_supported" if cards else "unsupported", answer=answer,
            citations=tuple(card.id for card in cards), step_count=1, tool_call_count=1,
            evidence_count=len(cards), model_token_count=0, evidence=cards, mode="demo",
            facts=(FactCard(fact="回答完整性", status="missing",
                            conclusion="演示仅展示词法召回，未做语义事实审核。请用 live 模式验证真实模型效果。",
                            evidence_ids=[]),),
            activity=(Activity(label="本地 BM25 检索", status="completed", detail=f"9 篇原创合成样本 · {len(cards)} 个命中 · 无网络请求"),),
        )
