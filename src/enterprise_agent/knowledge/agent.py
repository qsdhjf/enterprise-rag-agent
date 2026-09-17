"""Run the existing evidence-first graph against one owned upload library.

Only the storage/search adapter changes. Planner, policy, evidence assessment,
bounded follow-up searches and final citation validation remain shared code.
"""

import hashlib
from contextlib import ExitStack
from time import monotonic

from enterprise_agent.agent.contracts import AgentRuntimeContext, PrincipalContext, RetrievalPolicy, RunBudget
from enterprise_agent.agent.graph import build_agent_graph
from enterprise_agent.agent.state import create_initial_graph_state
from enterprise_agent.agent.tool_executor import SearchToolResponse
from enterprise_agent.generation.resilient_judge_client import RetryingEvidenceJudgeClient
from enterprise_agent.infrastructure.llm_client import OpenAICompatibleLLMClient
from enterprise_agent.knowledge.retrieval import row_key, corpus_hash
from enterprise_agent.knowledge.store import KnowledgeError
from enterprise_agent.retrieval.keyword_search import OpenSearchHit


class LibraryAdapter:
    def __init__(self, retriever, context, strategy):
        self.retriever, self.context, self.strategy = retriever, context, strategy
        self.selected = {}
        self.failure = None

    def reference(self, row):
        return dict(tenant_id=self.context.principal.tenant_id, source="uploaded_file",
                    document_id=row["document_id"], chunk_id=row_key(row), sequence=row["ordinal"],
                    content_sha256=hashlib.sha256(row["text"].encode()).hexdigest(), acl_version=str(row["revision"]))

    def run(self, request):
        try:
            p = self.context.principal
            if (request.tenant_id, request.subject_id, request.authz_version, request.source) != (
                    p.tenant_id, p.subject_id, p.authz_version, "uploaded_file"):
                raise KnowledgeError(403, "知识库检索身份或来源不匹配")
            rows = self.retriever.search(request.query, request.top_k, self.strategy)
            self.selected.update({row_key(r): r for r in rows})
            return SearchToolResponse(evidence=tuple(self.reference(r) for r in rows))
        except KnowledgeError as error:
            self.failure = error
            raise

    def resolve(self, references, runtime_context):
        if runtime_context.principal != self.context.principal:
            raise KnowledgeError(403, "知识库证据身份不匹配")
        r = self.retriever
        fresh = {row_key(row): row for row in r.store.corpus(r.owner, r.library_id)}
        hits = []
        for reference in references:
            row = fresh.get(reference["chunk_id"])
            if row is None or reference != self.reference(row) or reference["chunk_id"] not in self.selected:
                raise KnowledgeError(409, "证据已更新、删除或完整性校验失败；请重新提问")
            hits.append(OpenSearchHit(score=self.selected[row_key(row)]["score"], chunk_id=row_key(row),
                document_id=row["document_id"], source="uploaded_file", title=row["name"],
                sequence=row["ordinal"], text=row["text"], created_at=None, original_location=None,
                tenant_id=self.context.principal.tenant_id, acl_version=str(row["revision"])))
        return hits


class BoundedClient:
    def __init__(self, client, deadline, planner=False):
        self.client, self.deadline, self.planner = client, deadline, planner
        self.calls = 0
        self.tokens = 0
        self.usage_complete = True

    def generate(self, **kwargs):
        if monotonic() >= self.deadline:
            raise TimeoutError("Agent runtime budget exceeded")
        if self.planner:
            kwargs["system_prompt"] += ('\nThis run searches only the user-selected uploaded library. '
                'Every search must set source="uploaded_file" explicitly; never null or other sources. '
                'Keep top_k at most 3. Preserve the language of the question in queries.')
        self.calls += 1
        response = self.client.generate(**kwargs)
        if response.input_tokens is None or response.output_tokens is None:
            self.usage_complete = False
        else:
            self.tokens += response.input_tokens + response.output_tokens
        if monotonic() >= self.deadline:
            raise TimeoutError("Agent runtime budget exceeded")
        return response


def run_library_agent(service, owner, library_id, question, request_id, strategy="bm25", *, client=None):
    started = monotonic()
    profile = service.profiles.snapshot()
    # Validate ownership/index/config BEFORE sending a question to any provider.
    rows = service.store.corpus(owner, library_id)
    if not profile.llm:
        raise KnowledgeError(409, "请先在本机设置向导配置对话模型")
    if strategy not in {"bm25", "hybrid", "reranked"}:
        raise KnowledgeError(422, "Agent 支持关键词、混合检索或混合加重排")
    if strategy != "bm25":
        if not profile.embedding:
            raise KnowledgeError(409, "请配置向量模型并构建语义索引")
        service.store.read_vectors(owner, library_id, profile.embedding.fingerprint, corpus_hash(rows))
    if strategy == "reranked" and not profile.reranker:
        raise KnowledgeError(409, "请配置 TEI 重排服务")
    context = AgentRuntimeContext(request_id=request_id,
        principal=PrincipalContext(tenant_id="library-" + library_id, subject_id=hashlib.sha256(owner.encode()).hexdigest(),
            roles=frozenset({"library-owner"}), authz_version="owned-library-v1"),
        retrieval_policy=RetrievalPolicy(policy_version="owned-library-v1", allowed_sources=frozenset({"uploaded_file"}), max_top_k=3),
        budget=RunBudget(max_steps=3, max_tool_calls=3, max_retrieved_chunks=10, max_runtime_seconds=180, max_model_tokens=24000))
    with ExitStack() as stack:
        if client is None:
            client = OpenAICompatibleLLMClient(profile.llm.llm(), timeout_seconds=25, max_retries=0, follow_redirects=False)
            stack.callback(client.close)
        retriever = stack.enter_context(service.retriever(owner, library_id, strategy, profile))
        adapter = LibraryAdapter(retriever, context, strategy)
        planner = BoundedClient(client, started + 180, planner=True)
        judge = BoundedClient(client, started + 180)
        answer = BoundedClient(client, started + 180)
        graph = build_agent_graph(planner, adapter, adapter, RetryingEvidenceJudgeClient(judge), answer)
        activity = []
        state = None
        # Capture actual node names, not invented UI stages or model reasoning.
        for kind, value in graph.stream(create_initial_graph_state(question, context), context=context,
                                         stream_mode=["values", "updates"], config={"recursion_limit": 40}):
            if kind == "values":
                state = value
            else:
                activity.extend({"label": name, "status": "completed", "detail": "节点已执行"} for name in value if not name.startswith("__"))
        if adapter.failure:
            raise adapter.failure
        if state is None or state["status"] != "completed" or not state.get("final_response"):
            code = state.get("last_error_code", "INCOMPLETE") if state else "INCOMPLETE"
            raise KnowledgeError(502, f"Agent 未能可靠完成本次核验（{code}）；请重试或使用本地原文检索")
        adapter.resolve(state["evidence_refs"], context)
        cards = []
        for i, ref in enumerate(state["evidence_refs"], 1):
            row = adapter.selected[ref["chunk_id"]]
            cards.append(dict(id=f"E{i}", source="uploaded_file", chunk_id=row_key(row), document_id=row["document_id"],
                title=row["name"], excerpt=row["text"], location=row["location"], ordinal=row["ordinal"], revision=row["revision"], url=None))
        service.store.verify_evidence(owner, library_id, cards)
        final = state["final_response"]
        activity.extend(dict(label="检索", status=call["status"], detail=f'{call["query"]} · {call["returned_hits"]} 个切块') for call in state["tool_calls"])
        return dict(status="completed", request_id=request_id, task_id=state["task_id"], question=question,
            scope_category="personal_knowledge", outcome=final["outcome"], answer=final["answer"], citations=final["citations"],
            step_count=state["step_count"], tool_call_count=state["tool_call_count"], evidence_count=len(cards),
            model_token_count=sum(c.tokens for c in (planner,judge,answer)) if all(c.usage_complete for c in (planner,judge,answer)) else None,
            model_call_count=sum(c.calls for c in (planner,judge,answer)), embedding_calls=retriever.query_embeddings,
            rerank_calls=retriever.rerank_calls, retrieval_strategy=strategy, agent_elapsed_ms=round((monotonic()-started)*1000,2),
            mode="knowledge-agent", library_id=library_id, evidence=cards,
            facts=(state.get("evidence_assessment") or {}).get("facts", []), activity=activity,
            caveat="模型审核不是人工保证；请核对引用原文。模型用量不包含向量和重排服务的计费。")
