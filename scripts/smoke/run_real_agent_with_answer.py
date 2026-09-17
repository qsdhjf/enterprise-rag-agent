"""端到端冒烟测试：检索、审核证据并生成带引用的最终回答。"""

from contextlib import ExitStack
from typing import get_args

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
    SourceName,
)
from enterprise_agent.agent.search_adapter import (
    LocalEnterpriseSearchAdapter,
)
from enterprise_agent.agent.final_response import (
    generate_final_answer,
)
from enterprise_agent.agent.graph import build_agent_graph
from enterprise_agent.agent.state import (
    create_initial_graph_state,
)
from enterprise_agent.infrastructure.config import (
    load_embedding_settings,
    load_llm_settings,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.retrieval.enterprise_search import (
    EnterpriseSearchTool,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)


# 修改这个常量即可在本地尝试另一条端到端问题。
QUESTION = (

    "如何退款"
)


def make_context() -> AgentRuntimeContext:
    """创建端到端演示所需的本地身份、预算和检索策略。"""
    return AgentRuntimeContext(
        request_id="real-agent-answer-001",
        # 固定身份仅用于本地测试；生产请求应使用认证后的主体。
        principal=PrincipalContext(
            tenant_id="local-demo-tenant",
            subject_id="local-demo-user",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="local-demo-authz-v1",
        ),
        # 给规划、工具、运行时间、Token 和召回量设置安全上限。
        budget=RunBudget(
            max_steps=3,
            max_tool_calls=3,
            max_runtime_seconds=120,
            max_model_tokens=12000,
            max_retrieved_chunks=10,
        ),
        # 允许跨来源搜索，但限制单次工具调用的 top-k。
        retrieval_policy=RetrievalPolicy(
            policy_version="local-demo-policy-v1",
            allowed_sources=frozenset(
                get_args(SourceName)
            ),
            allow_cross_source=True,
            max_top_k=5,
        ),
    )


def main() -> None:
    """组装真实依赖，运行调查图，校验状态后生成最终回答。"""
    context = make_context()
    initial_state = create_initial_graph_state(
        goal=QUESTION,
        runtime_context=context,
    )

    # ExitStack 统一管理三个外部客户端，任何中途错误都会触发清理。
    with ExitStack() as stack:
        llm_client = OpenAICompatibleLLMClient(
            load_llm_settings()
        )
        stack.callback(llm_client.close)

        embedding_client = OpenAICompatibleEmbeddingClient(
            load_embedding_settings()
        )
        stack.callback(embedding_client.close)

        opensearch_settings = load_opensearch_settings()
        opensearch_client = create_opensearch_client(
            opensearch_settings
        )
        stack.callback(opensearch_client.close)

        # 搜索工具封装查询改写、Embedding 和 OpenSearch 混合召回。
        search_tool = EnterpriseSearchTool(
            llm_client=llm_client,
            embedding_client=embedding_client,
            opensearch_client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
        )
        # 适配器既执行图中的搜索动作，也能按证据引用取回正文。
        adapter = LocalEnterpriseSearchAdapter(search_tool)

        # 调查图负责规划、检索和证据审核；最终回答在图完成后生成。
        graph = build_agent_graph(
            planner_client=llm_client,
            search_tool=adapter,
            evidence_resolver=adapter,
            judge_client=llm_client,
        )

        final_state = graph.invoke(
            initial_state,
            context=context,
        )

        # 缺少完成状态或审核结果时，禁止基于不可靠证据继续作答。
        if (
            final_state["status"] != "completed"
            or final_state["evidence_assessment"] is None
        ):
            raise RuntimeError(
                "Agent investigation did not complete safely: "
                f"{final_state['last_error_code']}"
            )

        # 最终生成器重新解析获批证据，并返回答案、引用和 Token 统计。
        final_answer = generate_final_answer(
            state=final_state,
            runtime_context=context,
            resolver=adapter,
            client=llm_client,
        )

    print("========== 完整 RAG Agent 结果 ==========")
    print("问题：", QUESTION)
    print("调查状态：", final_state["status"])
    print(
        "证据判断：",
        final_state["evidence_assessment"]["decision"],
    )
    print("规划步数：", final_state["step_count"])
    print("搜索次数：", final_state["tool_call_count"])
    print("证据数量：", len(final_state["evidence_refs"]))
    print("调查阶段 Token：", final_state["model_token_count"])
    print(
        "回答阶段 Token：",
        (final_answer.input_tokens or 0)
        + (final_answer.output_tokens or 0),
    )

    print("\n========== 最终回答 ==========")
    print("结果类型：", final_answer.outcome)
    print("使用引用：", ", ".join(final_answer.citations))
    print(final_answer.answer)


if __name__ == "__main__":
    # 冒烟脚本被直接执行时才连接真实服务。
    main()
