"""真实依赖冒烟测试：运行 Agent 调查循环，但不生成最终自然语言答案。"""

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


# 该范围外问题用于观察 Scope Gate/Agent 是否按预期安全结束。
QUESTION = (
    "帮我生成一套贪吃蛇小游戏的代码"
)


def main() -> None:
    """连接真实模型与 OpenSearch，执行一次完整调查图并打印内部状态。"""
    # 这些身份和策略目前只用于本地开发演示。
    context = AgentRuntimeContext(
        request_id="real-agent-smoke-001",
        principal=PrincipalContext(
            tenant_id="local-demo-tenant",
            subject_id="local-demo-user",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="local-demo-authz-v1",
        ),
        budget=RunBudget(
            max_steps=3,
            max_tool_calls=3,
            max_runtime_seconds=120,
            max_model_tokens=12000,
            max_retrieved_chunks=15,
        ),
        retrieval_policy=RetrievalPolicy(
            policy_version="local-demo-policy-v1",
            allowed_sources=frozenset(
                get_args(SourceName)
            ),
            allow_cross_source=True,
            max_top_k=5,
        ),
    )

    # Agent 图以结构化初始状态为输入，而不是直接接收裸字符串。
    initial_state = create_initial_graph_state(
        goal=QUESTION,
        runtime_context=context,
    )

    # 本脚本故意连接真实依赖，因此需要有效的本地服务配置。
    llm_client = OpenAICompatibleLLMClient(
        load_llm_settings()
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        load_embedding_settings()
    )
    opensearch_settings = load_opensearch_settings()
    opensearch_client = create_opensearch_client(
        opensearch_settings
    )

    try:
        # 企业搜索工具组合模型改写、向量生成与混合索引查询。
        enterprise_search_tool = EnterpriseSearchTool(
            llm_client=llm_client,
            embedding_client=embedding_client,
            opensearch_client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
        )

        # 同一个适配器既执行搜索，也为证据审核器取回正文。
        search_adapter = LocalEnterpriseSearchAdapter(
            enterprise_search_tool
        )

        # 这里只验证调查循环，因此没有注入单独的最终回答客户端。
        graph = build_agent_graph(
            planner_client=llm_client,
            search_tool=search_adapter,
            evidence_resolver=search_adapter,
            judge_client=llm_client,
        )

        final_state = graph.invoke(
            initial_state,
            context=context,
        )
    finally:
        # 按依赖创建的相反顺序释放所有外部客户端。
        opensearch_client.close()
        embedding_client.close()
        llm_client.close()

    print("========== 真实 Agent 冒烟测试 ==========")
    print("问题：", QUESTION)
    print("最终状态：", final_state["status"])
    print("规划步数：", final_state["step_count"])
    print("搜索次数：", final_state["tool_call_count"])
    print("证据数量：", len(final_state["evidence_refs"]))
    print("模型 Token：", final_state["model_token_count"])

    # evidence_assessment 可能为空，例如请求在检索前就被范围门拒绝。
    assessment = final_state["evidence_assessment"]

    print("\n========== 证据审核 ==========")

    if assessment is None:
        print("没有生成证据审核结果")
    else:
        print("判断：", assessment["decision"])
        print("审核成功：", assessment["judge_succeeded"])
        print("原因：", assessment["reason"])

        for number, fact in enumerate(
            assessment["facts"],
            start=1,
        ):
            print(
                f"事实 {number}：{fact['fact']} | "
                f"{fact['status']}"
            )

    print("\n========== 证据引用 ==========")

    for number, evidence in enumerate(
        final_state["evidence_refs"],
        start=1,
    ):
        print(
            f"E{number}：{evidence['source']} | "
            f"{evidence['document_id']} | "
            f"{evidence['chunk_id']}"
        )

    print("\n========== 工具审计 ==========")

    for record in final_state["tool_calls"]:
        print(record)

    print("\n错误代码：", final_state["last_error_code"])
    print("错误信息：", final_state["last_error_message"])


if __name__ == "__main__":
    # 冒烟脚本被直接执行时才连接真实服务。
    main()
