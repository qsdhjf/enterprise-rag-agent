"""在本地终端运行完整企业 RAG Agent 的命令行入口。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from typing import get_args
from uuid import uuid4
from enterprise_agent.agent.scope_gate import (
    user_message_for_scope,
)
from enterprise_agent.agent.scope_service import (
    evaluate_request_scope,
)
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
    load_reranker_settings,
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
from enterprise_agent.infrastructure.reranker_client import (
    TEIRerankerClient,
)
from enterprise_agent.generation.resilient_judge_client import (
    RetryingEvidenceJudgeClient,
)


# 未传 --question 时使用该问题，便于本地快速演示。
DEFAULT_QUESTION = (
    "根据企业现有资料，项目被标记为 lifecycle:decom-30d 后，Slack 频道、Confluence 页面和 Jira 工单分别如何处理？请给出证据依据。"
)


def parse_args() -> argparse.Namespace:
    """声明并解析本地 Agent 的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run the local enterprise RAG Agent.",
    )
    # --question 接收用户目标；不传时使用上面的演示问题。
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question for the enterprise Agent.",
    )
    # --debug 仅增加诊断输出，不改变 Agent 的决策流程。
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show internal diagnostic details.",
    )
    return parser.parse_args()


def make_context(request_id: str) -> AgentRuntimeContext:
    """为一次本地请求构造身份、预算和检索策略上下文。"""
    return AgentRuntimeContext(
        request_id=request_id,
        # principal 描述“谁在请求”，生产环境通常来自认证系统。
        principal=PrincipalContext(
            tenant_id="local-demo-tenant",
            subject_id="local-demo-user",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="local-demo-authz-v1",
        ),
        # budget 为图执行设置硬上限，防止循环或资源消耗失控。
        budget=RunBudget(
            max_steps=2,
            max_tool_calls=2,
            max_runtime_seconds=120,
            max_model_tokens=12000,
            max_retrieved_chunks=10,
        ),
        # retrieval_policy 限定允许搜索的来源和单次召回数量。
        retrieval_policy=RetrievalPolicy(
            policy_version="local-demo-policy-v1",
            allowed_sources=frozenset(
                source for source in get_args(SourceName) if source != "uploaded_file"
            ),
            allow_cross_source=True,
            max_top_k=5,
        ),
    )


def print_temporary_failure(
    request_id: str,
    debug: bool,
    error_code: str | None = None,
    error: Exception | None = None,
) -> None:
    """输出面向用户的失败提示，并按需附加内部诊断信息。"""
    print("========== Agent 暂时无法完成请求 ==========")
    print(
        "抱歉，系统暂时无法可靠核验企业证据，"
        "因此没有生成答案。请稍后重试。"
    )
    print("请求编号：", request_id)

    if debug:
        print("\n========== 开发诊断 ==========")
        print("内部错误代码：", error_code)

        if error is not None:
            print(
                "异常：",
                f"{type(error).__name__}: {error}",
            )


def run(args: argparse.Namespace) -> int:
    """执行一次完整 Agent 请求，并返回适合作为进程退出码的整数。"""
    # 统一去除命令行输入两端空白，空问题直接作为参数错误处理。
    question = args.question.strip()

    if not question:
        print("问题不能为空。")
        return 2

    # 每次运行生成独立请求 ID，后续日志和错误信息都可据此关联。
    request_id = f"agent-{uuid4()}"
    context = make_context(request_id)
    initial_state = create_initial_graph_state(
        goal=question,
        runtime_context=context,
    )

    try:
        # ExitStack 集中注册关闭回调；任何中途异常都会释放已有客户端。
        with ExitStack() as stack:
            llm_client = OpenAICompatibleLLMClient(
                load_llm_settings()
            )
            stack.callback(llm_client.close)
            # Scope Gate 先判断问题是否属于企业知识查询，避免无关请求进入图。
            scope_result = evaluate_request_scope(
                question,
                llm_client,
            )

            # 分类器本身失败时不能安全放行，返回可重试的失败状态。
            if (
                not scope_result.classifier_succeeded
                or scope_result.category is None
            ):
                print("========== 暂时无法判断请求范围 ==========")
                print(
                    "系统暂时无法确认这个请求是否属于企业知识"
                    "查询，请稍后重试。"
                )
                print("请求编号：", request_id)

                if args.debug:
                    print("分类原因：", scope_result.reason)
                    print("分类尝试次数：", scope_result.attempts)

                return 2

            # 成功识别为范围外请求时，给出说明并正常结束，不启动 Planner。
            if not scope_result.allowed:
                print("========== Scope Gate ==========")
                print(
                    user_message_for_scope(
                        scope_result.category
                    )
                )
                print("请求编号：", request_id)

                if args.debug:
                    print("分类：", scope_result.category)
                    print("分类原因：", scope_result.reason)
                    print("分类尝试次数：", scope_result.attempts)
                    print("是否进入 Planner：否")

                return 0
            # 只有通过范围检查后，才创建较重的检索依赖。
            embedding_client = OpenAICompatibleEmbeddingClient(
                load_embedding_settings()
            )
            stack.callback(embedding_client.close)

            reranker_client = TEIRerankerClient(
                load_reranker_settings()
            )
            stack.callback(reranker_client.close)

            opensearch_settings = load_opensearch_settings()
            opensearch_client = create_opensearch_client(
                opensearch_settings
            )
            stack.callback(opensearch_client.close)

            # 底层搜索工具组合查询改写、向量生成与 OpenSearch 混合召回。
            search_tool = EnterpriseSearchTool(
                llm_client=llm_client,
                embedding_client=embedding_client,
                opensearch_client=opensearch_client,
                index_name=(
                    opensearch_settings.hybrid_index_name
                ),
                reranker_client=reranker_client,
            )
            # 适配器同时实现“执行搜索”和“按引用取回证据正文”两个协议。
            adapter = LocalEnterpriseSearchAdapter(search_tool)
            judge_client = RetryingEvidenceJudgeClient(
                llm_client
            )

            # 将 Planner、工具、证据审核和答案生成组件注入 LangGraph。
            graph = build_agent_graph(
                planner_client=llm_client,
                search_tool=adapter,
                evidence_resolver=adapter,
                judge_client=judge_client,
                answer_client=llm_client,
            )

            # invoke 会运行图直到完成、被策略拒绝或触发预算/错误终止。
            final_state = graph.invoke(
                initial_state,
                context=context,
            )

            # 只有 completed 且带最终回答才视为一次可靠完成。
            if (
                final_state["status"] != "completed"
                or final_state["final_response"] is None
            ):
                print_temporary_failure(
                    request_id=request_id,
                    debug=args.debug,
                    error_code=final_state["last_error_code"],
                )

                if args.debug:
                    print(
                        "审核自动重试次数：",
                        judge_client.retry_count,
                    )
                    print(
                        "最后校验原因：",
                        judge_client.last_validation_error,
                    )

                return 2

            final_answer = final_state["final_response"]
    except Exception as error:
        # CLI 边界统一兜底，避免向普通用户直接倾倒调用栈。
        print_temporary_failure(
            request_id=request_id,
            debug=args.debug,
            error_code="AGENT_EXECUTION_FAILED",
            error=error,
        )
        return 2

    print("========== Enterprise RAG Agent ==========")
    print("请求编号：", request_id)
    print("结果类型：", final_answer["outcome"])

    if final_answer["citations"]:
        print(
            "使用引用：",
            ", ".join(final_answer["citations"]),
        )

    print("\n" + final_answer["answer"])

    if args.debug:
        # 调试信息帮助开发者定位规划、召回和 Token 预算问题。
        print("\n========== 开发诊断 ==========")
        print("规划步数：", final_state["step_count"])
        print("搜索次数：", final_state["tool_call_count"])
        print("证据数量：", len(final_state["evidence_refs"]))
        print("调查 Token：", final_state["model_token_count"])
        print("审核自动重试次数：", judge_client.retry_count)

    return 0


def main() -> None:
    """把 ``run`` 的返回值转换为命令行进程退出码。"""
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    # 直接运行模块时启动 CLI；被测试导入时不会自动执行。
    main()
