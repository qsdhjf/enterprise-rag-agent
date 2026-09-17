# Enterprise Agent 架构与运行分支

更新时间：2026-08-18

本文档描述当前源码中的真实架构，包含模块分层、`POST /ask` 的主要运行分支、健康检查分支和当前尚未完成的生产能力。

## 1. 项目整体架构

```mermaid
flowchart TB
    ROOT["Enterprise Agent 项目"]

    ROOT --> ENTRY["交付与运行入口"]
    ENTRY --> API["FastAPI<br/>enterprise_agent.api:app"]
    ENTRY --> CLI["完整 Agent CLI<br/>enterprise-agent-run"]
    ENTRY --> DATA_CLI["数据与检索 CLI<br/>index / ingest / search"]
    ENTRY --> TUTORIAL["tutorial 与 smoke 脚本<br/>仅教学和冒烟测试"]

    ROOT --> CONTROL["Agent 控制与编排层<br/>src/enterprise_agent/agent"]

    CONTROL --> RUNNER["runner.py<br/>一次请求的总入口"]
    CONTROL --> SCOPE["scope_service.py<br/>范围门禁"]
    CONTROL --> STATE["contracts.py + state.py<br/>身份、权限、预算、图状态"]
    CONTROL --> GRAPH["graph.py<br/>LangGraph 状态机"]
    CONTROL --> PLANNER["planner.py<br/>规划下一步动作"]
    CONTROL --> POLICY["policy.py<br/>审批 Planner 动作"]
    CONTROL --> EXECUTOR["tool_executor.py<br/>执行批准动作"]
    CONTROL --> JUDGE["evidence_judge_node.py<br/>审核证据"]
    CONTROL --> ROUTING["evidence_routing.py<br/>回答、重搜、拒答、冲突"]
    CONTROL --> ANSWER["answer_node.py<br/>生成最终回答"]

    ROOT --> RETRIEVAL["在线检索层<br/>src/enterprise_agent/retrieval"]

    RETRIEVAL --> REWRITE["query_rewriter.py<br/>中文问题改写增强"]
    RETRIEVAL --> KEYWORD["keyword_search.py<br/>BM25 关键词检索"]
    RETRIEVAL --> VECTOR["vector_search.py<br/>BGE-M3 向量检索"]
    RETRIEVAL --> HYBRID["hybrid_search.py<br/>RRF 融合排序"]
    RETRIEVAL --> TOOL["enterprise_search.py<br/>统一企业检索工具"]
    HYBRID --> KEYWORD
    HYBRID --> VECTOR

    ROOT --> INGESTION["离线数据处理层<br/>src/enterprise_agent/ingestion"]

    INGESTION --> SOURCES["9 类原始 JSON 数据<br/>Jira、Slack、Confluence、Gmail<br/>Drive、GitHub、Linear、HubSpot、Fireflies"]
    INGESTION --> READER["data_reader.py<br/>统一读取文档"]
    INGESTION --> AUDIT["data_audit.py<br/>数据质量检查"]
    INGESTION --> CHUNKER["chunker.py<br/>结构化切块"]
    INGESTION --> JSONL["artifacts/*.jsonl<br/>切块中间产物"]
    INGESTION --> EMBED["生成 1024 维向量"]
    INGESTION --> BULK["批量写入 OpenSearch"]

    SOURCES --> READER --> AUDIT --> CHUNKER --> JSONL --> EMBED --> BULK

    ROOT --> GENERATION["生成与证据层<br/>src/enterprise_agent/generation"]
    GENERATION --> PROMPT["prompt_builder.py<br/>RAG 提示词"]
    GENERATION --> EJUDGE["evidence_judge.py<br/>结构化证据审核"]
    GENERATION --> RETRY_JUDGE["resilient_judge_client.py<br/>审核输出修复重试"]

    ROOT --> INFRA["基础设施层<br/>src/enterprise_agent/infrastructure"]

    INFRA --> LLM["AutoDL DeepSeek<br/>OpenAI-compatible API"]
    INFRA --> BGE["本地 Ollama BGE-M3<br/>Embedding"]
    INFRA --> OS["OpenSearch 3.7<br/>BM25 + 向量索引"]
    INFRA --> DOCKER["Docker Compose<br/>本地服务运行"]
    INFRA -. "依赖已安装但尚未接线" .-> PG["PostgreSQL Checkpoint"]

    ROOT --> QUALITY["质量保障"]
    QUALITY --> TESTS["tests/<br/>单元、图集成、冒烟、API 测试"]
    QUALITY --> EVAL["evaluation/ + eval/<br/>Hit@K、MRR、混合检索、拒答评测"]
```

## 2. `POST /ask` 完整运行分支

```mermaid
flowchart TD
    REQUEST["POST /ask<br/>{ question: ... }"]

    REQUEST --> INPUT{"请求是否合法?"}

    INPUT -->|"缺字段、空白、超长、类型错误、额外字段"| HTTP422["HTTP 422<br/>输入校验失败"]
    INPUT -->|"合法"| RUNNER["AgentRunner.ask"]

    RUNNER --> NORMALIZE["Unicode 统一<br/>删除隐藏控制字符"]
    NORMALIZE --> HARD_GUARD{"确定性硬规则"}

    HARD_GUARD -->|"提示注入、泄密、绕权组合"| SCOPE_UNSAFE["HTTP 200<br/>scope_rejected<br/>unsafe_request<br/>下游调用数为 0"]
    HARD_GUARD -->|"明确创建、修改、删除、发送等操作"| SCOPE_WRITE["HTTP 200<br/>scope_rejected<br/>unsupported_action<br/>下游调用数为 0"]
    HARD_GUARD -->|"未命中"| SCOPE_MODEL["Scope 分类模型<br/>严格 JSON，最多 2 次"]

    SCOPE_MODEL -->|"模型异常或两次输出无效"| HTTP503_SCOPE["HTTP 503<br/>SCOPE_CLASSIFIER_FAILED"]
    SCOPE_MODEL -->|"capability_help"| SCOPE_HELP["HTTP 200<br/>返回 Agent 能力说明"]
    SCOPE_MODEL -->|"needs_clarification"| SCOPE_CLARIFY["HTTP 200<br/>要求补充主题、对象或编号"]
    SCOPE_MODEL -->|"out_of_scope"| SCOPE_OUT["HTTP 200<br/>不属于企业知识查询"]
    SCOPE_MODEL -->|"unsupported_action"| SCOPE_WRITE
    SCOPE_MODEL -->|"unsafe_request"| SCOPE_UNSAFE
    SCOPE_MODEL -->|"enterprise_knowledge"| INIT["创建 AgentRuntimeContext<br/>创建初始 AgentGraphState"]

    INIT --> VALIDATE{"validate_context"}

    VALIDATE -->|"状态版本、身份、权限版本、时间戳等不变量异常"| HTTP500_CONTEXT["当前为 HTTP 500<br/>未捕获状态不变量异常"]
    VALIDATE -->|"截止时间或累计预算超限"| HTTP503_BUDGET["HTTP 503<br/>budget_exceeded"]
    VALIDATE -->|"通过"| PLANNER["Planner<br/>生成结构化动作"]

    PLANNER -->|"模型调用失败"| HTTP503_AGENT["HTTP 503<br/>Agent 未安全完成"]
    PLANNER -->|"Token 或步骤预算不足"| HTTP503_BUDGET
    PLANNER -->|"JSON 无效"| REPAIR["仅修复 1 次"]
    REPAIR -->|"仍然无效或修复失败"| HTTP503_AGENT
    REPAIR -->|"修复成功"| ACTION{"Planner 动作"}
    PLANNER -->|"合法输出"| ACTION

    ACTION -->|"search_enterprise_knowledge"| POLICY["Policy Gate"]
    ACTION -->|"finish"| POLICY

    POLICY -->|"身份、权限状态不一致"| HTTP500_POLICY["当前为 HTTP 500<br/>策略不变量异常"]
    POLICY -->|"来源无权、跨源不允许、参数非法"| HTTP503_AGENT
    POLICY -->|"工具或切块预算耗尽"| HTTP503_BUDGET
    POLICY -->|"finish 但证据未完全支持"| HTTP503_AGENT
    POLICY -->|"重复搜索且已有审核结果"| NO_PROGRESS["no_progress<br/>使用已有审核结果收尾"]
    POLICY -->|"批准 finish"| FINAL_ROUTE{"现有审核结论"}
    POLICY -->|"批准 search，top_k 已收紧"| EXECUTOR["Tool Executor"]

    EXECUTOR -->|"状态、action_id、预算不一致"| HTTP500_EXEC["当前为 HTTP 500<br/>执行不变量异常"]
    EXECUTOR -->|"工具异常、跨租户、来源错误、缺哈希或 ACL"| EXEC_BUG["status=failed"]
    EXEC_BUG --> BUG_EDGE["当前仍无条件进入 Judge"]
    BUG_EDGE --> HTTP500_BUG["当前路由缺口<br/>Judge 抛 RUN_NOT_ACTIVE<br/>可能 HTTP 500"]
    EXECUTOR -->|"成功，可能返回 0 到 N 条证据"| JUDGE["Evidence Judge"]

    JUDGE -->|"超过 10 条证据、Resolver 失败、审核失败"| HTTP503_AGENT
    JUDGE -->|"审核 Token 超限"| HTTP503_BUDGET
    JUDGE -->|"没有证据"| UNSUPPORTED["unsupported"]
    JUDGE -->|"完全支持"| SUPPORTED["supported"]
    JUDGE -->|"部分支持"| PARTIAL["partially_supported"]
    JUDGE -->|"证据冲突"| CONFLICT["conflicting"]

    SUPPORTED --> FINAL_ROUTE
    CONFLICT --> FINAL_ROUTE

    PARTIAL --> SEARCH_BUDGET{"还有搜索预算?"}
    UNSUPPORTED --> SEARCH_BUDGET

    SEARCH_BUDGET -->|"有预算"| VALIDATE
    SEARCH_BUDGET -->|"部分支持但无预算"| FINAL_ROUTE
    SEARCH_BUDGET -->|"不支持且无预算"| FINAL_ROUTE

    NO_PROGRESS --> FINAL_ROUTE

    FINAL_ROUTE -->|"supported"| FINAL_LLM["回答模型生成完整答案<br/>必须包含有效引用"]
    FINAL_ROUTE -->|"partially_supported"| FINAL_PARTIAL["回答模型只回答有证据部分<br/>明确缺失信息"]
    FINAL_ROUTE -->|"unsupported"| FINAL_DENY["固定证据不足模板<br/>不调用回答模型"]
    FINAL_ROUTE -->|"conflicting"| FINAL_CONFLICT["固定冲突报告模板<br/>不调用回答模型"]

    FINAL_LLM --> FINAL_CHECK{"答案与引用校验"}
    FINAL_PARTIAL --> FINAL_CHECK

    FINAL_CHECK -->|"引用缺失、越界、模型失败、加载失败"| HTTP503_AGENT
    FINAL_CHECK -->|"答案 Token 超预算"| HTTP503_BUDGET
    FINAL_CHECK -->|"通过"| HTTP200_SUPPORTED["HTTP 200<br/>completed<br/>supported 或 partially_supported"]

    FINAL_DENY --> HTTP200_UNSUPPORTED["HTTP 200<br/>completed<br/>unsupported"]
    FINAL_CONFLICT --> HTTP200_CONFLICT["HTTP 200<br/>completed<br/>conflicting"]
```

## 3. 健康检查分支

```mermaid
flowchart TD
    API["FastAPI"]

    API --> HEALTH["GET /health<br/>进程可响应即 200"]
    API --> LIVE["GET /health/live<br/>进程存活即 200"]
    API --> READY{"GET /health/ready"}

    READY -->|"OpenSearch 不可达"| R1["HTTP 503 not_ready"]
    READY -->|"索引不存在"| R2["HTTP 503 not_ready"]
    READY -->|"索引存在但 0 条切块"| R3["HTTP 503 not_ready"]
    READY -->|"检查过程异常"| R4["HTTP 503 not_ready"]
    READY -->|"OpenSearch、索引和数据正常"| R5["HTTP 200 ready"]
```

## 4. 关键代码阅读顺序

1. [`src/enterprise_agent/web/api.py`](../src/enterprise_agent/web/api.py)：HTTP 请求与响应。
2. [`src/enterprise_agent/agent/runner.py`](../src/enterprise_agent/agent/runner.py)：Scope Gate 与 LangGraph 总入口。
3. [`src/enterprise_agent/agent/scope_service.py`](../src/enterprise_agent/agent/scope_service.py)：范围门禁。
4. [`src/enterprise_agent/agent/graph.py`](../src/enterprise_agent/agent/graph.py)：节点与条件边。
5. [`src/enterprise_agent/agent/planner.py`](../src/enterprise_agent/agent/planner.py)：规划动作。
6. [`src/enterprise_agent/agent/policy.py`](../src/enterprise_agent/agent/policy.py)：动作审批。
7. [`src/enterprise_agent/agent/tool_executor.py`](../src/enterprise_agent/agent/tool_executor.py)：执行搜索。
8. [`src/enterprise_agent/agent/evidence_judge_node.py`](../src/enterprise_agent/agent/evidence_judge_node.py)：证据审核。
9. [`src/enterprise_agent/agent/evidence_routing.py`](../src/enterprise_agent/agent/evidence_routing.py)：重搜、回答、拒答和冲突路由。
10. [`src/enterprise_agent/agent/answer_node.py`](../src/enterprise_agent/agent/answer_node.py)：最终回答。

## 5. 当前尚未完成的生产能力

- `execute_search -> evidence_judge` 当前是无条件边，工具失败后仍会进入 Judge。
- PostgreSQL checkpoint 依赖已安装但尚未接入 `graph.compile()`；进程退出后
  不能恢复图的节点执行进度。
- API 仍使用固定的本地演示身份，没有接入 OIDC/JWT。
- 检索和摄取已实现 tenant + user/role/group ACL 的查询前过滤与返回后复验；
  生产环境仍需由真实源系统提供可信 ACL 导出、身份映射和策略版本。
- Agent 图当前只有只读搜索工具，没有企业写工具。
- `awaiting_approval`、`cancelled` 状态已定义，但当前图没有对应执行节点。
- 证据审核最多处理 10 个累计切块；可选 reranker 位于检索内部，尚无独立的
  LangGraph rerank/裁剪节点。
- Scope Gate 在 LangGraph 外，由 Runner 和 CLI 分别调用，存在少量编排重复。
