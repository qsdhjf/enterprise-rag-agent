# Historical development notes

保留早期开发与学习说明；其中本机路径和无认证 API 用法不适用于新版本。
安装、认证与索引迁移以根目录 README 和 docs/DEPLOYMENT.md 为准。

面向企业知识调查场景的只读 RAG Agent。项目使用 FastAPI 提供 HTTP
接口，LangGraph 负责显式状态编排，OpenSearch 同时承担 BM25 与向量
检索，BGE-M3 生成向量，OpenAI-compatible LLM 完成范围分类、规划、
证据审核和最终回答。

## 请求链路

```text
HTTP /ask
  -> Scope Gate
  -> LangGraph Planner
  -> Policy Gate
  -> Enterprise Search Tool
  -> BM25 + Vector + RRF
  -> Evidence Judge
  -> Final Answer with citations
```

## 源码结构

```text
src/enterprise_agent/
├── agent/          LangGraph、状态、规划、策略、工具执行和证据路由
├── retrieval/      BM25、向量检索、RRF、查询改写和企业搜索工具
├── ingestion/      文档读取、切块、索引结构和批量入库
├── generation/     RAG 提示词、证据审核和最终回答
├── infrastructure/ LLM、Embedding、OpenSearch 客户端及配置
├── evaluation/     检索与可回答性离线评测
├── cli/            运维、教学和本地诊断命令
├── web/            FastAPI 路由、生命周期和响应契约
├── api.py          稳定 ASGI 兼容入口
└── __init__.py     Python 包入口

scripts/
├── tutorial/       单节点、Planner、Policy 等教学脚本
└── smoke/          真实依赖的端到端冒烟脚本
```

依赖方向保持单向：

```text
web / cli
    -> agent
        -> retrieval / generation
            -> infrastructure

ingestion -> infrastructure
evaluation -> retrieval / generation / infrastructure
```

上层可以调用下层，下层不应反向依赖 FastAPI 或 CLI。

## 安装

```powershell
uv sync
```

### Anaconda 本地开发环境

Conda 负责隔离本机 Python 3.12，项目的精确 Python 依赖仍由
`requirements-conda.lock`（从 `uv.lock` 导出）统一约束：

```powershell
# 第一次创建环境
conda env create -f .\environment.yml

# 每次开始本地开发时激活
conda activate enterprise-agent

# 安装锁定依赖，再以可编辑模式安装当前项目
python -m pip install -r .\requirements-conda.lock
python -m pip install --no-deps -e .

# 验证环境
python --version
python -m pip check
pytest -q
```

更新 `uv.lock` 后，用下面的命令同步 Conda 锁文件：

```powershell
uv export --frozen --format requirements.txt --no-emit-project --all-groups --output-file requirements-conda.lock
```

激活 Conda 后直接使用 `python`、`pytest` 和 `uvicorn`；不要再运行
`uv run`，因为 `uv run` 会切换到项目自己的 `.venv`，而不是当前 Conda 环境。

本机旧版 Anaconda 4.5.11 尚不支持 PowerShell 中的 `conda run`，也没有
PowerShell 激活钩子。请打开 **Anaconda Prompt**，执行
`activate enterprise-agent`。IDE 的 Python 解释器选择：
`D:\app\annocanda\envs\enterprise-agent\python.exe`。

复制 `.env.example` 为本地 `.env`，再填写实际服务地址和密钥。
不要提交真实 `.env`。

## 常用命令

```powershell
# 数据检查、切块与 JSONL 导出
uv run enterprise-agent --help

# 运行完整 LangGraph Agent
uv run enterprise-agent-run --help

# 企业混合检索
uv run enterprise-agent-search --help

# 创建混合索引
uv run enterprise-agent-index --help

# 批量写入向量
uv run enterprise-agent-ingest --help

# 本地开发 API
uv run uvicorn enterprise_agent.api:app --host 127.0.0.1 --port 8000 --reload
```

## ACL 安全摄取

生产导出和入库应始终指定可信目标租户：

~~~powershell
uv run enterprise-agent --data-root .\sources --source jira --export-jsonl .\artifacts\jira.jsonl --expected-tenant-id tenant-a
uv run enterprise-agent-ingest --jsonl .\artifacts\jira.jsonl --expected-tenant-id tenant-a
~~~

旧生成数据完全没有 ACL 时，只能在本地演示环境显式使用
`--allow-legacy-local`；该参数与 `--expected-tenant-id` 互斥。默认模式会拒绝
缺失、部分、畸形或跨租户 ACL。生产索引若缺少四个 ACL keyword 映射也会
要求重建，详情见 [检索与摄取访问控制](retrieval-access-control.md)。

## Docker

```powershell
docker compose -f .\infra\opensearch\compose.yaml build api
docker compose -f .\infra\opensearch\compose.yaml up -d
docker compose -f .\infra\opensearch\compose.yaml ps
```

API 容器通过 `opensearch:9200` 访问同一 Compose 网络中的 OpenSearch，
通过 `host.docker.internal:11434` 访问 Windows 宿主机上的 Ollama。

不要执行 `docker compose down -v`，除非明确要删除 OpenSearch 数据卷。

## 测试

```powershell
uv run pytest -q
```

## 当前生产化边界

当前 Compose 是本地开发部署：OpenSearch 为单节点并关闭安全插件，
API 使用本地演示身份。生产部署前仍需接入 OIDC/JWT、TLS、集中式 Secret 管理、
PostgreSQL 持久化 Checkpoint、结构化审计和可观测性。
