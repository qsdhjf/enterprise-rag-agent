# Scripts

- `tutorial/`：拆解展示 Agent 的单个节点与状态变化，不作为生产入口。
- `smoke/`：连接真实 LLM、Ollama 和 OpenSearch 的端到端冒烟验证。

正式命令行入口位于 `src/enterprise_agent/cli/`，并通过
`pyproject.toml` 的 `[project.scripts]` 暴露。
