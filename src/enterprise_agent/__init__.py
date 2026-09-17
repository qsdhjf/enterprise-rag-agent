"""企业级 RAG Agent 的顶层包。

这里仅暴露最常用的数据处理入口 ``main``。更具体的检索、摄取和 Agent
编排能力分别位于对应子包中，避免导入顶层包时创建网络客户端等重对象。
"""

# 保留统一入口，供 ``enterprise-agent`` 命令及直接导入该包的调用方使用。
from enterprise_agent.cli.data_pipeline import main

# 明确限制 ``from enterprise_agent import *`` 时导出的公共符号。
__all__ = ["main"]
