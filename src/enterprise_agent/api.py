"""稳定的 ASGI 入口。

部署配置可以一直引用 ``enterprise_agent.api:app``；真正的 Web 实现位于
``enterprise_agent.web.api``，这样内部移动代码时无需同步修改部署命令。
"""

# 这里只做重新导出，不在兼容层重复创建 FastAPI 应用。
from enterprise_agent.web.api import app

__all__ = ["app"]
