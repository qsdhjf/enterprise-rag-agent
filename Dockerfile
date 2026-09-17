# syntax=docker/dockerfile:1

# 构建阶段使用 uv 创建锁定依赖对应的虚拟环境，避免把构建工具带入运行镜像。
FROM ghcr.io/astral-sh/uv:0.12.2-python3.12-trixie-slim AS builder

WORKDIR /app

# 编译字节码可缩短容器启动时间；copy 模式确保虚拟环境可跨阶段复制。
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# 先复制依赖元数据可充分利用 Docker 层缓存，源代码变化时无需重新下载依赖。
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src

# --frozen 要求严格使用 uv.lock；生产镜像不安装开发依赖和可编辑源码链接。
RUN uv sync --frozen --no-dev --no-editable


# 运行阶段只保留 Python 和已经构建好的虚拟环境，以减小镜像体积。
FROM python:3.12-slim-trixie AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_KNOWLEDGE_DIR=/data/knowledge

# 用无特权用户运行 Web 服务，降低容器进程被利用后的权限范围。
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /data/knowledge \
    && chown app:app /data/knowledge

COPY --from=builder --chown=app:app /app/.venv /app/.venv

USER app

EXPOSE 8000

# 活性探针只检查 API 进程是否可响应。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]

# 默认以前台方式启动 FastAPI，便于容器平台收集日志和管理进程生命周期。
CMD ["enterprise-agent-serve", "--mode", "demo", "--host", "0.0.0.0", "--port", "8000"]
