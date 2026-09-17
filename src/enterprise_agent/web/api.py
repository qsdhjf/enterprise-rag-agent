"""FastAPI 交付层：负责请求校验、依赖生命周期和 HTTP 状态映射。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import BoundedSemaphore
from contextlib import ExitStack, asynccontextmanager
from time import perf_counter
from typing import Literal, get_args
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
    SourceName,
)
from enterprise_agent.agent.state import (
    EvidenceDecision,
)
from enterprise_agent.agent.runner import (
    AgentRunner,
    AgentRunnerUnavailable,
)
from enterprise_agent.agent.presentation import Activity, EvidenceCard, FactCard
from enterprise_agent.web.security import authenticate, load_identities
from enterprise_agent.web.limits import BodyLimitMiddleware
from enterprise_agent.agent.scope_gate import (
    ScopeCategory,
)
from enterprise_agent.infrastructure.config import (
    load_embedding_settings,
    load_llm_settings,
    load_reranker_settings,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
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
# 使用模块级日志器记录服务端诊断信息，不把内部异常细节暴露给调用者。
logger = logging.getLogger(__name__)


class UTF8JSONResponse(JSONResponse):
    """显式声明 UTF-8 的 JSON 响应，方便不同 HTTP 客户端正确解码中文。"""

    media_type = "application/json; charset=utf-8"


class AskRequest(BaseModel):
    """``POST /ask`` 的输入模型，只接受一个非空问题。"""

    # extra="forbid" 会拒绝拼错或未声明的字段，及早发现调用方错误。
    model_config = ConfigDict(extra="forbid")

    # 长度限制同时保护模型上下文和接口资源消耗。
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """去除首尾空白，并拒绝清理后为空的问题。"""
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("question 不能为空")

        return cleaned


class AskResponse(BaseModel):
    """``POST /ask`` 的稳定响应契约，包含答案、引用和运行统计。"""

    model_config = ConfigDict(extra="forbid")

    # 业务状态只区分正常完成和被范围门拒绝。
    status: Literal[
        "completed",
        "scope_rejected",
    ]
    request_id: str
    task_id: str | None

    # 以下字段描述请求、范围判断和最终答案。
    question: str
    scope_category: ScopeCategory
    outcome: EvidenceDecision | None
    answer: str
    citations: list[str]

    # 以下统计字段帮助调用方观察本次 Agent 的资源消耗。
    step_count: int
    tool_call_count: int
    evidence_count: int
    model_token_count: int
    agent_elapsed_ms: float
    mode: str = "live"
    evidence: list[EvidenceCard] = Field(default_factory=list)
    facts: list[FactCard] = Field(default_factory=list)
    activity: list[Activity] = Field(default_factory=list)


class ReadinessResponse(BaseModel):
    """就绪探针响应：说明搜索服务、索引及数据是否可用。"""

    status: Literal["ready", "not_ready"]
    opensearch_reachable: bool
    hybrid_index_exists: bool
    indexed_chunks: int | None
    message: str
    reranker_ready: bool | None = None
    reranker_model: str | None = None


def create_local_runtime_context(
    request_id: str,
) -> AgentRuntimeContext:
    """创建本地演示上下文；生产身份应由 JWT/OIDC 等认证信息生成。"""

    return AgentRuntimeContext(
        request_id=request_id,
        # 本地固定身份只用于学习和开发，不代表真实授权信息。
        principal=PrincipalContext(
            tenant_id="local-demo-tenant",
            subject_id="local-demo-user",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="local-demo-authz-v1",
        ),
        # 预算限制图步数、工具调用、时长、Token 和召回切块数量。
        budget=RunBudget(
            max_steps=2,
            max_tool_calls=2,
            max_runtime_seconds=120,
            max_model_tokens=12000,
            max_retrieved_chunks=10,
        ),
        # 策略声明本次请求允许访问的数据源及最大 top-k。
        retrieval_policy=RetrievalPolicy(
            policy_version="local-demo-policy-v1",
            allowed_sources=frozenset(
                source for source in get_args(SourceName) if source != "uploaded_file"
            ),
            allow_cross_source=True,
            max_top_k=5,
        ),
    )


@asynccontextmanager
async def agent_lifespan(app: FastAPI):
    """在应用启动时创建共享依赖，并在应用关闭时统一释放。"""
    mode = os.environ.get("AGENT_MODE", "live")
    if mode not in {"demo", "workspace", "live"}:
        raise RuntimeError("AGENT_MODE must be demo, workspace or live")
    if mode == "live":
        from dotenv import load_dotenv
        load_dotenv(override=False)
    app.state.demo_mode = mode == "demo"
    concurrency = int(os.environ.get("AGENT_MAX_CONCURRENT", "2"))
    if not 1 <= concurrency <= 32:
        raise RuntimeError("AGENT_MAX_CONCURRENT must be between 1 and 32")
    app.state.request_slots = BoundedSemaphore(concurrency)
    if app.state.demo_mode:
        from enterprise_agent.demo.runner import DemoRunner
        app.state.identities = ()
        app.state.agent_runner = DemoRunner()
        yield
        return
    identity_file = os.environ.get("AGENT_IDENTITY_FILE")
    if not identity_file:
        raise RuntimeError("Live mode requires AGENT_IDENTITY_FILE; use --mode demo for offline samples")
    app.state.identities = load_identities(identity_file)
    if mode == "workspace":
        app.state.agent_runner = None
        yield
        return
    # ExitStack 保证初始化中途失败时，已创建的客户端也会关闭。
    with ExitStack() as stack:
        llm_client = OpenAICompatibleLLMClient(
            load_llm_settings()
        )
        stack.callback(llm_client.close)

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
        # runner 挂在 app.state 上，所有请求复用同一组已初始化客户端。
        app.state.agent_runner = AgentRunner(
            llm_client=llm_client,
            embedding_client=embedding_client,
            opensearch_client=opensearch_client,
            hybrid_index_name=(
                opensearch_settings.hybrid_index_name
            ),
            reranker_client=reranker_client,
            include_evidence=True,
        )

        # yield 之前是启动逻辑，之后由 ExitStack 执行关闭回调。
        yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    from enterprise_agent.knowledge.service import KnowledgeService
    settings = None
    if os.environ.get("AGENT_KNOWLEDGE_MODEL_ENABLED") == "1":
        settings = load_llm_settings()
    async with agent_lifespan(app):
        knowledge = KnowledgeService(Path(os.environ.get("AGENT_KNOWLEDGE_DIR", ".local/knowledge")), model_settings=settings)
        app.state.knowledge = knowledge
        try:
            yield
        finally:
            knowledge.close()


# 模块导入时只创建应用对象；外部客户端由 lifespan 在服务启动时创建。
app = FastAPI(
    title="Enterprise LangGraph Agent API",
    version="0.5.0",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)

STATIC_ROOT = Path(__file__).with_name("static")
app.add_middleware(BodyLimitMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

from enterprise_agent.web.knowledge_api import router as knowledge_router
from enterprise_agent.knowledge.store import KnowledgeError

app.include_router(knowledge_router)

from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request, error):
    if request.url.path.startswith("/api/knowledge/setup"):
        return UTF8JSONResponse({"detail": "配置请求格式无效；为保护密钥不回显输入"}, status_code=422)
    return await request_validation_exception_handler(request, error)


@app.exception_handler(KnowledgeError)
async def knowledge_error(request, error):
    return UTF8JSONResponse({"detail": str(error), "request_id": request.state.request_id}, status_code=error.status)


@app.get("/", include_in_schema=False)
def workspace():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/config")
def public_config(request: Request):
    demo = getattr(request.app.state, "demo_mode", False)
    mode = "demo" if demo else ("workspace" if getattr(request.app.state, "agent_runner", None) is None else "live")
    return {"mode": mode, "version": "0.5.0", "knowledge": True,
            "requires_token": not demo,
            "examples": ["Atlas 项目应该如何归档？", "Atlas 发布失败后怎么回滚？", "新员工入职需要做什么？"] if demo else []}


@app.middleware("http")
async def add_request_tracking(
    request: Request,
    call_next,
):
    """为每个请求添加关联 ID，并把服务端耗时写入响应头。"""
    # 请求 ID 同时存入 request.state，供后续路由和日志复用。
    request_id = str(uuid4())
    request.state.request_id = request_id

    # perf_counter 是单调时钟，适合计算耗时而不受系统时间调整影响。
    started_at = perf_counter()
    response = await call_next(request)
    elapsed_ms = (
        perf_counter() - started_at
    ) * 1000

    # 两个响应头让客户端能够关联日志并观察 API 层总处理时间。
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{elapsed_ms:.2f}"
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Swagger uses a CDN; the product workspace has no remote scripts or assets.
    if request.url.path not in {"/docs", "/redoc"}:
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

    return response


@app.get("/health")
def health() -> dict[str, str]:
    """返回最轻量的进程健康状态。"""
    return {"status": "ok"}


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """供编排系统判断 API 进程是否仍在运行。"""
    return {
        "status": "alive",
        "message": "API 进程正常",
    }


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
)
def readiness(
    http_request: Request,
    response: Response,
) -> ReadinessResponse:
    """检查 OpenSearch、混合索引和索引数据是否可供真实请求使用。"""
    if not getattr(http_request.app.state, "demo_mode", False):
        authenticate(http_request)
    if getattr(http_request.app.state, "agent_runner", None) is None:
        return ReadinessResponse(status="ready", opensearch_reachable=False, hybrid_index_exists=False,
                                 indexed_chunks=None, message="Personal workspace ready; enterprise search is not enabled")
    # lifespan 已把共享 runner 放入 app.state，这里只读取而不重新创建。
    runner: AgentRunner = (
        http_request.app.state.agent_runner
    )
    result = runner.check_readiness()

    # 未就绪时保留结构化响应，同时通过 503 告知负载均衡器暂停流量。
    if not result.ready:
        response.status_code = 503

    return ReadinessResponse(
        status=(
            "ready"
            if result.ready
            else "not_ready"
        ),
        opensearch_reachable=(
            result.opensearch_reachable
        ),
        hybrid_index_exists=(
            result.hybrid_index_exists
        ),
        indexed_chunks=result.indexed_chunks,
        message=result.message,
        reranker_ready=result.reranker_ready,
        reranker_model=result.reranker_model,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
    http_request: Request,
) -> AskResponse:
    """执行一次企业知识问答，并把领域结果映射为 HTTP 响应。"""
    # 复用共享 runner，并沿用中间件生成的请求关联 ID。
    runner: AgentRunner = (
        http_request.app.state.agent_runner
    )
    request_id = http_request.state.request_id
    runtime_context = (create_local_runtime_context(request_id)
                       if getattr(http_request.app.state, "demo_mode", False)
                       else authenticate(http_request))
    if runner is None:
        raise HTTPException(409, "个人工作空间请通过知识库问答接口提问；未启用企业索引")
    slots = http_request.app.state.request_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "Server is busy; retry later", headers={"Retry-After": "5"})

    started_at = perf_counter()

    try:
        # AgentRunner 封装范围判断、规划、检索、审核和答案生成全过程。
        result = runner.ask(
            question=request.question,
            runtime_context=runtime_context,
        )
    except AgentRunnerUnavailable as error:
        # 依赖暂不可用映射为 503，并只向客户端暴露安全的概要信息。
        logger.warning(
            "agent_unavailable request_id=%s code=%s",
            request_id,
            error.error_code,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Agent 暂时无法完成请求",
                "request_id": request_id,
            },
        ) from error
    except ValueError as error:
        # 领域层输入错误映射为 422，表示请求语义无法处理。
        raise HTTPException(
            status_code=422,
            detail="请求无法处理，请检查输入或联系管理员",
        ) from error
    except Exception as error:
        logger.warning("agent_unexpected_failure request_id=%s type=%s", request_id, type(error).__name__)
        raise HTTPException(503, detail={"message": "Agent 暂时无法完成请求", "request_id": request_id}) from error
    finally:
        slots.release()

    elapsed_ms = (
        perf_counter() - started_at
    ) * 1000

    # 显式组装响应模型，防止领域对象中的内部字段意外泄漏到 API。
    return AskResponse(
        status=result.status,
        request_id=result.request_id,
        task_id=result.task_id,
        question=result.question,
        scope_category=result.scope_category,
        outcome=result.outcome,
        answer=result.answer,
        citations=list(result.citations),
        step_count=result.step_count,
        tool_call_count=result.tool_call_count,
        evidence_count=result.evidence_count,
        model_token_count=result.model_token_count,
        agent_elapsed_ms=round(elapsed_ms, 2),
        evidence=list(result.evidence), facts=list(result.facts),
        activity=list(result.activity), mode=result.mode,
    )
