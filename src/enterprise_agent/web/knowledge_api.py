"""Personal knowledge HTTP surface; every document operation checks ownership."""

import hashlib
import json
import os
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from enterprise_agent.knowledge.parser import MAX_BYTES
from enterprise_agent.knowledge.store import KnowledgeError
from enterprise_agent.web.security import authenticate

router = APIRouter(prefix="/api/knowledge", tags=["Personal knowledge"])
COOKIE = "ea_knowledge_session"


def service(request):
    return request.app.state.knowledge


def guard(request: Request):
    """Reject cross-origin writes; local unauthenticated mode is loopback only.

    No forwarding headers are trusted to turn a remote peer into a local peer.
    Authenticated workspace/live deployments use the existing service identities.
    """
    origin = request.headers.get("origin")
    expected_origin = os.environ.get("AGENT_PUBLIC_ORIGIN", str(request.base_url)).rstrip("/")
    if origin and origin.rstrip("/") != expected_origin:
        raise HTTPException(403, "不允许跨站访问知识库")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "不允许跨站访问知识库")
    if getattr(request.app.state, "demo_mode", False):
        host = urlsplit(str(request.url)).hostname
        peer = request.client.host if request.client else ""
        if host not in {"localhost", "127.0.0.1", "::1"} or peer not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "无登录知识库仅限本机访问；远程部署请使用 workspace 或 live 认证模式")


def owner(request: Request):
    guard(request)
    if getattr(request.app.state, "demo_mode", False):
        return service(request).store.session(request.cookies.get(COOKIE))[0]
    context = authenticate(request)
    principal = context.principal
    # Token rotation preserves ownership; different tenants or subjects never share.
    return "identity:" + hashlib.sha256(json.dumps([principal.tenant_id, principal.subject_id]).encode()).hexdigest()


class LibraryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("名称不能为空或包含控制字符")
        return value.strip()


class KnowledgeQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    answer_mode: Literal["local", "model", "agent"] = "local"
    retrieval_strategy: Literal["bm25", "hybrid", "reranked"] = "bm25"

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


@router.get("/session")
def session(request: Request, response: Response):
    guard(request)
    if getattr(request.app.state, "demo_mode", False):
        _, token = service(request).store.session(request.cookies.get(COOKIE), create=True)
        if token:
            response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=request.url.scheme == "https",
                                max_age=365 * 86400, path="/api/knowledge")
    else:
        owner(request)
    return {"model_enabled": service(request).model_settings is not None,
            "providers": service(request).profiles.summary(), "setup_enabled": setup_available(request),
            "max_file_bytes": MAX_BYTES, "formats": ["txt", "md", "pdf", "docx"],
            "max_documents": 200, "max_total_bytes": 128 * 1024**2,
            "storage": "persistent", "retrieval": "local-bm25", "ocr": False}


@router.get("/libraries")
def libraries(request: Request, identity=Depends(owner)):
    return service(request).store.libraries(identity)


@router.post("/libraries", status_code=201)
def create_library(body: LibraryRequest, request: Request, identity=Depends(owner)):
    return service(request).store.create_library(identity, body.name)


@router.delete("/libraries/{library_id}", status_code=204)
def delete_library(library_id: str, request: Request, identity=Depends(owner)):
    service(request).store.delete_library(identity, library_id)
    return Response(status_code=204)


@router.get("/libraries/{library_id}/documents")
def documents(library_id: str, request: Request, identity=Depends(owner)):
    return service(request).store.documents(identity, library_id)


async def upload(request, identity, library_id, filename, document_id=None):
    # Raw-body upload avoids multipart overhead. Both middleware and store cap size.
    raw = await request.body()
    result = await run_in_threadpool(service(request).store.put_document, identity, library_id, filename, raw, document_id)
    service(request).wake.set()
    return result


@router.post("/libraries/{library_id}/documents", status_code=202)
async def upload_document(library_id: str, request: Request, filename: str = Query(min_length=1, max_length=180), identity=Depends(owner)):
    return await upload(request, identity, library_id, filename)


@router.put("/libraries/{library_id}/documents/{document_id}", status_code=202)
async def replace_document(library_id: str, document_id: str, request: Request,
                           filename: str = Query(min_length=1, max_length=180), identity=Depends(owner)):
    return await upload(request, identity, library_id, filename, document_id)


@router.get("/libraries/{library_id}/documents/{document_id}")
def preview(library_id: str, document_id: str, request: Request, offset: int = Query(0, ge=0),
            limit: int = Query(50, ge=1, le=100), identity=Depends(owner)):
    return service(request).store.preview(identity, library_id, document_id, offset, limit)


@router.delete("/libraries/{library_id}/documents/{document_id}", status_code=204)
def delete_document(library_id: str, document_id: str, request: Request, identity=Depends(owner)):
    service(request).store.mutate_document(identity, library_id, document_id, "delete")
    return Response(status_code=204)


@router.post("/libraries/{library_id}/documents/{document_id}/retry", status_code=202)
def retry(library_id: str, document_id: str, request: Request, identity=Depends(owner)):
    service(request).store.mutate_document(identity, library_id, document_id, "retry")
    service(request).wake.set()
    return {"status": "queued"}


@router.post("/libraries/{library_id}/ask")
def ask(library_id: str, body: KnowledgeQuestion, request: Request, identity=Depends(owner)):
    slots = request.app.state.request_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "服务器繁忙，请稍后重试", headers={"Retry-After": "5"})
    try:
        if body.answer_mode != "agent" and body.retrieval_strategy != "bm25":
            raise KnowledgeError(422, "混合/重排检索请使用完整 Agent 模式")
        return service(request).ask(identity, library_id, body.question, body.answer_mode, request.state.request_id, body.retrieval_strategy)
    finally:
        slots.release()


def setup_available(request):
    return (os.environ.get("AGENT_SETUP_ENABLED") == "1"
            and urlsplit(str(request.url)).hostname in {"127.0.0.1", "localhost", "::1"}
            and request.client is not None and request.client.host in {"127.0.0.1", "::1"})


def setup_guard(request):
    if not setup_available(request):
        raise KnowledgeError(403, "配置向导仅在本机使用 --enable-setup 显式启用；不支持远程修改模型地址")


@router.get("/setup")
def setup(request: Request, identity=Depends(owner)):
    setup_guard(request)
    return service(request).profiles.summary(detailed=True)


@router.put("/setup/{kind}")
def save_provider(kind: str, body: dict, request: Request, identity=Depends(owner)):
    setup_guard(request)
    from enterprise_agent.knowledge.profiles import Provider
    # Do not let Pydantic/FastAPI echo a submitted API key in validation errors.
    if set(body) != {"provider", "revision"} or type(body["revision"]) is not int:
        raise KnowledgeError(422, "配置字段无效，需要 provider 和整数 revision")
    try:
        provider = Provider.model_validate(body["provider"]) if body["provider"] is not None else None
    except (ValidationError, ValueError, TypeError):
        raise KnowledgeError(422, "配置无效：检查地址、模型名和维度。远程需 HTTPS；地址不能包含账号、密钥或查询串") from None
    return service(request).update_profile(kind, provider, body["revision"])


class ExternalConsent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    consent: bool

    @field_validator("consent", mode="before")
    @classmethod
    def explicit_consent(cls, value):
        if value is not True:
            raise ValueError("需要明确 consent=true；外部请求可能传输文本并产生费用")
        return True


@router.post("/setup/{kind}/test")
def test_provider(kind: Literal["llm", "embedding", "reranker"], body: ExternalConsent, request: Request, identity=Depends(owner)):
    setup_guard(request)
    from contextlib import ExitStack
    from time import monotonic
    from enterprise_agent.infrastructure.llm_client import OpenAICompatibleLLMClient
    from enterprise_agent.infrastructure.embedding_client import OpenAICompatibleEmbeddingClient
    from enterprise_agent.infrastructure.reranker_client import TEIRerankerClient
    from enterprise_agent.knowledge.retrieval import normalized
    provider = getattr(service(request).profiles.snapshot(), kind)
    if provider is None:
        raise KnowledgeError(409, "请先保存此模型配置")
    slots = request.app.state.request_slots
    if not slots.acquire(blocking=False):
        raise KnowledgeError(429, "服务忙，请稍后测试")
    started = monotonic()
    try:
        with ExitStack() as stack:
            if kind == "llm":
                client = OpenAICompatibleLLMClient(provider.llm(), timeout_seconds=15, max_retries=0, follow_redirects=False)
                stack.callback(client.close)
                client.generate(system_prompt="Connection test. Reply OK only.", user_prompt="This is synthetic test text, not a user document.", max_output_tokens=8)
            elif kind == "embedding":
                client = OpenAICompatibleEmbeddingClient(provider.embedding(), timeout_seconds=15, max_retries=0, follow_redirects=False)
                stack.callback(client.close)
                normalized(client.embed_text("Synthetic connection test."), provider.dimensions)
            else:
                client = TEIRerankerClient(provider.reranker())
                stack.callback(client.close)
                if not client.is_ready():
                    raise ValueError("TEI health/model/capacity mismatch")
                client.rerank("synthetic test", ["Synthetic connection test.", "Unrelated example."])
        return {"status": "ok", "elapsed_ms": round((monotonic()-started)*1000),
                "note": "仅验证接口可用，不代表检索质量或模型回答正确；测试可能产生费用"}
    except Exception:
        raise KnowledgeError(502, "连接或响应校验失败：检查服务地址、密钥、模型名及向量维度；TEI 还须匹配 /info 模型与容量。未返回供应商原始错误以保护密钥") from None
    finally:
        slots.release()


@router.get("/libraries/{library_id}/index")
def index_status(library_id: str, request: Request, identity=Depends(owner)):
    profile = service(request).profiles.snapshot()
    return service(request).store.vector_status(identity, library_id, profile.embedding.fingerprint if profile.embedding else "")


@router.get("/storage")
def storage(request: Request, identity=Depends(owner)):
    return service(request).store.storage_usage(identity)


@router.get("/libraries/{library_id}/index/plan")
def index_plan(library_id: str, request: Request, force: bool = False, identity=Depends(owner)):
    profile = service(request).profiles.snapshot()
    if not profile.embedding:
        raise KnowledgeError(409, "管理员尚未配置向量服务")
    return service(request).store.vector_plan(identity, library_id, profile.embedding.fingerprint, profile.embedding.dimensions, force)


class IndexBuild(ExternalConsent):
    plan_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    force: bool = False


@router.post("/libraries/{library_id}/index", status_code=202)
def index_build(library_id: str, body: IndexBuild, request: Request, identity=Depends(owner)):
    profile = service(request).profiles.snapshot()
    if not profile.embedding:
        raise KnowledgeError(409, "管理员尚未配置向量服务")
    result = service(request).store.queue_vectors(identity, library_id, profile.embedding.fingerprint,
        profile.embedding.dimensions, plan_id=body.plan_id, force=body.force)
    service(request).wake.set()
    return result


@router.delete("/libraries/{library_id}/index", status_code=204)
def index_cancel(library_id: str, request: Request, identity=Depends(owner)):
    service(request).store.cancel_vectors(identity, library_id)
    return Response(status_code=204)


@router.delete("/libraries/{library_id}/vectors", status_code=204)
def clear_vectors(library_id: str, request: Request, identity=Depends(owner)):
    service(request).store.clear_vectors(identity, library_id)
    return Response(status_code=204)
