"""从环境变量加载并校验大语言模型与向量模型的连接配置。"""

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """大语言模型客户端所需的不可变配置。

    ``api_key`` 设置为不参与 ``repr``，可避免调试打印对象时意外泄露密钥。
    """

    base_url: str
    model: str
    api_key: str = field(repr=False)


def load_llm_settings() -> LLMSettings:
    """读取并校验 ``LLM_*`` 环境变量。

    Returns:
        去除首尾空白、规范化 URL 后的模型配置。

    Raises:
        RuntimeError: 必填项缺失，或服务地址不是合法的 HTTP(S) URL。
    """

    # 不覆盖进程中已存在的环境变量，使部署平台传入的配置优先于 .env。
    load_dotenv(override=False)

    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()

    # 一次收集所有缺失项，方便使用者一次性补齐，而不是逐个排错。
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", base_url),
            ("LLM_MODEL", model),
            ("LLM_API_KEY", api_key),
        )
        if not value
    ]

    if missing:
        raise RuntimeError(
            "缺少模型配置：" + ", ".join(missing)
        )

    parsed_url = urlparse(base_url)

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise RuntimeError(
            f"LLM_BASE_URL 格式不正确：{base_url}"
        )

    return LLMSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
    )

@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """向量模型客户端所需的不可变配置与预期向量维度。"""

    base_url: str
    model: str
    dimensions: int
    api_key: str = field(repr=False)


def load_embedding_settings() -> EmbeddingSettings:
    """读取并校验 ``EMBEDDING_*`` 环境变量。

    Returns:
        包含服务地址、模型名、密钥和正整数维度的配置。

    Raises:
        RuntimeError: 配置缺失、URL 非法，或维度不是正整数。
    """

    load_dotenv(override=False)

    base_url = os.getenv(
        "EMBEDDING_BASE_URL",
        "",
    ).strip()
    model = os.getenv(
        "EMBEDDING_MODEL",
        "",
    ).strip()
    api_key = os.getenv(
        "EMBEDDING_API_KEY",
        "",
    ).strip()
    dimensions_text = os.getenv(
        "EMBEDDING_DIMENSIONS",
        "",
    ).strip()

    # 维度也属于索引数据契约，缺失时不能静默采用默认值。
    missing = [
        name
        for name, value in (
            ("EMBEDDING_BASE_URL", base_url),
            ("EMBEDDING_MODEL", model),
            ("EMBEDDING_API_KEY", api_key),
            ("EMBEDDING_DIMENSIONS", dimensions_text),
        )
        if not value
    ]

    if missing:
        raise RuntimeError(
            "缺少 Embedding 配置："
            + ", ".join(missing)
        )

    parsed_url = urlparse(base_url)

    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
    ):
        raise RuntimeError(
            f"EMBEDDING_BASE_URL 格式不正确：{base_url}"
        )

    # 环境变量都是字符串，在客户端创建前转换为明确的整数类型。
    try:
        dimensions = int(dimensions_text)
    except ValueError as error:
        raise RuntimeError(
            "EMBEDDING_DIMENSIONS 必须是整数"
        ) from error

    if dimensions <= 0:
        raise RuntimeError(
            "EMBEDDING_DIMENSIONS must be greater than 0"
        )

    return EmbeddingSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        dimensions=dimensions,
        api_key=api_key,
    )

@dataclass(frozen=True, slots=True)
class RerankerSettings:
    """外部重排模型服务的连接、容量和失败策略配置。"""

    base_url: str
    model: str
    timeout_seconds: float
    max_candidates: int
    required: bool
    api_key: str = field(default="", repr=False)


def _parse_boolean_environment(name: str, value: str) -> bool:
    """严格解析常见环境变量布尔写法，非法值直接报错。"""

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        f"{name} must be true/false, 1/0, yes/no, or on/off"
    )


def load_reranker_settings() -> RerankerSettings:
    """读取并校验 ``RERANKER_*`` 环境变量。"""

    load_dotenv(override=False)

    base_url = os.getenv(
        "RERANKER_BASE_URL",
        "http://127.0.0.1:8081",
    ).strip()
    model = os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    ).strip()
    timeout_text = os.getenv(
        "RERANKER_TIMEOUT_SECONDS",
        "30",
    ).strip()
    max_candidates_text = os.getenv(
        "RERANKER_MAX_CANDIDATES",
        "40",
    ).strip()
    required_text = os.getenv(
        "RERANKER_REQUIRED",
        "true",
    ).strip()
    api_key = os.getenv("RERANKER_API_KEY", "").strip()

    parsed_url = urlparse(base_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
    ):
        raise RuntimeError(
            f"RERANKER_BASE_URL is not a valid HTTP(S) URL: {base_url}"
        )
    if not model:
        raise RuntimeError("RERANKER_MODEL cannot be empty")

    try:
        timeout_seconds = float(timeout_text)
    except ValueError as error:
        raise RuntimeError(
            "RERANKER_TIMEOUT_SECONDS must be a number"
        ) from error
    if timeout_seconds <= 0:
        raise RuntimeError(
            "RERANKER_TIMEOUT_SECONDS must be greater than 0"
        )

    try:
        max_candidates = int(max_candidates_text)
    except ValueError as error:
        raise RuntimeError(
            "RERANKER_MAX_CANDIDATES must be an integer"
        ) from error
    if not 1 <= max_candidates <= 100:
        raise RuntimeError(
            "RERANKER_MAX_CANDIDATES must be between 1 and 100"
        )

    return RerankerSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_seconds=timeout_seconds,
        max_candidates=max_candidates,
        required=_parse_boolean_environment(
            "RERANKER_REQUIRED",
            required_text,
        ),
        api_key=api_key,
    )

