"""读取 OpenSearch 配置并构造启用重试的客户端。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv
from opensearchpy import OpenSearch


@dataclass(frozen=True)
class OpenSearchSettings:
    """OpenSearch 服务地址以及关键词、混合检索使用的索引名。"""

    url: str
    index_name: str
    hybrid_index_name: str


def load_opensearch_settings() -> OpenSearchSettings:
    """从环境变量读取并校验 OpenSearch 连接和索引配置。

    Raises:
        RuntimeError: 必填变量缺失，或 URL 不是有效的 HTTP(S) 地址。
    """

    load_dotenv(override=False)

    url = os.getenv("OPENSEARCH_URL", "").strip()
    index_name = os.getenv("OPENSEARCH_INDEX", "").strip()
    hybrid_index_name = os.getenv(
        "OPENSEARCH_HYBRID_INDEX",
        "",
    ).strip()

    if not url:
        raise RuntimeError("缺少环境变量 OPENSEARCH_URL")

    if not index_name:
        raise RuntimeError("缺少环境变量 OPENSEARCH_INDEX")

    if not hybrid_index_name:
        raise RuntimeError(
            "缺少环境变量 OPENSEARCH_HYBRID_INDEX"
        )

    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise RuntimeError(
            "OPENSEARCH_URL 必须使用 http 或 https"
        )

    if not parsed_url.hostname:
        raise RuntimeError(
            "OPENSEARCH_URL 缺少主机地址"
        )

    return OpenSearchSettings(
        url=url,
        index_name=index_name,
        hybrid_index_name=hybrid_index_name,
    )


def create_opensearch_client(
    settings: OpenSearchSettings | None = None,
) -> OpenSearch:
    """使用显式配置或环境配置创建 OpenSearch 客户端。

    HTTPS 地址会同时启用 TLS 和证书/主机名校验；连接超时会自动重试。
    """

    settings = settings or load_opensearch_settings()
    parsed_url = urlparse(settings.url)

    # SSL 相关开关保持一致，避免 HTTPS 被误配成未校验证书的连接。
    use_ssl = parsed_url.scheme == "https"

    return OpenSearch(
        hosts=[
            {
                "host": parsed_url.hostname,
                "port": parsed_url.port or 9200,
            }
        ],
        http_compress=True,
        use_ssl=use_ssl,
        verify_certs=use_ssl,
        ssl_assert_hostname=use_ssl,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
