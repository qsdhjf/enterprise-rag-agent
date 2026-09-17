"""读取数据集约定的 JSON 结构，并标准化成统一的企业文档对象。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_agent.ingestion.access_metadata import (
    extract_document_acl_metadata,
)


@dataclass(frozen=True, slots=True)
class EnterpriseDocument:
    """屏蔽不同数据源字段差异后的统一文档表示。"""

    document_id: str
    source: str
    title: str
    content: str
    created_at: str | None
    updated_at: str | None
    original_location: str | None
    # 可选默认值只为保持旧构造器兼容；严格 reader/writer 不会写入缺失值。
    tenant_id: str | None = None
    visibility: str | None = None
    acl_principals: tuple[str, ...] = ()
    acl_version: str | None = None


def _to_text(value: Any) -> str:
    """把任意 JSON 值转换为适合进入正文的可读文本。"""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    # 对列表、字典和数值使用 JSON 表示，既保留结构又不转义中文。
    return json.dumps(value, ensure_ascii=False, indent=2)


def _load_document(
    path: Path,
    source: str,
    *,
    expected_tenant_id: str | None,
    allow_local_demo_defaults: bool,
) -> EnterpriseDocument:
    """加载单个 JSON 文件，并按其字段元数据提取标题与正文。

    ``title_field_name`` 指向标题字段，``content_field_names`` 列出需要依次
    拼入正文的字段；不存在的值会被忽略，文档 ID 缺失时退回文件名。

    Raises:
        json.JSONDecodeError: 文件不是合法 JSON。
        ValueError: JSON 顶层不是对象。
    """

    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"文档不是 JSON 对象：{path}")

    # 先校验 ACL，避免把没有明确所有者的正文带入后续切块流程。
    acl = extract_document_acl_metadata(
        raw,
        expected_tenant_id=expected_tenant_id,
        allow_local_demo_defaults=allow_local_demo_defaults,
    )

    title_field = raw.get("title_field_name")
    content_fields = raw.get("content_field_names") or []

    # 兼容只有一个正文栏位时把字段名直接写成字符串的数据。
    if isinstance(content_fields, str):
        content_fields = [content_fields]

    title = ""
    if isinstance(title_field, str):
        title = _to_text(raw.get(title_field))

    content_parts: list[str] = []

    if isinstance(content_fields, list):
        for field_name in content_fields:
            # 非字符串字段名无法用于字典查找，跳过而不污染正文。
            if not isinstance(field_name, str):
                continue

            text = _to_text(raw.get(field_name))
            if text:
                # 保留字段名，让后续检索结果仍能解释内容来自哪个业务字段。
                content_parts.append(f"{field_name}:\n{text}")

    return EnterpriseDocument(
        document_id=str(raw.get("dataset_doc_uuid") or path.stem),
        source=source,
        title=title or path.stem,
        content="\n\n".join(content_parts),
        created_at=raw.get("created_at"),
        updated_at=raw.get("updated_at"),
        original_location=raw.get("original_location"),
        tenant_id=acl.tenant_id,
        visibility=acl.visibility,
        acl_principals=acl.acl_principals,
        acl_version=acl.acl_version,
    )


def load_document(
    path: Path,
    source: str,
    *,
    expected_tenant_id: str | None = None,
) -> EnterpriseDocument:
    """严格加载生产文档；缺失、部分或跨租户 ACL 都会失败关闭。"""

    return _load_document(
        path,
        source,
        expected_tenant_id=expected_tenant_id,
        allow_local_demo_defaults=False,
    )


def load_local_demo_document(
    path: Path,
    source: str,
) -> EnterpriseDocument:
    """显式加载旧本地样本，并赋予固定且不可用于生产的 demo ACL。"""

    return _load_document(
        path,
        source,
        expected_tenant_id=None,
        allow_local_demo_defaults=True,
    )
