"""把标准化企业文档切成可检索、带稳定标识的重叠文本块。"""

from dataclasses import dataclass
from enterprise_agent.ingestion.identity import chunk_identity

from enterprise_agent.ingestion.data_reader import EnterpriseDocument


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """一段可独立建立索引的文档内容及其来源元数据。"""

    chunk_id: str
    document_id: str
    source: str
    title: str
    sequence: int
    text: str
    created_at: str | None
    updated_at: str | None
    original_location: str | None
    # 默认值保留旧的手工构造方式；真正写入索引前仍会执行完整 ACL 校验。
    tenant_id: str | None = None
    visibility: str | None = None
    acl_principals: tuple[str, ...] = ()
    acl_version: str | None = None


def chunk_document(
    document: EnterpriseDocument,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[DocumentChunk]:
    """按字符上限切分文档，并在相邻块之间保留指定重叠。

    算法优先在后半段的换行符处截断，以减少从段落中间切开的情况；
    每个非空块按租户、来源、文档 ID、顺序和正文生成可复现的完整哈希 ID。

    Raises:
        ValueError: 块大小非正数，或重叠长度不在合法范围内。
    """

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须大于等于 0 且小于 max_chars")

    text = document.content.strip()

    if not text:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    sequence = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # 尚未到文末时，尝试退到当前窗口后半段的最后一个换行符。
        if end < len(text):
            boundary = text.rfind(
                "\n",
                start + max_chars // 2,
                end,
            )
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            # 内容参与哈希：同一文档、同一位置的正文不变时，块 ID 也不变。
            chunk_id = chunk_identity(
                document.tenant_id, document.source, document.document_id,
                sequence, chunk_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    title=document.title,
                    sequence=sequence,
                    text=chunk_text,
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                    original_location=document.original_location,
                    tenant_id=document.tenant_id,
                    visibility=document.visibility,
                    acl_principals=document.acl_principals,
                    acl_version=document.acl_version,
                )
            )
            sequence += 1

        if end >= len(text):
            break

        # 至少前进一个字符，避免极端边界下循环停在原位置。
        start = max(end - overlap_chars, start + 1)

    return chunks
