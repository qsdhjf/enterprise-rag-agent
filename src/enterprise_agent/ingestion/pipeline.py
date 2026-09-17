"""用惰性迭代器串联“发现文件 → 读取文档 → 文本切块”的摄取流程。"""

from collections.abc import Iterator
from pathlib import Path

from enterprise_agent.ingestion.chunker import DocumentChunk, chunk_document
from enterprise_agent.ingestion.data_reader import (
    EnterpriseDocument,
    load_document,
    load_local_demo_document,
)


def iter_source_documents(
    data_root: Path,
    source: str,
    limit: int | None = None,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> Iterator[EnterpriseDocument]:
    """递归遍历一个数据源目录，并按需逐个产出标准化文档。

    ``limit`` 限制的是文档数而非文件扫描深度；返回迭代器可避免一次性占用内存。
    """

    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于 0")

    source_dir = data_root / source

    if not source_dir.is_dir():
        raise ValueError(f"数据源目录不存在：{source_dir}")

    document_count = 0

    for path in source_dir.rglob("*.json"):
        if allow_legacy_local:
            # 调用方必须显式选择固定 local-demo 身份，生产路径不会走这里。
            yield load_local_demo_document(path, source)
        else:
            yield load_document(
                path,
                source,
                expected_tenant_id=expected_tenant_id,
            )

        document_count += 1

        if limit is not None and document_count >= limit:
            break


def iter_source_chunks(
    data_root: Path,
    source: str,
    limit: int | None = None,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> Iterator[DocumentChunk]:
    """逐个读取文档并展开其全部文本块，形成端到端的惰性数据流。"""

    documents = iter_source_documents(
        data_root=data_root,
        source=source,
        limit=limit,
        expected_tenant_id=expected_tenant_id,
        allow_legacy_local=allow_legacy_local,
    )

    for document in documents:
        # yield from 会依次向调用方转发当前文档生成的每个块。
        yield from chunk_document(
            document=document,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
