"""把文档块逐行写成适合流式摄取的 UTF-8 JSONL 文件。"""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from enterprise_agent.ingestion.chunker import DocumentChunk
from enterprise_agent.ingestion.access_metadata import (
    ACL_FIELD_NAMES,
    normalize_index_document_acl,
)


def write_chunks_jsonl(
    chunks: Iterable[DocumentChunk],
    output_path: Path,
    overwrite: bool = False,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> int:
    """将块序列写入 JSONL，并返回实际写入的记录数。

    默认拒绝覆盖已有文件；调用方必须显式传入 ``overwrite=True`` 才能替换。
    ``chunks`` 可以是生成器，因此无需把全部数据先加载到内存。
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已存在：{output_path}。"
            "如需替换，请启用 overwrite。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0

    # 在同一个文件上下文中逐块序列化，异常时也会自动关闭文件句柄。
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            record = asdict(chunk)
            # 旧构造器的安全默认会由 asdict 展开为四个空字段。把“全部缺失”
            # 恢复成真正缺字段，之后由严格或显式 local-demo 策略统一处理。
            if (
                record.get("tenant_id") is None
                and record.get("visibility") is None
                and record.get("acl_version") is None
                and not record.get("acl_principals")
            ):
                for field_name in ACL_FIELD_NAMES:
                    record.pop(field_name, None)
            record = normalize_index_document_acl(
                record,
                expected_tenant_id=expected_tenant_id,
                allow_legacy_local=allow_legacy_local,
            )
            line = json.dumps(record, ensure_ascii=False)

            output_file.write(line + "\n")
            written_count += 1

    return written_count
