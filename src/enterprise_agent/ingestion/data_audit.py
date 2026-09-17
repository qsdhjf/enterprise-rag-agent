"""抽样加载各数据源的 JSON 文档，快速发现格式或正文标准化问题。"""

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from enterprise_agent.ingestion.data_reader import (
    load_document,
    load_local_demo_document,
)


@dataclass(frozen=True, slots=True)
class AuditResult:
    """单个数据源的抽样审计计数和逐文件错误摘要。"""

    source: str
    checked: int
    passed: int
    failed: int
    errors: tuple[str, ...]


def audit_source(
    source_dir: Path,
    sample_size: int,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> AuditResult:
    """审计数据源目录中最多 ``sample_size`` 个 JSON 文件。

    单个坏文件只记录错误而不中断整个数据源的审计，便于一次查看多处问题。
    """

    checked = 0
    passed = 0
    errors: list[str] = []

    json_files = source_dir.rglob("*.json")

    # islice 让审计保持轻量，不必扫描并加载数据源中的全部文件。
    for path in islice(json_files, sample_size):
        checked += 1

        try:
            if allow_legacy_local:
                document = load_local_demo_document(
                    path,
                    source_dir.name,
                )
            else:
                document = load_document(
                    path,
                    source_dir.name,
                    expected_tenant_id=expected_tenant_id,
                )

            if not document.content.strip():
                raise ValueError("标准化后的正文为空")

            passed += 1
        # 审计的目标是汇总坏样本，因此这里有意捕获解析、IO、字段校验等异常。
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return AuditResult(
        source=source_dir.name,
        checked=checked,
        passed=passed,
        failed=checked - passed,
        errors=tuple(errors),
    )


def audit_data_root(
    data_root: Path,
    sample_size: int,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> list[AuditResult]:
    """按目录名排序后，对数据根目录下的每个子目录执行抽样审计。"""

    if sample_size < 1:
        raise ValueError("sample_size 必须大于 0")

    source_dirs = sorted(
        path for path in data_root.iterdir() if path.is_dir()
    )

    return [
        audit_source(
            source_dir,
            sample_size,
            expected_tenant_id=expected_tenant_id,
            allow_legacy_local=allow_legacy_local,
        )
        for source_dir in source_dirs
    ]
