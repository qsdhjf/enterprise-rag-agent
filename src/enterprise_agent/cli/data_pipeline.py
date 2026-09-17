"""检查、审计并导出企业数据源的命令行数据流水线。"""

from argparse import ArgumentParser
from pathlib import Path

from enterprise_agent.ingestion.chunker import chunk_document
from enterprise_agent.ingestion.data_audit import audit_data_root
from enterprise_agent.ingestion.data_reader import load_document
from enterprise_agent.ingestion.pipeline import iter_source_chunks
from enterprise_agent.ingestion.jsonl_writer import write_chunks_jsonl


def inspect_document(
    data_root: Path,
    source: str,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> None:
    """读取指定数据源的一份样例文档，并展示其切块结果。"""
    source_dir = data_root / source

    # 先确认数据源目录存在，给出比底层文件异常更清楚的错误。
    if not source_dir.is_dir():
        raise SystemExit(f"数据源目录不存在：{source_dir}")

    # 这里只取第一份 JSON 作为人工抽查样例，不会遍历处理全部文件。
    sample_path = next(source_dir.rglob("*.json"), None)

    if sample_path is None:
        raise SystemExit(f"目录中没有 JSON 文件：{source_dir}")

    # reader 负责统一源格式，chunker 再把标准文档拆成可检索片段。
    if allow_legacy_local:
        from enterprise_agent.ingestion.data_reader import (
            load_local_demo_document,
        )

        document = load_local_demo_document(sample_path, source)
    else:
        document = load_document(
            sample_path,
            source,
            expected_tenant_id=expected_tenant_id,
        )
    chunks = chunk_document(document)

    print(f"文件：{sample_path}")
    print(f"文档 ID：{document.document_id}")
    print(f"数据源：{document.source}")
    print(f"标题：{document.title}")
    print(f"正文长度：{len(document.content)}")
    print(f"正文预览：\n{document.content[:500]}")
    print(f"\n切块数量：{len(chunks)}")

    for chunk in chunks:
        print(
            f"- 序号 {chunk.sequence} | "
            f"{chunk.chunk_id} | "
            f"{len(chunk.text)} 字符"
        )


def run_audit(
    data_root: Path,
    sample_size: int,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> None:
    """抽样审计所有数据源；任一来源失败时以非零状态退出。"""
    # 每个结果记录该来源检查、通过和失败的数量及错误详情。
    results = audit_data_root(
        data_root,
        sample_size,
        expected_tenant_id=expected_tenant_id,
        allow_legacy_local=allow_legacy_local,
    )
    has_failure = False

    for result in results:
        # 没有可检查记录也不能算通过，避免空目录产生假阳性。
        passed = result.checked > 0 and result.failed == 0
        status = "通过" if passed else "失败"

        print(
            f"[{status}] {result.source}: "
            f"检查 {result.checked}，"
            f"成功 {result.passed}，"
            f"失败 {result.failed}"
        )

        for error in result.errors:
            print(f"  - {error}")

        if not passed:
            has_failure = True

    # SystemExit(1) 让 shell/CI 能依据退出码判断审计失败。
    if has_failure:
        raise SystemExit(1)


def export_documents(
    data_root: Path,
    source: str,
    limit: int,
    output_path: Path,
    overwrite: bool,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> None:
    """把指定来源的文档切块流式写入一个 JSONL 文件。"""
    # iter_source_chunks 返回迭代器，避免一次把全部文档装入内存。
    chunks = iter_source_chunks(
        data_root=data_root,
        source=source,
        limit=limit,
        expected_tenant_id=expected_tenant_id,
        allow_legacy_local=allow_legacy_local,
    )

    try:
        written_count = write_chunks_jsonl(
            chunks=chunks,
            output_path=output_path,
            overwrite=overwrite,
            expected_tenant_id=expected_tenant_id,
            allow_legacy_local=allow_legacy_local,
        )
    except FileExistsError as exc:
        # 是否覆盖由 --overwrite 显式控制，默认保护已有产物。
        raise SystemExit(str(exc)) from exc

    print(f"数据源：{source}")
    print(f"读取文档上限：{limit}")
    print(f"写入切块数量：{written_count}")
    print(f"输出文件：{output_path.resolve()}")


def main() -> None:
    """解析模式与路径参数，并分派到检查、审计或导出流程。"""
    parser = ArgumentParser(description="企业 Agent 数据工具")

    # --data-root 是包含 jira、slack 等来源子目录的根目录。
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="企业数据 sources 目录",
    )
    # --source 用于样例检查和单来源导出，默认选择 jira。
    parser.add_argument(
        "--source",
        default="jira",
        help="数据源名称，默认是 jira",
    )

    # 审计与导出互斥；两者都不传时进入单文档检查模式。
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit",
        action="store_true",
        help="抽样检查全部数据源",
    )
    mode.add_argument(
        "--export-jsonl",
        type=Path,
        help="将切块写入指定的 JSONL 文件",
    )

    # 以下参数分别控制审计抽样量、导出上限和覆盖策略。
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="审计时每个数据源的抽样数量",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="导出时最多读取多少份文档",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有的 JSONL 文件",
    )
    acl_mode = parser.add_mutually_exclusive_group()
    acl_mode.add_argument(
        "--expected-tenant-id",
        help="生产摄取的可信目标租户；源 ACL 必须与其完全一致",
    )
    acl_mode.add_argument(
        "--allow-legacy-local",
        action="store_true",
        help="仅本地演示：为完全缺失 ACL 的旧数据赋固定 demo ACL",
    )

    args = parser.parse_args()

    # 按互斥模式进行清晰的入口分派。
    if args.audit:
        run_audit(
            args.data_root,
            args.sample_size,
            expected_tenant_id=args.expected_tenant_id,
            allow_legacy_local=args.allow_legacy_local,
        )
    elif args.export_jsonl is not None:
        export_documents(
            data_root=args.data_root,
            source=args.source,
            limit=args.limit,
            output_path=args.export_jsonl,
            overwrite=args.overwrite,
            expected_tenant_id=args.expected_tenant_id,
            allow_legacy_local=args.allow_legacy_local,
        )
    else:
        inspect_document(
            args.data_root,
            args.source,
            expected_tenant_id=args.expected_tenant_id,
            allow_legacy_local=args.allow_legacy_local,
        )


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
