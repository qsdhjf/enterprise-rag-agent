"""从多个企业数据源均衡抽取文档切块并生成本地 JSONL 索引。"""

from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from enterprise_agent.ingestion.chunker import DocumentChunk
from enterprise_agent.ingestion.pipeline import iter_source_chunks
from enterprise_agent.ingestion.jsonl_writer import write_chunks_jsonl


def iter_balanced_chunks(
    data_root: Path,
    limit_per_source: int,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> Iterator[DocumentChunk]:
    """按来源目录顺序产出切块，并限制每个来源的文档数量。"""
    # 在创建迭代器入口处验证参数，尽早暴露路径或数量错误。
    if not data_root.is_dir():
        raise ValueError(f"数据根目录不存在：{data_root}")

    if limit_per_source < 1:
        raise ValueError("limit_per_source 必须大于 0")

    # 排序让相同输入在不同机器上也有稳定、可复现的来源顺序。
    source_dirs = sorted(
        path for path in data_root.iterdir() if path.is_dir()
    )

    for source_dir in source_dirs:
        # yield from 保持流式处理，不把所有来源的切块一次性载入内存。
        yield from iter_source_chunks(
            data_root=data_root,
            source=source_dir.name,
            limit=limit_per_source,
            expected_tenant_id=expected_tenant_id,
            allow_legacy_local=allow_legacy_local,
        )


def track_chunk_counts(
    chunks: Iterable[DocumentChunk],
    counts: Counter[str],
) -> Iterator[DocumentChunk]:
    """在不打断流式处理的前提下，累计每个来源的切块数量。"""
    for chunk in chunks:
        counts[chunk.source] += 1
        yield chunk


def main() -> None:
    """解析多来源导出参数，写入索引并输出来源分布统计。"""
    parser = ArgumentParser(
        description="建立多数据源开发索引"
    )

    # --data-root 的每个一级子目录被视作一个独立数据源。
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="企业 sources 数据目录",
    )
    # --output 指定最终 JSONL 文件；默认不允许覆盖。
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出的 JSONL 索引文件",
    )
    # --limit-per-source 防止大来源挤占整个开发索引。
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=10,
        help="每个来源最多读取多少份文档",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出文件",
    )
    acl_mode = parser.add_mutually_exclusive_group()
    acl_mode.add_argument(
        "--expected-tenant-id",
        help="生产摄取的可信目标租户；所有源文档 ACL 必须匹配",
    )
    acl_mode.add_argument(
        "--allow-legacy-local",
        action="store_true",
        help="仅本地演示：为完全缺失 ACL 的旧文档赋固定 demo ACL",
    )

    args = parser.parse_args()

    # Counter 由包装迭代器在切块真正被消费时更新。
    counts: Counter[str] = Counter()

    chunks = iter_balanced_chunks(
        data_root=args.data_root,
        limit_per_source=args.limit_per_source,
        expected_tenant_id=args.expected_tenant_id,
        allow_legacy_local=args.allow_legacy_local,
    )

    tracked_chunks = track_chunk_counts(chunks, counts)

    try:
        # writer 消费上面的惰性迭代链，并返回最终写入总数。
        total = write_chunks_jsonl(
            chunks=tracked_chunks,
            output_path=args.output,
            overwrite=args.overwrite,
            expected_tenant_id=args.expected_tenant_id,
            allow_legacy_local=args.allow_legacy_local,
        )
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"写入切块总数：{total}")

    for source, count in sorted(counts.items()):
        print(f"- {source}: {count} 个切块")

    print(f"输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
