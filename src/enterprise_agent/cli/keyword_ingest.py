"""把切块 JSONL 批量写入 OpenSearch 关键词索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.ingestion.opensearch.index import ensure_chunk_index
from enterprise_agent.ingestion.opensearch.ingestion import bulk_index_jsonl


def main() -> None:
    """解析导入参数，校验输入文件并执行批量入库。"""
    parser = argparse.ArgumentParser(
        description="将企业文档切块 JSONL 批量写入 OpenSearch",
    )

    # --jsonl 指定上游数据流水线生成的切块文件。
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="需要导入的切块 JSONL 文件",
    )

    # --batch-size 控制每次提交给 OpenSearch 的操作数量。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="每批发送给 OpenSearch 的切块数量",
    )
    acl_mode = parser.add_mutually_exclusive_group()
    acl_mode.add_argument(
        "--expected-tenant-id",
        help="生产摄取的可信目标租户；JSONL ACL 必须与其一致",
    )
    acl_mode.add_argument(
        "--allow-legacy-local",
        action="store_true",
        help="仅本地演示：接收完全无 ACL 的旧 JSONL 并赋固定 demo ACL",
    )

    args = parser.parse_args()
    jsonl_path = args.jsonl.resolve()

    # 尽早校验路径，避免建立连接后才发现输入无效。
    if not jsonl_path.is_file():
        parser.error(f"JSONL 文件不存在：{jsonl_path}")

    # 连接信息和索引名统一从基础设施配置中读取。
    settings = load_opensearch_settings()
    client = create_opensearch_client(settings)

    try:
        if not client.ping():
            raise RuntimeError("无法连接 OpenSearch")

        # ensure 函数可重复调用：索引存在时只做检查，不会重复创建。
        index_created = ensure_chunk_index(
            client=client,
            index_name=settings.index_name,
            allow_legacy_local=args.allow_legacy_local,
        )

        count_before = client.count(
            index=settings.index_name,
        )["count"]

        # 批量写入函数会返回成功、失败操作的汇总统计。
        result = bulk_index_jsonl(
            client=client,
            index_name=settings.index_name,
            jsonl_path=jsonl_path,
            batch_size=args.batch_size,
            expected_tenant_id=args.expected_tenant_id,
            allow_legacy_local=args.allow_legacy_local,
        )

        count_after = client.count(
            index=settings.index_name,
        )["count"]
    finally:
        # 即使检查或导入失败，也要释放 HTTP 连接池。
        client.close()

    print(f"输入文件：{jsonl_path}")
    print(f"目标索引：{settings.index_name}")
    print(f"索引本次新建：{'是' if index_created else '否'}")
    print(f"导入前文档数：{count_before}")
    print(f"成功操作数：{result.succeeded}")
    print(f"失败操作数：{result.failed}")
    print(f"导入后文档数：{count_after}")


if __name__ == "__main__":
    # 模块直接执行时进入命令行流程。
    main()
