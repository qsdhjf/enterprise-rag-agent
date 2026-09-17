"""生成文档切块向量并批量写入 OpenSearch 混合索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_agent.infrastructure.config import load_embedding_settings
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.ingestion.opensearch.hybrid_index import (
    ensure_hybrid_index,
)
from enterprise_agent.ingestion.opensearch.hybrid_ingestion import (
    bulk_index_hybrid_jsonl,
)


def main() -> None:
    """校验导入参数，确保索引存在，再执行带 Embedding 的批量入库。"""
    parser = argparse.ArgumentParser(
        description="生成切块向量并写入 OpenSearch 混合索引"
    )
    # --jsonl 指向已经标准化并切块的数据文件。
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="需要写入的切块 JSONL 文件",
    )
    # --embedding-batch-size 控制单次模型请求包含的切块数。
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="每次交给 Embedding 模型的切块数量",
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

    # 建立任何外部连接前先完成本地参数校验。
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"JSONL 文件不存在：{jsonl_path}"
        )

    if args.embedding_batch_size <= 0:
        raise ValueError(
            "--embedding-batch-size 必须大于 0"
        )

    # OpenSearch 与 Embedding 使用各自的配置对象和客户端。
    opensearch_settings = load_opensearch_settings()
    embedding_settings = load_embedding_settings()

    opensearch_client = create_opensearch_client(
        opensearch_settings
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        embedding_settings
    )

    try:
        if not opensearch_client.ping():
            raise RuntimeError("无法连接 OpenSearch")

        # 混合索引映射必须同时兼容文本字段和当前向量维度。
        index_created = ensure_hybrid_index(
            client=opensearch_client,
            index_name=opensearch_settings.hybrid_index_name,
            embedding_model=embedding_settings.model,
            dimensions=embedding_settings.dimensions,
            allow_legacy_local=args.allow_legacy_local,
        )

        # 入库前后计数便于人工核对实际写入影响。
        before_count = opensearch_client.count(
            index=opensearch_settings.hybrid_index_name
        )["count"]

        print("开始生成向量并写入 OpenSearch……", flush=True)
        print(f"输入文件：{jsonl_path}", flush=True)
        print(
            f"Embedding 批次大小："
            f"{args.embedding_batch_size}",
            flush=True,
        )

        # 该适配层负责分批生成向量，并将文本与向量一起批量提交。
        result = bulk_index_hybrid_jsonl(
            client=opensearch_client,
            index_name=opensearch_settings.hybrid_index_name,
            jsonl_path=jsonl_path,
            embedding_client=embedding_client,
            embedding_batch_size=args.embedding_batch_size,
            expected_tenant_id=args.expected_tenant_id,
            allow_legacy_local=args.allow_legacy_local,
        )

        after_count = opensearch_client.count(
            index=opensearch_settings.hybrid_index_name
        )["count"]
    finally:
        # 任一外部请求失败时也要释放两类客户端资源。
        embedding_client.close()
        opensearch_client.close()

    print("\n========== 向量入库结果 ==========")
    print(f"索引：{opensearch_settings.hybrid_index_name}")
    print(f"本次是否新建索引：{index_created}")
    print(f"生成向量数量：{result.embedded}")
    print(f"写入成功数量：{result.succeeded}")
    print(f"写入失败数量：{result.failed}")
    print(f"写入前索引数量：{before_count}")
    print(f"写入后索引数量：{after_count}")

    # 部分失败也以异常结束，避免流水线误认为产物完整。
    if result.failed:
        raise RuntimeError(
            f"存在 {result.failed} 个写入失败的切块"
        )


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
