"""创建或检查 OpenSearch 混合检索索引的命令行入口。"""

from enterprise_agent.infrastructure.config import load_embedding_settings
from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.ingestion.opensearch.hybrid_index import (
    ensure_hybrid_index,
)


def main() -> None:
    """确保混合索引可用，并打印索引配置与当前文档数量。"""
    # 两类配置分别描述 OpenSearch 连接和向量模型维度。
    opensearch_settings = load_opensearch_settings()
    embedding_settings = load_embedding_settings()

    # 客户端属于外部资源，后面的 finally 会保证连接被关闭。
    client = create_opensearch_client(opensearch_settings)

    try:
        if not client.ping():
            raise RuntimeError("无法连接 OpenSearch")

        # 索引不存在时创建；已存在时校验映射是否符合当前配置。
        created = ensure_hybrid_index(
            client=client,
            index_name=opensearch_settings.hybrid_index_name,
            embedding_model=embedding_settings.model,
            dimensions=embedding_settings.dimensions,
        )

        # 读取数量用于给操作者一个直观的索引状态反馈。
        document_count = client.count(
            index=opensearch_settings.hybrid_index_name
        )["count"]

    finally:
        client.close()

    print(
        "操作结果："
        + ("创建新索引" if created else "索引已存在并通过检查")
    )
    print(
        "索引名称："
        f"{opensearch_settings.hybrid_index_name}"
    )
    print(
        "Embedding模型："
        f"{embedding_settings.model}"
    )
    print(
        "向量维度："
        f"{embedding_settings.dimensions}"
    )
    print(f"当前文档数：{document_count}")


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 CLI，作为模块导入时不会自动连接服务。
    main()
