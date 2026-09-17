# BGE Reranker 部署与检索链路

本项目使用 `BAAI/bge-reranker-v2-m3` 作为 Cross-Encoder 重排模型。模型通过独立的 Hugging Face Text Embeddings Inference（TEI）GPU 容器提供 HTTP 服务，FastAPI 主容器不直接安装 PyTorch 或模型权重。

## 在线检索顺序

```text
用户问题
  ├─ BM25 召回候选
  └─ BGE-M3 向量召回候选
          ↓
       RRF 融合
          ↓
  bge-reranker-v2-m3 对候选逐条联合编码打分
          ↓
  每篇文档最多保留 2 个切块
          ↓
       最终 Top-K
```

Reranker 必须处理 RRF 候选池，而不是只给最终 Top-5 重新排序。否则原来位于第 6～20 名、但语义上更相关的切块没有机会进入最终结果。

## 启动

```powershell
docker compose -f .\infra\opensearch\compose.yaml up -d reranker
docker compose -f .\infra\opensearch\compose.yaml up -d --build api
docker compose -f .\infra\opensearch\compose.yaml ps
```

本地端口：

- TEI health：`http://127.0.0.1:8081/health`
- TEI info：`http://127.0.0.1:8081/info`
- FastAPI readiness：`http://127.0.0.1:8000/health/ready`

模型权重保存在 Docker 命名卷 `enterprise-agent-reranker-model-cache` 中。重建容器不会重新下载权重；执行 `docker compose down -v` 会删除数据卷，不应作为普通停止命令使用。

## 配置

配置项位于 `.env.example`：

- `RERANKER_BASE_URL`
- `RERANKER_MODEL`
- `RERANKER_TIMEOUT_SECONDS`
- `RERANKER_MAX_CANDIDATES`
- `RERANKER_REQUIRED`
- `RERANKER_API_KEY`

默认 `RERANKER_REQUIRED=true`。服务不可用或响应契约错误时，检索链路安全失败；设为 `false` 时，仅对明确的 Reranker 服务错误回退到原 RRF 排序，代码错误不会被静默吞掉。

TEI 的 `max_concurrent_requests`、`max_batch_requests` 和 `max_client_batch_size` 都必须不小于 `RERANKER_MAX_CANDIDATES`。readiness 会检查模型身份、Reranker 类型和这三项容量，避免出现“探针正常但真实候选请求失败”。

## 分数含义

当前请求使用 `raw_scores=true`。输出是 Cross-Encoder 的原始相关性 logit：数值越大表示在同一次查询的候选中越相关，但它不是概率，也不应跨不同查询设置统一业务阈值。

## 验证

```powershell
uv run pytest .\tests\test_reranker_client.py .\tests\test_reranker_readiness.py .\tests\test_reranked_hybrid_search.py -q

uv run enterprise-agent-search `
  --query "How should project lifecycle archival close Slack channels, snapshot Confluence pages, and archive Jira issues?" `
  --source jira `
  --top-k 5
```
