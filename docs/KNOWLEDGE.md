# 个人文件知识库 / Personal document workspaces

0.5.0 supports upload → parse → optional incremental vectors → retrieval → Agent audit → source preview. This is **document storage,
not the previously removed conversation-memory feature**. Personal files do not
enter the EnterpriseRAG-Bench corpus or OpenSearch alias.

## 本机先用起来

```powershell
uv sync --locked
uv run enterprise-agent-serve --mode demo --enable-setup --port 8010
```

打开 http://127.0.0.1:8010 → **我的知识库** → 填写名称并创建 → 上传文件 →
等待“可用” → 确认问题输入框上方的“检索范围”是自己的知识库 → 提问。
点击证据卡上的“查看文档对应原文”，查看实际参与检索的文字及位置。

网页支持拖放、多文件选择，每批最多 20 个文件。上传完成只是进入后台队列。
状态：`queued` 排队 → `parsing` 解析 → `indexing` 写入切块 → `ready` 可用；
失败显示 `failed` 和原因，可重试或替换文件。“建立检索数据”不代表生成向量：
默认建立的是本地关键词检索数据。语义索引必须另外点击“构建 / 重建”，明确同意调用向量模型后才建立。

Windows 后台启动（不安装系统服务、不修改开机启动项）：

```powershell
.\scripts\start-workspace.ps1 -EnableSetup
# 只停止已核对为本项目的 demo 进程，保留文件。
.\scripts\stop-workspace.ps1
```

默认保存在工作目录的 `.local/knowledge/knowledge.sqlite3`，不是临时内存。
重启后资料保留，未完成的文件解析重新排队；中断的向量任务标为失败，需要重新确认费用后构建。同一目录只允许一个实例；不要开多个
Uvicorn worker 或把 SQLite 放在网络盘。

**本机 demo 权限边界：** 上传仅允许 loopback 客户端和 localhost/127.0.0.1
Host。HttpOnly、SameSite=Strict Cookie 区分浏览器会话；刷新和服务重启不清空
身份，清除 Cookie、换浏览器或会话一年后过期则不能自动访问原库。这不是账号
体系。重要资料请使用下面的认证模式。不要将无认证 demo 代理成公共服务。

## 稳定身份，不需要 OpenSearch

先创建新的身份配置；命令只显示一次令牌，请保存到密码管理器。
`--source github` 是复用的身份契约字段，不会把文件发送到 GitHub。

```powershell
uv run enterprise-agent-identity --tenant my-team --subject alice --source github --output .local/workspace-identities.json
uv run enterprise-agent-serve --mode workspace --identity-file .local/workspace-identities.json --port 8010
```

页面填写令牌，点击“连接我的知识库”。资料绑定服务端身份的 **tenant + subject**，
同租户不同用户也不能互读；轮换令牌但 tenant/subject 不变仍能访问。
修改或撤销身份需要重启服务；没有用户注册、SSO、邀请、共享库或权限管理页面。
已有企业 `live` 模式也提供此个人空间，两种索引相互独立。

容器：先生成上述 `.local/workspace-identities.json`，然后运行
`docker compose -f compose.workspace.yaml up --build`（默认端口 8011）。资料
保存在命名卷，不要 `down -v`。Compose demo 仅展示样例；端口转发不等于应用内
loopback，容器上传应使用认证 workspace，不放宽本机模式的来源检查。

## 模型向导与完整 Agent

以 `--enable-setup` 启动后，本机浏览器出现“首次使用 · 模型配置”：

1. **保存对话模型**：填写 OpenAI 兼容服务的完整 base URL、模型 ID、API Key。
2. **测试连接**：需要确认，仅发送固定测试文本；可能计费，不发送文件。
3. 创建知识库并上传文件；就绪后勾选“使用模型回答”，默认选择“完整 Agent”。

设置向导只接受本机真实 loopback 连接与 Host，认证模式还需要令牌。
远程服务必须 HTTPS；本机服务可 HTTP。保存到 `.local/knowledge/provider-settings.private.json`，
不回显密钥，不存入浏览器 localStorage。文件**没有加密**，请保护系统权限，不要提交仓库。
留空只保留同服务地址的旧密钥；换地址必须重新输入，防止误传给另一供应商。
删除配置不会删除文件或擦除历史向量，但不会再用未配置的服务检索。

个人完整 Agent 复用 `agent/graph.py`：身份/预算校验 → Planner 规划检索 →
策略校验 → 检索 → 模型逐条审核事实 → 缺证据时补充检索 → 最终带引用回答。
每次只绑定一个当前用户的知识库；内部来源 `uploaded_file` 不代表第十个企业 SaaS。
个人入口不运行企业业务范围分类器，只能读取当前库，不能调用企业索引或写工具。
最多 3 次检索、每次最多 3 块、累计最多 10 条证据；预算 24,000 model tokens、
180 秒，网络超时与预算在调用边界检查，不能撤销已发出的请求。
Planner/Judge 有格式纠错，模型调用次数不等于检索次数。

页面显示实际执行节点、事实缺口、引用和调用数。**模型审核不是人工正确性保证**；
向量/重排费用不包含在 LLM token 数里。缺失供应商 usage 显示未知。

## 可选：混合检索和重排

1. 在向导保存向量服务并测试：OpenAI 兼容 embeddings API，模型名与维度必须匹配实际部署。
2. 文件都就绪后，点击“预估并构建 / 继续索引”，先看可复用/待发送切块、字符量和批次数，再确认发送**需新生成向量的切块与文件名**到向量服务。
3. 就绪后在完整 Agent 中选择“BM25 + 向量混合”。查询只向量化本次检索问题。
4. 若需要重排，配置 Hugging Face TEI 地址和实际模型 ID，测试 `/health`、`/info`、`/rerank`；
   服务容量字段须 ≥20。再选择“混合检索 + TEI 重排”，候选原文也会发送给该服务。

每种召回最多取 20 块，RRF 常数 60，重排最多 20 个候选。向量存在 SQLite，
采用精确余弦计算而不是分布式 ANN；语义索引仅支持每库 1–2000 个切块。
增删改文件或更换向量模型/地址/维度会让索引失效，明确报错，不偷偷降级成关键词检索。
0.5 开始复用同一模型配置下未变动文件的切块；服务地址、模型、维度、文件版本和实际
向量化输入哈希必须匹配。文件名也是输入的一部分。不同用户、不同知识库不共享缓存。
修改一个文件会重新解析/向量化该文件，不保证只重算该文件内部改变的段落。
再次上传同名、相同字节的就绪文件会作为无变更操作处理，不增加版本或清掉索引。

每成功一批（最多 16 块）就落库。后续失败或服务重启时，已保存批次可在**再次确认**后复用；
未完成索引不会用于问答，服务重启不会自动继续付费调用。预估绑定当前文件与模型，
确认后若发生变化则拒绝，要求重新预估。字符数不是 token 数或费用金额。
取消只阻止后续批次/写入，已发出的请求可能计费；响应丢失或提交前崩溃可能使重试重复计费，
不是供应商“恰好一次”保证。只有明确需要时才勾选“强制全部重算”（例如同名模型更新权重）。

“清理语义索引”删除该库的向量和可复用批次、取消任务，保留原文件、切块和关键词检索。
操作不可撤销，再生成可能计费；SQLite 主文件可能不会立即缩小，也不清除外部备份。
文件解析和向量调用分别有一个后台线程，仍然仅允许单个服务进程拥有该目录。

## 兼容的环境变量配置和简易问答

默认只有 **本地 BM25 召回 + 原文摘录**，不把文件发送到外部。
配置 `.env` 的 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 后显式开启：

```powershell
uv run enterprise-agent-serve --mode workspace --identity-file .local/workspace-identities.json --enable-knowledge-model --port 8010
# 本机试用也可以 demo + --enable-knowledge-model。
```

选择自己的库，再勾选“使用模型回答”，将问答方式改为“简易问答”。仅本次问题、命中的最多 5 个切块及
文件名/位置发送给配置的供应商，不上传完整知识库。使用角色分离提示词、
结构化输出和引用 ID 校验。模型失败或未知引用返回错误，不假装成功。没有
命中不调用模型。等待期间资料删除/更新时，返回前校验版本并拒绝失效证据。

**引用 ID 存在不等于语义已验证。** 页面标注“模型回答 · 请核对原文”，不伪装
成完整 Agent 的审核结果（完整 Agent 是上面的另一个明确选项）。未返回 token 用量时显示
“未知”，不是编造零费用。“停止等待”只取消浏览器等待，不保证供应商取消
调用或停止收费。服务器模型调用也占用有限并发槽。

## 格式、配额与边界

| 格式 | 提取范围 | 位置 |
|---|---|---|
| TXT / MD | UTF-8 正文，纯文本展示，不执行 HTML | 行范围 |
| PDF | 文本层，不做 OCR，不保证复杂表格/多栏顺序 | 页码 |
| DOCX | 主文档段落和表格单元格文字 | 段落序号，不是 Word 页码 |

不支持 `.doc`、Excel、PPT、加密 PDF、扫描图片 OCR。空白/纯扫描 PDF 明确
失败；混合 PDF 对没有提取出文字的页显示警告，不能把这些页计为已经读取。
DOCX 图片、批注、页眉不索引；压缩包不落地展开、不跟随外部关系链接。
参见 [pypdf 文字提取限制](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)。

- 单文件 ≤ 8 MiB，提取文字 ≤ 50 万字符，PDF ≤ 300 页，单文件 ≤ 1200 块。
- 每块最多 900 字符，长段相邻块最多重叠 200 字符，不跨 PDF 页或 DOCX 段落。
- 每身份 ≤ 20 个库、200 个文件、128 MiB 原文件、10,000 块。
- 服务总计 ≤ 200 个库、512 MiB 活跃原文件；不是大规模部署的容量承诺。
- 默认逻辑向量载荷每身份 ≤128 MiB、全局 ≤512 MiB，排队任务先预留空间，取消/失败会释放预留。
  页面显示当前身份用量，不显示其他身份数据。已有向量与重算预留会同时计入，可能保守拒绝重算。
- 文件写入/向量构建前检查剩余磁盘空间，默认至少保留 256 MiB 及写入余量。
  这不是硬磁盘配额；SQLite 页、WAL、索引元数据、迁移备份、其他程序写入需额外空间。
- 每进程最多 2 个上传请求、16 个其他写请求同时缓冲/处理，请求体总读取时间 ≤15 秒。
  慢网络可能收到 408，请重试或使用更小文件；并发满返回 429。仍需代理层限流。
- DOCX 压缩比/展开大小限制；解析子进程最长 30 秒、RSS 监测阈值 512 MiB。
  POSIX 另有内核虚拟内存限制；Windows 轮询 RSS 不能保证瞬间不越界。
- 子进程不继承服务密钥，但不是完整安全沙箱。公开接收不可信文件前仍需要
  独立容器、出站网络限制、权限隔离和安全审计。

内容改变时替换文件保持文档 ID、版本递增，旧块立即退出检索，新解析成功后可用；失败
不会悄悄恢复旧版。删除移除数据库当前原文件/切块，旧后台任务不得重新写回。
这不是法证意义的安全擦除：备份、WAL、崩溃留下的解析临时目录可能留有副本，
按组织数据保留策略由管理员清理。个人库没有与外部 SaaS 自动同步删除。

## 备份与恢复

```powershell
uv run enterprise-agent-knowledge-backup --data-dir .local/knowledge --output .local/backups/knowledge-20260917.sqlite3
```

使用 SQLite 在线备份 API，包含原文件、切块、状态和会话哈希；检查完整性，
拒绝覆盖已有备份。**这是私密数据，不能放 GitHub、Release 或公共网盘。**
需要加密和独立权限，身份配置另行私密备份，令牌原文不在知识库数据库中。
0.5 数据库版本为 3，含向量输入指纹、批次进度与额度预留；从版本 1/2 首次启动时先生成
`before-v3-*.sqlite3` 私密备份再事务迁移，不删除旧文档。能确定归属和模型的 v2 就绪向量
可继续复用；无法确认模型的旧失败任务向量需重新生成。旧版程序不能打开版本 3。
模型配置 JSON 不在 SQLite 备份内，需要另行私密备份。若回退应用版本，停止服务，
把迁移前备份恢复到新目录；迁移之后的新数据不在该备份里，不要覆盖原目录。

恢复到新目录：停止相关服务，新建空私密目录，将备份复制为其中的
`knowledge.sqlite3`，然后用 `--knowledge-dir 新目录` 启动。保留旧目录核对，
不要混入旧 `-wal/-shm` 文件。未完成的文件解析重启后重新处理，向量任务需再次确认；浏览器 Cookie 必须
仍有效，或使用相同 tenant/subject 的认证身份。跨模式所有权迁移没有自动工具。
建议用合成文件定期演练，不只检查备份文件存在。

## API 与运维

前缀 `/api/knowledge`，所有资料操作校验所有权，不接受客户端传 owner。

- `GET /session` 建立本机会话/校验令牌，返回能力与配额。
- `GET/POST /libraries`，`DELETE /libraries/{id}`。
- `GET/POST /libraries/{id}/documents`。
- `GET/PUT/DELETE /libraries/{id}/documents/{doc}`，`POST .../{doc}/retry`。
- `POST /libraries/{id}/ask`：`{"question":"...","answer_mode":"local"}`。
- 完整 Agent：`{"question":"...","answer_mode":"agent","retrieval_strategy":"bm25"}`；可选 `hybrid` / `reranked`。
- `GET /setup`、`PUT /setup/{llm|embedding|reranker}`：仅显式启用的本机向导。
- `POST /setup/{kind}/test`：`{"consent":true}`，可能调用付费服务。
- `GET/POST/DELETE /libraries/{id}/index`：状态 / 确认构建 / 取消；POST 需要 `{"consent":true}`。
- `GET /libraries/{id}/index/plan?force=false`：只预估，不调用模型；返回 `plan_id`。
- 建议 POST 使用 `{"consent":true,"plan_id":"预估返回的哈希","force":false}`，状态变化时返回 409。
- `DELETE /libraries/{id}/vectors`：清理该库全部向量（不删文件），不能恢复。
- `GET /storage`：当前身份的原文件、向量与预留用量。

资源额度可通过启动进程的环境变量配置（正整数字节数）：`AGENT_VECTOR_BYTES_LIMIT`、
`AGENT_OWNER_VECTOR_BYTES_LIMIT`、`AGENT_MIN_FREE_BYTES`。离线 demo 不自动加载 `.env`；
PowerShell 请先 `$env:AGENT_VECTOR_BYTES_LIMIT="536870912"`，容器用 `environment`。
调整额度后重启；降低额度不会自动删除已有数据，但可能拒绝新任务。

## 可复现的个人检索评测

```powershell
uv run enterprise-agent-eval-knowledge --output artifacts/personal-bm25.json
# 可选：需要私有配置，明确确认外部文本传输和费用。不要未经授权运行真实资料。
uv run enterprise-agent-eval-knowledge --fixture .local/held-out.json --methods bm25 vector hybrid reranked --providers .local/knowledge/provider-settings.private.json --allow-external --output .local/reports/comparison.json
```

默认用独立临时库解析 6 篇原创合成文档、10 个问题，不修改你正在使用的库。
报告含数据指纹、参数、逐题排序、Hit/Recall/MRR/nDCG@K、p50/p95 和待人工填写的
回答审核模板。真实评测先准备有授权、固定快照、独立人工标注的 fixture，格式参见
`src/enterprise_agent/knowledge/eval_fixture.json` 与 [评测协议](../eval/PERSONAL_KNOWLEDGE.md)。
这里的 K 是生产路径的切块预算，之后再按文档去重，不额外补足文档数量。
无答案样本的空召回率不是模型拒答率。模板的正确性字段默认空，**尚未自动测量回答质量**。

上传是原始请求体 + `?filename=...`，不是 multipart；其他 JSON 接口仍限制
16 KiB。预览使用 `offset`、`limit`（≤100）。状态码：401 需登录，403 来源受限，
404 不存在或无权限，409 同名/状态冲突，413 配额，422 参数，429 问答并发已满，
502 模型或引用失败。`X-Request-ID` 关联日志；日志不打印正文、令牌或模型响应。

公网仍需 HTTPS、正确代理配置、上传并发/速率限制、身份管理、磁盘监控、备份
轮换、恶意文件隔离和安全审查。全局并发槽不是按用户配额，也不是 DDoS 防护。
未承诺多机、多 worker、流式 token、原生 SaaS 权限同步或真实企业问答质量。

CLI 不信任转发头来判断本机客户端。认证模式放在 TLS 反向代理后面时，将
`AGENT_PUBLIC_ORIGIN` 设置为用户访问的准确来源，例如 `https://rag.example.org`，
用于浏览器 Origin 校验。不要用通配符，也不要把 demo 用于这种部署。
