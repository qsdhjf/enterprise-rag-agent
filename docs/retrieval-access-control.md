# 检索与摄取访问控制

检索授权以服务端注入的可信 `PrincipalContext` 为准。查询正文、模型生成参数和普通请求 body 中出现的 `tenant_id`、用户、角色或用户组都不能改变检索身份。`SearchToolRequest` 只负责把运行时主体快照传入检索适配器；BM25、向量和 hybrid 检索使用同一份租户与主体过滤条件，返回结果、缓存命中和 fallback resolve 还会再次校验元数据。

## 文档 ACL 合同

每个生产文档和 chunk 必须显式携带以下四项：

- `tenant_id`：非空租户 ID；摄取任务指定 `expected_tenant_id` 时必须完全相同。
- `visibility`：只允许 `tenant`、`internal`、`public`、`restricted` 或 `private`。前三项表示租户内可见；后两项还必须至少提供一个 ACL 主体。
- `acl_principals`：数组，元素只能是非空的 `user:<id>`、`role:<id>` 或 `group:<id>`，命名空间不可省略。
- `acl_version`：非空授权版本，用于审计、缓存复验和授权变更失效。

原始 JSON 可以把完整四项放在顶层、`access_control`、`metadata.access_control`，或明确的 metadata ACL 区域。多个位置同时声明时，内容必须一致；冲突、部分缺失、未知 visibility、无前缀主体、空版本以及跨租户数据都会拒绝摄取。ACL 会沿 `EnterpriseDocument → DocumentChunk → JSONL → OpenSearch _source` 原样传播；keyword 与 hybrid 摄取执行同一套严格校验。

生产示例：

```json
{
  "tenant_id": "tenant-a",
  "visibility": "restricted",
  "acl_principals": ["user:alice", "group:engineering"],
  "acl_version": "acl-v7"
}
```

## 本地演示兼容

旧生成数据可能完全没有 ACL 字段。此类数据默认失败关闭，不能进入生产索引。只有调用名称明确的 local-demo reader/helper，或在 keyword/hybrid 摄取时显式启用 local-demo 兼容参数，才会补充固定值：

```json
{
  "tenant_id": "local-demo-tenant",
  "visibility": "local-demo",
  "acl_principals": [],
  "acl_version": "local-dev-v1"
}
```

该兼容分支只接受完全缺失 ACL 的旧记录，并只对固定本地租户生效。部分 ACL、畸形 ACL，或已经声明其他租户的记录仍会被拒绝；不要在生产任务中启用它。

## 运维检查

构建索引前应为任务设置预期租户并保留错误行号；ACL 校验失败应中止或隔离该记录，不能改成通用可见性继续写入。授权规则变化时应递增 `acl_version`，并重建或更新相应文档。上线前至少验证：跨租户检索为空、无权限主体无法读取 restricted/private 文档、缓存与 fallback 不会绕过 ACL、旧无 ACL 索引仅能由固定 local-demo 身份访问。

四个摄取 CLI 使用同一组互斥参数：生产任务传
`--expected-tenant-id <tenant>`，旧本地样本必须显式传
`--allow-legacy-local`。不传本地开关时，无 ACL 或部分 ACL 的记录一律拒绝。
生产环境发现已有 strict 索引缺少任何 ACL keyword 映射时会报
`ACL_INDEX_MAPPING_INCOMPLETE`，应重建索引；本地兼容开关只会受控补齐
缺失映射，不能修复字段类型错误。
