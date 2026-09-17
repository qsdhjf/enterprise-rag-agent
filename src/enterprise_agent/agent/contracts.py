"""定义运行 Agent 所需的可信身份、检索权限和资源预算。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# 所有可被策略层批准的企业数据来源；新增来源时需同步检查检索实现。
SourceName = Literal[
    "confluence",
    "fireflies",
    "github",
    "gmail",
    "google_drive",
    "hubspot",
    "jira",
    "linear",
    "slack",
    "uploaded_file",
]


class PrincipalContext(BaseModel):
    """由认证系统提供的可信用户身份，不能由用户问题正文构造。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tenant_id: str = Field(
        min_length=1,
        max_length=128,
    )
    subject_id: str = Field(
        min_length=1,
        max_length=256,
    )

    roles: frozenset[str] = Field(
        default_factory=frozenset,
    )
    groups: frozenset[str] = Field(
        default_factory=frozenset,
    )

    authz_version: str = Field(
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_identity(self) -> PrincipalContext:
        """保证身份至少能通过角色或用户组参与授权判断。"""

        if not self.roles and not self.groups:
            raise ValueError(
                "用户必须至少拥有一个角色或用户组"
            )

        return self

class RetrievalPolicy(BaseModel):
    """服务端根据用户权限生成的检索策略。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    policy_version: str = Field(
        default="deny-all-v1",
        min_length=1,
        max_length=128,
    )

    allowed_sources: frozenset[SourceName] = Field(
        default_factory=frozenset,
    )

    allow_cross_source: bool = False

    max_top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

class RunBudget(BaseModel):
    """一次 Agent 任务不可突破的步骤、工具、时间和 token 上限。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    max_steps: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    max_tool_calls: int = Field(
        default=5,
        ge=1,
        le=30,
    )
    max_runtime_seconds: int = Field(
        default=120,
        ge=5,
        le=1800,
    )
    max_model_tokens: int = Field(
        default=12000,
        ge=500,
        le=200000,
    )
    max_retrieved_chunks: int = Field(
        default=50,
        ge=1,
        le=500,
    )


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """一次调用的可信运行上下文，由服务端传入而非写入图状态。"""

    request_id: str
    principal: PrincipalContext
    budget: RunBudget

    retrieval_policy: RetrievalPolicy = field(
        default_factory=RetrievalPolicy,
    )
