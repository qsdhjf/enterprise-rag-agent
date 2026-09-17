"""用严格 Pydantic 模型约束规划器允许返回的 JSON 动作。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
)

from enterprise_agent.agent.contracts import SourceName


class SearchEnterpriseKnowledgeProposal(BaseModel):
    """规划器提出的一次只读企业知识检索。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    kind: Literal["search_enterprise_knowledge"]

    query: str = Field(
        min_length=1,
        max_length=1000,
    )
    source: SourceName | None

    top_k: int = Field(
        ge=1,
        le=10,
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
    )

    @field_validator("query", "reason")
    @classmethod
    def reject_blank_text(
        cls,
        value: str,
    ) -> str:
        """去除首尾空白，并拒绝视觉上为空的查询和原因。"""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("字段不能只包含空格")

        return cleaned


class FinishProposal(BaseModel):
    """规划器认为现有证据已经足够时提出的结束动作。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    kind: Literal["finish"]

    reason: str = Field(
        min_length=1,
        max_length=500,
    )

    @field_validator("reason")
    @classmethod
    def reject_blank_reason(
        cls,
        value: str,
    ) -> str:
        """规范化结束原因并拒绝纯空白内容。"""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason不能只包含空格")

        return cleaned


# discriminator 先读取 kind，再选择对应模型进行严格校验。
PlannerProposal = Annotated[
    SearchEnterpriseKnowledgeProposal | FinishProposal,
    Field(discriminator="kind"),
]


PLANNER_PROPOSAL_ADAPTER = TypeAdapter(
    PlannerProposal
)


def parse_planner_proposal(
    text: str,
) -> PlannerProposal:
    """把模型原始文本解析为唯一一种合法规划动作。"""

    content = text.strip()

    if not content:
        raise ValueError("Planner返回内容不能为空")

    return PLANNER_PROPOSAL_ADAPTER.validate_json(
        content,
        strict=True,
    )
