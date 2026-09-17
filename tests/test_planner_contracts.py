"""验证规划模型输出必须严格符合动作契约。

这些测试展示搜索、结束两种合法响应，并确认未知动作、多余字段和 Markdown 包装都会被拒绝。
"""

import pytest
from pydantic import ValidationError

from enterprise_agent.agent.planner_contracts import (
    FinishProposal,
    SearchEnterpriseKnowledgeProposal,
    parse_planner_proposal,
)


def test_parse_search_proposal() -> None:
    """验证合法的搜索动作 JSON 会解析为搜索提案并保留关键字段。"""
    proposal = parse_planner_proposal(
        """
        {
          "kind": "search_enterprise_knowledge",
          "query": "project lifecycle archival",
          "source": "jira",
          "top_k": 5,
          "reason": "需要检索企业证据"
        }
        """
    )

    assert isinstance(
        proposal,
        SearchEnterpriseKnowledgeProposal,
    )
    assert proposal.source == "jira"
    assert proposal.top_k == 5


def test_parse_finish_proposal() -> None:
    """验证合法的结束动作 JSON 会解析为结束提案。"""
    proposal = parse_planner_proposal(
        """
        {
          "kind": "finish",
          "reason": "现有证据已经足够"
        }
        """
    )

    assert isinstance(proposal, FinishProposal)


def test_unknown_action_is_rejected() -> None:
    """验证规划器契约会拒绝未定义的危险动作类型。"""
    with pytest.raises(ValidationError):
        parse_planner_proposal(
            """
            {
              "kind": "delete_database",
              "reason": "执行危险操作"
            }
            """
        )


def test_extra_field_is_rejected() -> None:
    """验证严格契约会拒绝模型自行添加的多余字段。"""
    with pytest.raises(ValidationError):
        parse_planner_proposal(
            """
            {
              "kind": "finish",
              "reason": "调查结束",
              "action_id": "model-created-id"
            }
            """
        )


def test_markdown_is_rejected() -> None:
    """验证解析器只接受原始 JSON，不接受 Markdown 代码块包装。"""
    with pytest.raises(ValidationError):
        parse_planner_proposal(
            """
            ```json
            {
              "kind": "finish",
              "reason": "调查结束"
            }
            ```
            """
        )
