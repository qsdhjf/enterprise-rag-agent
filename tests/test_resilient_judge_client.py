"""验证证据审核客户端面对非法模型输出时的有限重试策略。

顺序伪客户端记录调用次数，用于区分首次成功、一次修复成功和连续失败关闭三种情况。
"""

import json

from enterprise_agent.generation.evidence_judge import judge_evidence
from enterprise_agent.infrastructure.llm_client import LLMResult
from enterprise_agent.retrieval.keyword_search import OpenSearchHit
from enterprise_agent.generation.resilient_judge_client import (
    RetryingEvidenceJudgeClient,
)


def make_hit() -> OpenSearchHit:
    """构造一条供证据审核测试使用的固定检索命中。"""
    return OpenSearchHit(
        score=1.0,
        chunk_id="chunk-001",
        document_id="document-001",
        source="jira",
        title="Lifecycle design",
        sequence=0,
        text="The design requires Slack channels to become read-only.",
        created_at=None,
        original_location="sources/jira/design.json",
    )


def valid_response_text() -> str:
    """生成符合审核契约的 supported 决策 JSON 文本。"""
    return json.dumps(
        {
            "decision": "supported",
            "facts": [
                {
                    "fact": "Slack archival behavior",
                    "status": "supported",
                    "conclusion": (
                        "The design requires read-only channels."
                    ),
                    "evidence_ids": ["E1"],
                }
            ],
            "reason": "The supplied evidence supports the fact.",
        }
    )


class SequenceClient:
    """按预设顺序返回文本并记录调用次数的假 LLM 客户端。"""

    def __init__(self, texts: list[str]) -> None:
        """保存待返回文本序列，并将调用计数初始化为零。"""
        self._texts = texts
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """返回当前序号的模拟结果，并把调用计数推进一次。"""
        text = self._texts[self.calls]
        self.calls += 1
        return LLMResult(
            text=text,
            model="fake-model",
            request_id=f"request-{self.calls}",
            input_tokens=10,
            output_tokens=10,
        )


def test_valid_first_response_is_not_retried() -> None:
    """验证首次审核响应有效时直接成功且不会触发重试。"""
    delegate = SequenceClient([valid_response_text()])
    client = RetryingEvidenceJudgeClient(delegate)

    result = judge_evidence(
        question="How should Slack archival work?",
        evidence=[make_hit()],
        client=client,
    )

    assert result.judge_succeeded is True
    assert delegate.calls == 1
    assert client.retry_count == 0


def test_invalid_first_response_is_retried() -> None:
    """验证首次响应非法时仅重试一次，并接受随后合法的响应。"""
    delegate = SequenceClient(
        ["not json", valid_response_text()]
    )
    client = RetryingEvidenceJudgeClient(delegate)

    result = judge_evidence(
        question="How should Slack archival work?",
        evidence=[make_hit()],
        client=client,
    )

    assert result.judge_succeeded is True
    assert delegate.calls == 2
    assert client.retry_count == 1
    assert client.last_validation_error is None


def test_two_invalid_responses_fail_closed() -> None:
    """验证连续两次非法响应会以审核失败状态保守关闭。"""
    delegate = SequenceClient(["not json", "still not json"])
    client = RetryingEvidenceJudgeClient(delegate)

    result = judge_evidence(
        question="How should Slack archival work?",
        evidence=[make_hit()],
        client=client,
    )

    assert result.judge_succeeded is False
    assert result.error_code == "INVALID_JUDGE_OUTPUT"
    assert delegate.calls == 2
    assert client.retry_count == 1
    assert client.last_validation_error is not None
