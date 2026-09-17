"""在模型分类前执行确定性安全规则，并提供统一的范围判断入口。"""

from __future__ import annotations

import re
import unicodedata

from enterprise_agent.agent.scope_gate import (
    ScopeGateResult,
    classify_agent_scope,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)


# 高风险提示注入/秘密外泄表达应在调用模型和检索前直接拦截。
SECURITY_PATTERNS = (
    re.compile(
        r"ignore\s+(all\s+|previous\s+|system\s+)?instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"(reveal|show|print|leak).{0,30}"
        r"(system\s+prompt|api\s*key|password|access\s*token)",
        re.IGNORECASE,
    ),
    re.compile(
        r"bypass.{0,20}(authorization|permission|policy|security)",
        re.IGNORECASE,
    ),
    re.compile(r"忽略.{0,20}(规则|指令|提示词)"),
    re.compile(
        r"(显示|泄露|输出).{0,20}"
        r"(系统提示词|API密钥|密码|访问令牌)"
    ),
    re.compile(r"绕过.{0,20}(权限|授权|策略|安全)"),
)


# 当前 Agent 只读，因此明确的外部系统写操作同样由规则优先拒绝。
MUTATION_PATTERNS = (
    re.compile(
        r"^\s*(please\s+)?"
        r"(create|update|edit|delete|close|archive|send|post|deploy|"
        r"approve|merge|push)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthen\s+"
        r"(create|update|edit|delete|close|archive|send|post|deploy|"
        r"approve|merge|push)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(请帮我|请你?|帮我|替我)"
        r"(直接|立即|马上|现在)?"
        r"(创建|新建|修改|更新|删除|关闭|归档|发送|发布|部署|"
        r"审批|批准|合并|推送)"
    ),
    re.compile(
        r"然后(直接|立即|马上|现在)?"
        r"(创建|新建|修改|更新|删除|关闭|归档|发送|发布|部署|"
        r"审批|批准|合并|推送)"
    ),
)


def normalize_scope_input(text: str) -> str:
    """统一 Unicode 表示并移除可能干扰规则匹配的控制字符。"""

    normalized = unicodedata.normalize("NFKC", text)

    # 保留常用换行/制表符，删除其余 Unicode C 类控制与格式字符。
    return "".join(
        character
        for character in normalized
        if (
            character in "\n\r\t"
            or not unicodedata.category(character).startswith("C")
        )
    ).strip()


def _rule_result(
    category: str,
    reason: str,
) -> ScopeGateResult:
    """把确定性规则命中转换为无需模型调用的范围结果。"""

    return ScopeGateResult(
        category=category,
        reason=reason,
        classifier_succeeded=True,
        attempts=0,
        model=None,
        request_id=None,
        input_tokens=0,
        output_tokens=0,
    )


def apply_pre_scope_guard(
    question: str,
) -> ScopeGateResult | None:
    """优先识别安全违规和写操作；未命中时交给模型分类。"""

    for pattern in SECURITY_PATTERNS:
        if pattern.search(question):
            return _rule_result(
                "unsafe_request",
                "DETERMINISTIC_SECURITY_RULE",
            )

    for pattern in MUTATION_PATTERNS:
        if pattern.search(question):
            return _rule_result(
                "unsupported_action",
                "DETERMINISTIC_WRITE_RULE",
            )

    return None


def evaluate_request_scope(
    question: str,
    client: OpenAICompatibleLLMClient,
) -> ScopeGateResult:
    """规范化请求，执行硬规则，再按需调用范围分类模型。"""

    normalized = normalize_scope_input(question)

    if not normalized:
        raise ValueError("问题不能为空")

    if len(normalized) > 2000:
        raise ValueError("问题最多允许2000个字符")

    # 硬规则先运行，可避免明显危险请求进入任何模型或检索链路。
    rule_result = apply_pre_scope_guard(normalized)

    if rule_result is not None:
        return rule_result

    return classify_agent_scope(
        normalized,
        client,
    )
