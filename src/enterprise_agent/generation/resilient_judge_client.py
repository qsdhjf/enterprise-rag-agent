"""为证据审核模型增加一次严格 JSON 校验后的自动纠错重试。"""

from __future__ import annotations

import json

from enterprise_agent.generation.evidence_judge import (
    _parse_json_object,
    _validate_judge_output,
)
from enterprise_agent.infrastructure.llm_client import (
    LLMResult,
    OpenAICompatibleLLMClient,
)


class RetryingEvidenceJudgeClient:
    """只供证据审核节点使用的单次自动纠错包装器。

    对外保持与普通 LLM 客户端相同的 ``generate`` 接口，同时记录实际调用数、
    重试数及最后一次校验错误，便于评估结构化输出的稳定性。
    """

    def __init__(
        self,
        client: OpenAICompatibleLLMClient,
    ) -> None:
        """保存被包装客户端，并初始化调用与错误统计。"""

        self._client = client
        self.total_calls = 0
        self.retry_count = 0
        self.last_validation_error: str | None = None

    @staticmethod
    def _validate_response(
        response: LLMResult,
        user_prompt: str,
    ) -> str | None:
        """按原审核请求中的证据数量校验模型响应，返回错误文字或 ``None``。"""

        try:
            # 从请求中读取证据数，随后可拒绝响应引用 E<n> 范围外的证据。
            request_payload = json.loads(user_prompt)
            evidence = request_payload["evidence"]

            if not isinstance(evidence, list):
                raise ValueError(
                    "审核请求中的evidence必须是列表"
                )

            response_payload = _parse_json_object(
                response.text
            )
            _validate_judge_output(
                payload=response_payload,
                evidence_count=len(evidence),
            )
        # 校验器只返回诊断文字，让调用方决定是否重试，而不在此处终止流程。
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return str(error)

        return None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> LLMResult:
        """调用审核模型；首次结构校验失败时追加错误原因并重试一次。

        第二次响应无论是否合格都会返回，因为上层 ``judge_evidence`` 还会执行
        最终权威校验，并在不合格时生成安全失败结果。
        """

        first_response = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        self.total_calls += 1

        validation_error = self._validate_response(
            first_response,
            user_prompt,
        )

        if validation_error is None:
            self.last_validation_error = None
            return first_response

        self.retry_count += 1
        self.last_validation_error = validation_error

        # 把具体校验错误写入系统提示词，帮助模型只修正格式/字段问题。
        retry_system_prompt = (
            system_prompt
            + "\n\nYour previous response failed strict JSON "
            "validation. Return a corrected JSON object only. "
            "Do not add Markdown or explanations outside JSON. "
            f"Validation error: {validation_error}"
        )

        retry_response = self._client.generate(
            system_prompt=retry_system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        self.total_calls += 1

        second_error = self._validate_response(
            retry_response,
            user_prompt,
        )

        if second_error is None:
            self.last_validation_error = None
        else:
            self.last_validation_error = second_error

        # 最终是否合格仍由原 Evidence Judge 再做一次权威校验。
        return retry_response
