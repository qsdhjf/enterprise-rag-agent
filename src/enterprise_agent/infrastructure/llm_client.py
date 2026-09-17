"""OpenAI 兼容的对话生成客户端及其统一响应对象。"""

from dataclasses import dataclass
from typing import Any

from openai import OpenAI, DefaultHttpxClient

from enterprise_agent.infrastructure.config import LLMSettings


@dataclass(frozen=True, slots=True)
class LLMResult:
    """一次模型调用的正文、模型标识、请求标识和可选用量信息。"""

    text: str
    model: str
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None


class OpenAICompatibleLLMClient:
    """将系统提示词和用户提示词提交给 OpenAI 兼容的聊天接口。"""

    def __init__(
        self,
        settings: LLMSettings,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        follow_redirects: bool = True,
    ) -> None:
        """根据已校验的配置创建底层客户端。"""

        self._model = settings.model
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=DefaultHttpxClient(follow_redirects=follow_redirects),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
        response_format: dict[str, object] | None = None,
    ) -> LLMResult:
        """生成一条回答并整理成稳定的 ``LLMResult``。

        Raises:
            ValueError: 系统提示词或用户提示词为空。
            RuntimeError: 服务未返回候选答案，或候选答案正文为空。
        """

        if not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")

        if not user_prompt.strip():
            raise ValueError("user_prompt 不能为空")

        # 消息角色分离：system 约束行为，user 承载本次任务和上下文。
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            # 兼容支持该扩展字段的服务，关闭额外思考输出以便稳定解析正文。
            "extra_body": {
                "thinking": {"type": "disabled"}
            },
        }

        # 仅由需要结构化输出的调用方显式开启，避免影响普通回答。
        if response_format is not None:
            request["response_format"] = response_format

        response = self._client.chat.completions.create(
            **request
        )

        if not response.choices:
            raise RuntimeError("模型没有返回任何候选答案")

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("模型返回了空答案")

        # 部分兼容服务不提供 usage，因此令牌数在结果中允许为 None。
        usage = response.usage

        return LLMResult(
            text=answer.strip(),
            model=response.model or self._model,
            request_id=response.id,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
    def close(self) -> None:
        """释放底层 HTTP 连接资源。"""

        self._client.close()
