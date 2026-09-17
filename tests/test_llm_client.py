"""验证通用 LLM 客户端只在调用方要求时转发 response_format。"""

from types import SimpleNamespace

from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)


class FakeCompletions:
    """记录提交给 OpenAI SDK 的参数并返回最小合法响应。"""

    def __init__(self) -> None:
        """初始化请求记录列表。"""

        self.requests: list[dict[str, object]] = []

    def create(
        self,
        **kwargs: object,
    ) -> SimpleNamespace:
        """记录请求参数并返回固定的最小完成响应。"""

        self.requests.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"kind":"finish"}'
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
            ),
            model="fake-model",
            id="fake-request-001",
        )


def make_client() -> tuple[
    OpenAICompatibleLLMClient,
    FakeCompletions,
]:
    """绕过网络构造客户端，并注入记录请求的 SDK 替身。"""

    completions = FakeCompletions()
    client = object.__new__(
        OpenAICompatibleLLMClient
    )
    client._model = "fake-model"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    return client, completions


def test_response_format_is_forwarded() -> None:
    """验证 JSON 模式会原样传入 chat.completions.create。"""

    client, completions = make_client()

    client.generate(
        system_prompt="Return JSON.",
        user_prompt="Plan the next action.",
        response_format={
            "type": "json_object",
        },
    )

    assert completions.requests[0][
        "response_format"
    ] == {
        "type": "json_object",
    }


def test_response_format_is_omitted_by_default() -> None:
    """验证普通回答调用不会被强制切换成 JSON 模式。"""

    client, completions = make_client()

    client.generate(
        system_prompt="Answer normally.",
        user_prompt="Hello.",
    )

    assert (
        "response_format"
        not in completions.requests[0]
    )
