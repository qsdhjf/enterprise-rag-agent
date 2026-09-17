"""OpenAI 兼容的文本向量客户端，并对响应数量、维度和数值做防御性校验。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from openai import OpenAI, DefaultHttpxClient

from enterprise_agent.infrastructure.config import EmbeddingSettings


class OpenAICompatibleEmbeddingClient:
    """封装 OpenAI 兼容 Embedding API 的批量与单条向量化操作。"""

    def __init__(
        self,
        settings: EmbeddingSettings,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        follow_redirects: bool = True,
    ) -> None:
        """根据配置创建底层客户端，并保存后续校验所需的预期维度。"""

        self._model = settings.model
        self._dimensions = settings.dimensions

        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=DefaultHttpxClient(follow_redirects=follow_redirects),
        )

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """按输入顺序将一批非空文本转换成浮点向量。

        Raises:
            ValueError: 输入序列为空，或其中包含空白文本。
            RuntimeError: 服务返回数量、向量维度或向量数值不符合约定。
        """

        if not texts:
            raise ValueError("待向量化的文本不能为空")

        # 统一转成字符串并清理空白，防止把无意义文本发送给模型服务。
        cleaned_texts = [
            str(text).strip()
            for text in texts
        ]

        if any(not text for text in cleaned_texts):
            raise ValueError("待向量化的文本中不能包含空文本")

        response = self._client.embeddings.create(
            model=self._model,
            input=cleaned_texts,
        )

        # 某些兼容服务不保证响应数组本身有序，以响应中的 index 恢复输入顺序。
        ordered_items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        if len(ordered_items) != len(cleaned_texts):
            raise RuntimeError(
                "Embedding返回的向量数量与输入文本数量不一致"
            )
        if [item.index for item in ordered_items] != list(range(len(cleaned_texts))):
            raise RuntimeError("Embedding 返回重复或越界的 index，无法可靠对应原文")

        vectors: list[list[float]] = []

        for item in ordered_items:
            vector = [
                float(value)
                for value in item.embedding
            ]

            if len(vector) != self._dimensions:
                raise RuntimeError(
                    "Embedding向量维度不正确："
                    f"期望 {self._dimensions}，"
                    f"实际 {len(vector)}"
                )

            # NaN/Inf 会污染相似度计算和索引，因此在数据入口立即拒绝。
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(
                    "Embedding向量包含NaN或无穷值"
                )

            vectors.append(vector)

        return vectors

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """向量化单条文本；复用批量路径以保持相同的输入和响应校验。"""

        return self.embed_texts([text])[0]

    def close(self) -> None:
        """释放底层 HTTP 连接资源。"""

        self._client.close()
