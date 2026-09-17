"""调用 Hugging Face 文本推理服务的严格重排客户端。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from enterprise_agent.infrastructure.config import RerankerSettings


class RerankerServiceError(RuntimeError):
    """重排服务违反传输或响应数据契约时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class RerankerScore:
    """一条重排分数及其对应的从零开始输入文本位置。"""

    index: int
    score: float


class TEIRerankerClient:
    """严格封装 Hugging Face TEI ``POST /rerank`` 接口。"""

    def __init__(
        self,
        settings: RerankerSettings,
        http_client: httpx.Client | None = None,
    ) -> None:
        """保存模型契约，并创建或复用底层 HTTP 客户端。"""

        self._model = settings.model
        self._max_candidates = settings.max_candidates
        self._required = settings.required
        self._base_url = settings.base_url.rstrip("/")
        self._health_endpoint = f"{self._base_url}/health"
        self._info_endpoint = f"{self._base_url}/info"
        self._headers = (
            {"Authorization": f"Bearer {settings.api_key}"}
            if settings.api_key
            else {}
        )
        self._endpoint = f"{settings.base_url.rstrip('/')}/rerank"
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            timeout=settings.timeout_seconds,
            headers=self._headers,
        )

    @property
    def model(self) -> str:
        """返回服务必须实际加载的重排模型名称。"""

        return self._model

    @property
    def max_candidates(self) -> int:
        """返回单次允许提交的最大候选文本数量。"""

        return self._max_candidates

    @property
    def required(self) -> bool:
        """说明重排服务失败时是否必须终止检索。"""

        return self._required

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
    ) -> tuple[RerankerScore, ...]:
        """按官方 TEI 契约计算查询与每条候选文本的相关性分数。"""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("reranker query cannot be empty")
        if not texts:
            raise ValueError("reranker texts cannot be empty")
        if len(texts) > self._max_candidates:
            raise ValueError(
                "reranker candidate count exceeds configured maximum: "
                f"{len(texts)} > {self._max_candidates}"
            )

        cleaned_texts = [str(text).strip() for text in texts]
        if any(not text for text in cleaned_texts):
            raise ValueError("reranker texts cannot contain blank values")

        try:
            response = self._http_client.post(
                self._endpoint,
                json={
                    "query": cleaned_query,
                    "texts": cleaned_texts,
                    "truncate": True,
                    "truncation_direction": "right",
                    "raw_scores": True,
                    "return_text": False,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RerankerServiceError(
                f"reranker request failed for model {self._model}"
            ) from error

        try:
            raw_items = response.json()
        except ValueError as error:
            raise RerankerServiceError(
                "reranker returned invalid JSON"
            ) from error

        if not isinstance(raw_items, list):
            raise RerankerServiceError(
                "reranker response must be a JSON array"
            )

        scores: list[RerankerScore] = []
        seen_indices: set[int] = set()

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise RerankerServiceError(
                    "each reranker response item must be an object"
                )

            index = raw_item.get("index")
            score = raw_item.get("score")

            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankerServiceError(
                    "reranker item index must be an integer"
                )
            if index < 0 or index >= len(cleaned_texts):
                raise RerankerServiceError(
                    f"reranker item index is out of range: {index}"
                )
            if index in seen_indices:
                raise RerankerServiceError(
                    f"reranker returned duplicate index: {index}"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RerankerServiceError(
                    f"reranker score is invalid at index {index}"
                )

            seen_indices.add(index)
            scores.append(
                RerankerScore(index=index, score=float(score))
            )

        if len(scores) != len(cleaned_texts):
            raise RerankerServiceError(
                "reranker result count does not match candidate count"
            )

        return tuple(scores)

    def is_ready(self) -> bool:
        """仅当服务健康、模型匹配且容量足够时返回就绪。"""

        try:
            health_response = self._http_client.get(
                self._health_endpoint,
                headers=self._headers,
            )
            health_response.raise_for_status()
            info_response = self._http_client.get(
                self._info_endpoint,
                headers=self._headers,
            )
            info_response.raise_for_status()
            info = info_response.json()
        except (httpx.HTTPError, ValueError):
            return False

        capacity_fields = (
            "max_concurrent_requests",
            "max_batch_requests",
            "max_client_batch_size",
        )
        capacity_ready = all(
            isinstance(info.get(field), int)
            and not isinstance(info.get(field), bool)
            and info[field] >= self._max_candidates
            for field in capacity_fields
        )

        return bool(
            isinstance(info, dict)
            and info.get("model_id") == self._model
            and isinstance(info.get("model_type"), dict)
            and "reranker" in info["model_type"]
            and capacity_ready
        )

    def close(self) -> None:
        """释放由当前包装器自行创建的 HTTP 资源。"""

        if self._owns_http_client:
            self._http_client.close()
