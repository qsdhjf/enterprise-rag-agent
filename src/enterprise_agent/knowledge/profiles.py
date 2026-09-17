"""Private, server-side provider configuration. Saving never calls a provider.

Only the loopback setup API can mutate this process-wide configuration. Public
responses contain readiness flags, never URLs, keys or provider error bodies.
"""

import hashlib
import ipaddress
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from enterprise_agent.infrastructure.config import LLMSettings, EmbeddingSettings, RerankerSettings
from enterprise_agent.knowledge.store import KnowledgeError


class Provider(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: str = Field(max_length=500)
    model: str = Field(min_length=1, max_length=150)
    api_key: SecretStr = SecretStr("")
    dimensions: int = Field(default=1024, ge=1, le=4096)

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value):
        parsed = urlsplit(value.strip())
        if any(ord(c) <= 32 for c in value.strip()):
            raise ValueError("地址不能包含空格或控制字符")
        _ = parsed.port  # Validate malformed / out-of-range ports before saving.
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("使用完整 HTTP(S) 地址，不允许 URL 中包含密钥、查询串或用户名密码")
        local = parsed.hostname == "localhost"
        try:
            local = local or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass
        if parsed.scheme == "http" and not local:
            raise ValueError("远程模型必须使用 HTTPS；HTTP 仅用于本机模型")
        return value.strip().rstrip("/")

    @field_validator("model")
    @classmethod
    def clean_model(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("模型名称不能为空")
        return value.strip()

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps([self.base_url, self.model, self.dimensions]).encode()).hexdigest()

    def llm(self):
        return LLMSettings(self.base_url, self.model, self.api_key.get_secret_value() or "local-model")

    def embedding(self):
        return EmbeddingSettings(self.base_url, self.model, self.dimensions, self.api_key.get_secret_value() or "local-model")

    def reranker(self):
        return RerankerSettings(base_url=self.base_url, model=self.model, timeout_seconds=20,
                               max_candidates=20, required=True, api_key=self.api_key.get_secret_value())


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: int = 0
    llm: Provider | None = None
    embedding: Provider | None = None
    reranker: Provider | None = None


class ProfileStore:
    def __init__(self, directory: Path, environment_llm=None):
        self.path = directory / "provider-settings.private.json"
        self.lock = RLock()
        if self.path.exists():
            try:
                self._config = ProfileConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
            except ValidationError:
                raise RuntimeError("私有模型配置无效；请在本机检查，原内容未输出以保护密钥") from None
        else:
            self._config = ProfileConfig(llm=Provider(base_url=environment_llm.base_url,
                model=environment_llm.model, api_key=environment_llm.api_key) if environment_llm else None)

    def snapshot(self):
        with self.lock:
            return self._config

    def summary(self, *, detailed=False):
        config = self.snapshot()
        result = {"revision": config.revision}
        for name in ("llm", "embedding", "reranker"):
            value = getattr(config, name)
            result[name] = {"configured": value is not None}
            if detailed and value:
                result[name].update(base_url=value.base_url, model=value.model, dimensions=value.dimensions,
                                    key_configured=bool(value.api_key.get_secret_value()))
        return result

    def update(self, kind, provider: Provider | None, expected_revision):
        if kind not in {"llm", "embedding", "reranker"}:
            raise KnowledgeError(422, "未知的模型用途")
        with self.lock:
            if expected_revision != self._config.revision:
                raise KnowledgeError(409, "配置已被修改，请刷新向导后重试")
            previous = getattr(self._config, kind)
            if provider and previous and not provider.api_key.get_secret_value():
                # Reusing a key across a changed endpoint would leak a credential.
                if provider.base_url != previous.base_url and previous.api_key.get_secret_value():
                    raise KnowledgeError(422, "更换服务地址必须重新填写密钥，不能自动转发旧密钥")
                provider = provider.model_copy(update={"api_key": previous.api_key})
            config = self._config.model_copy(update={kind: provider, "revision": self._config.revision + 1})
            raw = {"revision": config.revision}
            for name in ("llm", "embedding", "reranker"):
                value = getattr(config, name)
                raw[name] = None if value is None else {**value.model_dump(mode="json"), "api_key": value.api_key.get_secret_value()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".provider-", delete=False) as file:
                temporary = Path(file.name)
                json.dump(raw, file, ensure_ascii=False)
            try:
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            self._config = config
        return self.summary(detailed=True)
