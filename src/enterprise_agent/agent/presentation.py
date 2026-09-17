"""Explicit, bounded evidence contract; prompts and private reasoning stay private."""

from pydantic import BaseModel, ConfigDict, Field


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    source: str
    document_id: str
    chunk_id: str
    title: str
    excerpt: str = Field(max_length=4000)
    url: str | None = None


class FactCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fact: str
    status: str
    conclusion: str | None
    evidence_ids: list[str]


class Activity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    status: str
    detail: str
