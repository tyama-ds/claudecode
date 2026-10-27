"""Runner API v1 の共通データモデル。

deep-research-orchestrator の runners/common/runner_core/models.py と同一の契約
（Control Plane と全 Runner が共有する）。このツールでは SearchRunConfig.provider に
内蔵の検索クライアント（duckduckgo / selenium）を追加している以外は変更しない。
共通化できない engine 固有オプションは options（自由形式）に保持し、無理に共通抽象化しない。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RunState = Literal[
    "queued", "starting", "researching", "finalizing", "succeeded", "failed",
    "timed_out", "cancelled",
]

TERMINAL_STATES: set = {"succeeded", "failed", "timed_out", "cancelled"}


class LlmRunConfig(BaseModel):
    """run 単位で Runner へ渡す LLM 設定。profile ID と run-scoped secret のみを渡し、
    Runner 側はこれを job/event へ永続化しない。"""

    profile_id: str
    api: Literal["openai-compatible", "anthropic"] = "openai-compatible"
    endpoint: str
    model: str
    api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    timeout_seconds: int = 120


class SearchRunConfig(BaseModel):
    # orchestrator: searxng | disabled. This tool adds its own clients.
    provider: Literal["duckduckgo", "selenium", "searxng", "disabled"] = "duckduckgo"
    endpoint: Optional[str] = None  # SearXNG base URL
    timeout_seconds: int = 20
    max_results: int = 10


class RunInput(BaseModel):
    topic: str
    objective: Optional[str] = None
    instructions: Optional[str] = None
    language: str = "ja"
    input_urls: List[str] = Field(default_factory=list)
    documents: List[Dict[str, Any]] = Field(default_factory=list)  # {name, text}


class RunRequest(BaseModel):
    client_run_id: str  # 採番済みの run ID（冪等性キー）
    engine_id: str
    input: RunInput
    # engine 固有オプション（breadth/depth/seed/speed_factor 等）
    options: Dict[str, Any] = Field(default_factory=dict)
    llm: Optional[LlmRunConfig] = None
    search: Optional[SearchRunConfig] = None
    proxy_env: Dict[str, str] = Field(default_factory=dict)
    max_time_seconds: Optional[int] = None
    max_searches: Optional[int] = None


class RunEvent(BaseModel):
    seq: int
    type: str
    # type 例: status, stage, log, search, source_found, token_usage, cost,
    #          warning, error, result_ready
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: str


class SourceRecord(BaseModel):
    url: str
    title: Optional[str] = None
    fetched_at: Optional[str] = None
    excerpt: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class EvidenceRecord(BaseModel):
    source_url: str
    excerpt: Optional[str] = None
    locator: Optional[str] = None
    stance: Literal["supports", "contradicts", "context"] = "supports"
    verification: Literal["verified", "unverified", "failed"] = "unverified"


class ClaimRecord(BaseModel):
    text: str
    # key/value は engine が構造化された主張を返せる場合のみ（mock 等）。
    # 実エンジンは通常 text のみで、比較は正規化側のクラスタリングに委ねる。
    key: Optional[str] = None
    value: Optional[str] = None
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """取得できた値のみ。不明値は null のまま返し、推測しない。"""

    searches: Optional[int] = None
    sources: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    llm_cost_usd: Optional[float] = None
    llm_cost_is_estimate: Optional[bool] = None
    search_api_cost_usd: Optional[float] = None  # self-hosted search では 0
    infra_cost: Optional[str] = None  # self-hosted では "not_measured"
    duration_seconds: Optional[float] = None


class RunResult(BaseModel):
    output_kind: Literal["report", "answer", "evidence"] = "report"
    summary: Optional[str] = None
    report_markdown: Optional[str] = None
    claims: List[ClaimRecord] = Field(default_factory=list)
    sources: List[SourceRecord] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    warnings: List[str] = Field(default_factory=list)
    # engine 生出力（正規化前）— raw artifact として保存
    raw: Dict[str, Any] = Field(default_factory=dict)


class RunStatusResponse(BaseModel):
    run_id: str
    engine_id: str
    state: RunState
    stage: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_seq: int = 0


class EngineCapabilities(BaseModel):
    engine_id: str
    name: str
    version: Optional[str] = None
    output_kind: Literal["report", "answer", "evidence"] = "report"
    streaming: bool = True
    cancel: bool = True
    citations: bool = False
    token_usage: bool = False
    cost: bool = False
    local_files: bool = False
    # engine 固有オプションの JSON Schema（UI が動的フォームに使う）
    options_schema: Dict[str, Any] = Field(default_factory=dict)
    # available | unhealthy | disabled | unsupported
    health: str = "available"
    health_reason: Optional[str] = None
    required_config: List[str] = Field(default_factory=list)
