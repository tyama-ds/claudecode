"""
Run presets of the Web UI and the recommended settings of its simple mode,
defined in ONE place (server side).

- ``DELIVERABLE_PRESETS``  仕上がり: 要約 / 標準レポート / 詳細調査
- ``RESEARCH_PRESETS``     時間のかけ方: 短時間 / 標準 / 詳細
- ``RECOMMENDED``          シンプルモードの推奨設定（クラウドAPI / ローカルLLM）

Values use the request parameter names of ``POST /api/research`` (the
names ``webui.server.build_config_kwargs`` maps to ``create_config``). The
page reads all of this from ``GET /api/presets`` and keeps no copy of its
own, so the form, the simple mode and API clients cannot drift apart.

A request with ``use_recommended`` runs with the recommended settings:
``apply_recommended`` keeps only the simple-mode inputs (theme, sources,
output format, language) and the connection settings, and fills every
other setting from ``recommended_params``. A setting sent next to
``use_recommended`` (``verification_profile: "strict"``…) is dropped.
"""

import os
from typing import Any, Dict, Iterable, List, Optional

# 仕上がり: what each card changes (shown under the cards and in history)
DELIVERABLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "summary": {
        "label": "要約",
        "desc": "文章量=固定 約4,000字／図表の自動生成=OFF／不足時の再調査=0回",
        "values": {"length_mode": "fixed", "preferred_body_chars": 4000,
                   "auto_figures": False, "gap_fill_rounds": 0},
    },
    "standard": {
        "label": "標準レポート",
        "desc": "文章量=適応（情報量から算出）／図表の自動生成=ON／不足時の再調査=1回",
        # None = the field is cleared (no target length)
        "values": {"length_mode": "adaptive", "preferred_body_chars": None,
                   "auto_figures": True, "gap_fill_rounds": 1},
    },
    "detailed": {
        "label": "詳細調査",
        "desc": "文章量=適応（目標 約20,000字・水増しなし）／図表の自動生成=ON／不足時の再調査=2回",
        "values": {"length_mode": "adaptive", "preferred_body_chars": 20000,
                   "auto_figures": True, "gap_fill_rounds": 2},
    },
}

# 時間のかけ方 (調査全体のプリセット). NOTE: not the verification preset
# (Fast / Balanced / Strict) — it SETS that preset as one of its effects.
RESEARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "label": "短時間",
        "desc": "章ごとの調査回数=1〜2／ページ数/クエリ=2／クロール=高速（並列取得）／"
                "検証モード=Fast（追加調査・自動修正なし）／DeepThink=OFF／推敲パス=OFF",
        "values": {"iterations": 1, "max_iterations": 2, "max_pages_per_query": 2,
                   "ai_crawl_max_pages": 8, "verification_profile": "fast",
                   "deep_think": False, "v2_enable_polish": False,
                   "crawl_mode": "fast_batch"},
    },
    "standard": {
        "label": "標準",
        "desc": "章ごとの調査回数=2〜4／ページ数/クエリ=3／クロール=標準／"
                "検証モード=Balanced（追加調査1回・自動修正1回）／DeepThink=OFF／推敲パス=ON",
        "values": {"iterations": 2, "max_iterations": 4, "max_pages_per_query": 3,
                   "ai_crawl_max_pages": 15, "verification_profile": "balanced",
                   "deep_think": False, "v2_enable_polish": True,
                   "crawl_mode": "standard"},
    },
    "thorough": {
        "label": "詳細",
        "desc": "章ごとの調査回数=3〜6／ページ数/クエリ=5／クロール=標準／"
                "検証モード=Strict（追加調査2回・自動修正2回・長時間化の可能性）／DeepThink=ON／推敲パス=ON",
        "values": {"iterations": 3, "max_iterations": 6, "max_pages_per_query": 5,
                   "ai_crawl_max_pages": 25, "verification_profile": "strict",
                   "deep_think": True, "v2_enable_polish": True,
                   "crawl_mode": "standard"},
    },
}

# the crawl mode of a research preset is not applied with only local
# documents (no Web access, the form keeps the crawl cards as they were)
_CRAWL_FREE_SOURCES = ("local",)

# シンプルモードの推奨設定. local_think / local_num_ctx are deliberately NOT
# here: the right value depends on the model.
RECOMMENDED: Dict[str, Dict[str, Any]] = {
    "cloud": {"deliverable": "standard", "research_preset": "standard",
              "extraction_max_chunks": None, "extraction_max_tokens": None,
              "plan_review": False},
    "local": {"deliverable": "standard", "research_preset": "quick",
              "extraction_max_chunks": 3, "extraction_max_tokens": 3000,
              "plan_review": False},
}

PROVIDER_KINDS: Dict[str, str] = {"openai": "cloud", "anthropic": "cloud", "local": "local"}

# what a simple-mode run always is: the built-in research loop with the
# verification (the run-mode / engine cards are not shown in that mode)
SIMPLE_FIXED: Dict[str, Any] = {"run_mode": "research_verify",
                                "enable_verification": True,
                                "research_engine": "builtin",
                                "plan_review_timeout": 0}

# what the simple mode itself sends: the theme, the sources and the output
SIMPLE_INPUT_KEYS = frozenset({
    "query", "requirements",
    "source_mode", "source_urls", "url_crawl_depth", "preferred_domains",
    "local_documents",
    "output_format", "language",
})

# connection settings (⚙ 設定 → 生成AI・プロキシ): where and how to reach the
# LLM, never how to research. local_think / local_num_ctx belong to the
# model, so an API client may still send them.
CONNECTION_KEYS = frozenset({
    "provider", "model",
    "openai_api_key", "anthropic_api_key", "local_api_key",
    "openai_base_url", "anthropic_base_url",
    "local_base_url", "local_backend", "local_model",
    "local_timeout", "local_concurrency", "local_think", "local_num_ctx",
    "local_llm_role", "local_llm_fallback",
    "http_proxy", "https_proxy", "verify_ssl",
})

SIMPLE_KEYS = SIMPLE_INPUT_KEYS | CONNECTION_KEYS

DEFAULT_PROVIDER = "openai"          # create_config's default


def provider_kind(provider: Optional[str]) -> str:
    """'cloud' or 'local' for a provider name (unknown names: 'cloud')."""
    name = str(provider or DEFAULT_PROVIDER).strip().lower()
    return PROVIDER_KINDS.get(name, "cloud")


def wants_recommended(params: Dict[str, Any]) -> bool:
    """True when a request asks for the recommended settings."""
    value = (params or {}).get("use_recommended")
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return value is True or value == 1


def preset_values(deliverable: Optional[str] = None,
                  research_preset: Optional[str] = None,
                  source_mode: Optional[str] = None) -> Dict[str, Any]:
    """The concrete request parameters of a 仕上がり + 時間のかけ方 pair
    (a value of None = the field is left to the config default)."""
    out: Dict[str, Any] = {}
    if deliverable in DELIVERABLE_PRESETS:
        out.update(DELIVERABLE_PRESETS[deliverable]["values"])
    if research_preset in RESEARCH_PRESETS:
        values = dict(RESEARCH_PRESETS[research_preset]["values"])
        if (source_mode or "web") in _CRAWL_FREE_SOURCES:
            values.pop("crawl_mode", None)
        out.update(values)
    return out


def recommended_params(provider: Optional[str] = None,
                       source_mode: Optional[str] = None) -> Dict[str, Any]:
    """Every setting a simple-mode run gets for this provider, as request
    parameters (None-valued settings are left out: config default)."""
    rec = RECOMMENDED[provider_kind(provider)]
    out: Dict[str, Any] = {"deliverable": rec["deliverable"],
                           "research_preset": rec["research_preset"]}
    out.update(preset_values(rec["deliverable"], rec["research_preset"], source_mode))
    for key in ("extraction_max_chunks", "extraction_max_tokens", "plan_review"):
        if rec.get(key) is not None:
            out[key] = rec[key]
    out.update(SIMPLE_FIXED)
    return {k: v for k, v in out.items() if v is not None}


def apply_recommended(params: Dict[str, Any],
                      keep: Iterable[str] = ()) -> Dict[str, Any]:
    """The request a ``use_recommended`` run really uses.

    Only the simple-mode inputs and the connection settings survive (plus
    ``keep``: the control keys of the endpoint, e.g. a feedback's own plan
    review choice); every other setting comes from the recommended
    settings, whatever else the request carried.
    """
    params = params or {}
    out = {k: v for k, v in params.items() if k in SIMPLE_KEYS}
    provider = str(out.get("provider") or DEFAULT_PROVIDER).strip().lower()
    out["provider"] = provider
    out.update(recommended_params(provider, out.get("source_mode")))
    for key in keep:
        if key in params:
            out[key] = params[key]
    out["use_recommended"] = True
    out["ui_mode"] = "simple"
    return out


# ⚙設定 field each connection value is typed in (the page highlights it)
_MISSING_MESSAGES = {
    "openai_api_key": "OpenAI の APIキーが設定されていません（⚙ 設定 → OpenAI、またはサーバーの環境変数 OPENAI_API_KEY）",
    "anthropic_api_key": "Anthropic の APIキーが設定されていません（⚙ 設定 → Anthropic、またはサーバーの環境変数 ANTHROPIC_API_KEY）",
    "local_base_url": "ローカルLLMへの委譲（ロールルーティング）にはサーバーURLが必要です（⚙ 設定 → ローカルLLM、またはサーバーの環境変数 LOCAL_LLM_BASE_URL）",
    "provider": "使用するAI（プロバイダー）は OpenAI・Anthropic・ローカルLLM のいずれかを選んでください",
}


def missing_connection(params: Dict[str, Any]) -> List[Dict[str, str]]:
    """Connection settings a run with these parameters cannot start without
    (the same checks as Config.validate, environment variables included):
    [{"param": ..., "message": ...}]; [] when nothing is missing."""
    params = params or {}

    def given(key: str, env: str) -> bool:
        value = params.get(key)
        return bool((str(value).strip() if value is not None else "") or os.getenv(env))

    missing = []
    provider = str(params.get("provider") or DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDER_KINDS:
        missing.append("provider")
    elif provider == "openai" and not given("openai_api_key", "OPENAI_API_KEY"):
        missing.append("openai_api_key")
    elif provider == "anthropic" and not given("anthropic_api_key", "ANTHROPIC_API_KEY"):
        missing.append("anthropic_api_key")
    role = str(params.get("local_llm_role") or "off").strip().lower()
    if role != "off" and not given("local_base_url", "LOCAL_LLM_BASE_URL"):
        missing.append("local_base_url")
    return [{"param": key, "message": _MISSING_MESSAGES[key]} for key in missing]


def presets_payload() -> Dict[str, Any]:
    """``GET /api/presets``: the preset tables, the recommended settings
    per provider kind, and those settings expanded per provider."""
    return {
        "deliverable_presets": DELIVERABLE_PRESETS,
        "research_presets": RESEARCH_PRESETS,
        "recommended": RECOMMENDED,
        "provider_kinds": PROVIDER_KINDS,
        "recommended_params": {p: recommended_params(p) for p in PROVIDER_KINDS},
        "simple_keys": sorted(SIMPLE_KEYS),
    }
