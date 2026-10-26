"""Research engines — the built-in loop or an external deep-research engine.

``research_engine`` in the config selects who performs the research
stage (planning → search → reading → writing):

- ``builtin``             the tool's own Researcher (default, unchanged)
- ``gpt-researcher``      GPT Researcher in its own Python environment
- ``open-deep-research``  LangChain Open Deep Research in its own environment
- ``mock-*``              deterministic mocks (developer toggle / tests)

Whatever the engine, its output is ingested into the same session /
evidence structures (``engines.ingest``) and the rest of the pipeline
(evidence review, verification, figures, rendering, resume) is the one
the built-in engine uses. The engine contract (``engines.runner_core``)
is the orchestrator's Runner API v1, kept identical on purpose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import __version__
from .runner_core.engine import Engine
from .venv import ENGINE_SPECS, engine_health

BUILTIN_ID = "builtin"

# External engines known to the UI. ``implemented`` flips per phase: an
# engine that is listed but not yet wired shows as 未実装 and cannot be
# selected, so the catalog (and the UI) is final from Phase 1 on.
EXTERNAL_ENGINES: Dict[str, Dict[str, Any]] = {
    "gpt-researcher": {
        "name": "GPT Researcher",
        "version": ENGINE_SPECS["gpt-researcher"]["version"],
        "implemented": False,
        "description": "GPT Researcher（別の Python 環境で実行）。独自に計画・検索・執筆した"
                       "本文と出典を取り込み、このツールの根拠管理と検証にかけます。",
        "needs": "LLM: 設定中のプロバイダ（OpenAI 互換 / ローカル、埋め込みモデル必須） / "
                 "検索: DuckDuckGo または SearXNG",
        "flow": ["GPT Researcher が計画・検索・執筆", "本文と出典を取り込み",
                 "出典ページを取得して根拠化", "検証（任意）", "出力"],
    },
    "open-deep-research": {
        "name": "Open Deep Research",
        "version": ENGINE_SPECS["open-deep-research"]["version"],
        "implemented": False,
        "description": "LangChain Open Deep Research（別の Python 環境で実行）。監督者＋"
                       "調査員の多段エージェントが書いた本文と出典を取り込みます。",
        "needs": "LLM: 設定中のプロバイダ（OpenAI 互換 / Anthropic / ローカル） / "
                 "検索: DuckDuckGo または SearXNG（MCP 経由）",
        "flow": ["Open Deep Research が計画・検索・執筆", "本文と出典を取り込み",
                 "出典ページを取得して根拠化", "検証（任意）", "出力"],
    },
}

BUILTIN_ENTRY: Dict[str, Any] = {
    "id": BUILTIN_ID, "kind": "builtin", "name": "内蔵（Deep Research Tool）",
    "version": __version__, "implemented": True, "health": "available",
    "reason": None,
    "description": "このツール自身の調査ループ（計画 → 章ごとの検索・取得・抽出 → 執筆）。"
                   "手元の資料・計画の確認・全クロールモードが使えます。",
    "needs": "LLM・検索とも ⚙設定どおり（追加の環境は不要）",
    "flow": ["計画（題名・目次・検索クエリ）", "章ごとに検索・取得・抽出", "執筆",
             "検証（任意）", "出力"],
}


class EngineUnavailable(ValueError):
    """The selected engine cannot run here (unknown, not implemented, or
    its environment is missing). The message is for the user."""


def is_mock(engine_id: str) -> bool:
    return str(engine_id or "").startswith("mock-")


def is_external(engine_id: str) -> bool:
    return bool(engine_id) and engine_id != BUILTIN_ID


def known_engine_ids(include_mocks: bool = True) -> List[str]:
    ids = [BUILTIN_ID, *EXTERNAL_ENGINES]
    if include_mocks:
        from .mock.mock_engines import ALL_ENGINES
        ids.extend(ALL_ENGINES)
    return ids


def engine_display_name(engine_id: str) -> str:
    if engine_id == BUILTIN_ID:
        return BUILTIN_ENTRY["name"]
    if engine_id in EXTERNAL_ENGINES:
        return EXTERNAL_ENGINES[engine_id]["name"]
    if is_mock(engine_id):
        from .mock.mock_engines import ALL_ENGINES
        eng = ALL_ENGINES.get(engine_id)
        if eng is not None:
            return eng.capabilities().name
    return str(engine_id)


def get_engine(engine_id: str, config=None,
               engine_python: Optional[Dict[str, str]] = None) -> Engine:
    """The Engine instance for an id, or EngineUnavailable with a reason
    the user can act on. The built-in engine is not an Engine (it is the
    Researcher itself) and raises too. ``engine_python`` overrides the
    config's interpreter map (the Web UI checks a request before it is
    queued)."""
    if engine_id == BUILTIN_ID:
        raise EngineUnavailable("builtin は Engine ではありません（Researcher を使います）")
    if is_mock(engine_id):
        from .mock.mock_engines import ALL_ENGINES
        eng = ALL_ENGINES.get(engine_id)
        if eng is None:
            raise EngineUnavailable(f"不明なモックエンジンです: {engine_id}")
        return eng
    spec = EXTERNAL_ENGINES.get(engine_id)
    if spec is None:
        raise EngineUnavailable(f"不明な調査エンジンです: {engine_id}")
    if not spec.get("implemented"):
        raise EngineUnavailable(f"{spec['name']} はこのバージョンではまだ使えません"
                                "（次の更新で対応します）")
    entry = _external_entry(engine_id, config, engine_python)
    if entry["health"] != "available":
        raise EngineUnavailable(f"{spec['name']} は使えません: {entry.get('reason')}")
    raise EngineUnavailable(f"{spec['name']} の実装が登録されていません")   # Phase 2/3


def _external_entry(engine_id: str, config=None,
                    engine_python: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    spec = EXTERNAL_ENGINES[engine_id]
    entry: Dict[str, Any] = {
        "id": engine_id, "kind": "external", "name": spec["name"],
        "version": spec["version"], "implemented": bool(spec.get("implemented")),
        "description": spec["description"], "needs": spec["needs"],
        "flow": list(spec["flow"]), "min_python": ".".join(
            str(x) for x in ENGINE_SPECS[engine_id]["min_python"]),
    }
    if not spec.get("implemented"):
        entry.update(health="planned",
                     reason="このバージョンでは未実装です（次の更新で対応）",
                     setup_command=None, python=None)
        return entry
    configured: Dict[str, str] = {}
    if config is not None:
        configured = dict(getattr(config.research, "engine_python", None) or {})
    if engine_python:
        configured.update({k: v for k, v in engine_python.items() if v})
    entry.update(engine_health(engine_id, configured))
    return entry


def engine_catalog(config=None, include_mocks: bool = False,
                   engine_python: Optional[Dict[str, str]] = None
                   ) -> List[Dict[str, Any]]:
    """What the UI shows as 調査エンジン cards: id, name, kind, version,
    health (available / needs_setup / unsupported / planned), reason,
    description, needs (通信先), flow, setup_command, python."""
    out: List[Dict[str, Any]] = [dict(BUILTIN_ENTRY)]
    for engine_id in EXTERNAL_ENGINES:
        out.append(_external_entry(engine_id, config, engine_python))
    if include_mocks:
        from .mock.mock_engines import ALL_ENGINES
        for engine_id, eng in ALL_ENGINES.items():
            caps = eng.capabilities()
            out.append({
                "id": engine_id, "kind": "mock", "name": caps.name,
                "version": caps.version, "implemented": True,
                "health": "available", "reason": None,
                "description": "開発用の決定論的モック（通信なし）。"
                               + {"mock-fail": "途中で失敗します。",
                                  "mock-timeout": "完了しません（制限時間の確認用）。",
                                  "mock-cancellable": "キャンセルされるまで続きます。",
                                  "mock-partial": "指標の欠損と引用のない主張を含みます。",
                                  "mock-slow": "時間をかけて成功します。"}.get(engine_id, ""),
                "needs": "なし（ネットワーク・LLM を使いません）",
                "flow": ["モックが計画・検索・執筆", "本文と出典を取り込み",
                         "出典ページを取得して根拠化", "検証（任意）", "出力"],
                "options_schema": caps.options_schema,
            })
    return out


def check_engine_selectable(engine_id: str, config=None,
                            engine_python: Optional[Dict[str, str]] = None
                            ) -> Optional[str]:
    """None when the engine can be started here, else the user-facing
    reason (used by the Web UI before a job is queued)."""
    if engine_id == BUILTIN_ID:
        return None
    try:
        get_engine(engine_id, config, engine_python)
    except EngineUnavailable as e:
        return str(e)
    return None


__all__ = [
    "BUILTIN_ID", "EXTERNAL_ENGINES", "EngineUnavailable", "engine_catalog",
    "engine_display_name", "get_engine", "is_external", "is_mock",
    "known_engine_ids", "check_engine_selectable",
]
