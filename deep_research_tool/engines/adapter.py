"""EngineResearcher — an external engine behind the Researcher's surface.

``DeepResearchTool.run`` drives ``Researcher`` through a small set of
attributes and methods (conduct_research / resume_research / get_session /
get_evidence_locker / _checkpoint / current_activity / search / cancel_check
/ run_config / resume_plan). This class offers the same surface, so the
stages after the research loop (evidence review, Fermi, figures, report
writing, verification, rendering, resume) run unchanged on an engine's
output.

The engine runs IN THIS THREAD on its own asyncio loop
(``asyncio.run``): no worker thread, no shared loop. The tool's
``cancel_check`` is polled while the engine runs and turned into the
engine's cooperative cancel; a hard time limit (``engine_max_time_seconds``)
cancels the coroutine. Progress events become the tool's usual
``(message, percent)`` reports: 0–10 preparation, 10–85 the engine's
stages, 85–99 ingestion.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..evidence.locker import EvidenceLocker
from ..research.researcher import ResearchSession, ResearchState
from ..utils.cancellation import RunCancelled
from .ingest import (build_session_from_result, save_engine_artifacts,
                     seed_evidence_from_sources)
from .runner_core.engine import CancelledByUser, Engine, RunContext
from .runner_core.models import (LlmRunConfig, RunInput, RunRequest, RunResult,
                                 SearchRunConfig)

STAGE_LABELS = {
    "planning": "計画", "searching": "検索", "reading": "読み込み",
    "writing": "執筆", "starting_worker": "ワーカー起動", "finalizing": "仕上げ",
    "research": "調査", "report": "レポート作成", "draft": "下書き",
    # gpt-researcher worker stages (ours + the library's research steps)
    "conduct_research": "調査（検索と読み込み）", "write_report": "レポート執筆",
    "start": "開始", "conducting_research": "検索と読み込み",
    "research_completed": "調査完了", "writing_report": "レポート執筆",
    "report_completed": "レポート完成",
    "action:plan": "計画", "action:search": "検索", "action:scrape": "ページ取得",
    "action:write": "執筆",
    # open-deep-research graph nodes (worker.py maps the node names)
    "starting_mcp_server": "検索サーバー起動", "clarify": "質問の確認",
    "brief": "調査方針の作成", "supervisor": "監督者が調査を割り当て",
    "supervisor_tools": "監督者のツール実行", "researcher": "調査員が検索・読み込み",
    "researcher_tools": "調査員のツール実行", "compress": "調査結果の圧縮",
    "final_report": "最終レポートの執筆",
}
# where a stage sits in the engine's 10–85 progress band when the engine
# reports no step / total (fractions of the band; the bar never moves back)
STAGE_FRACTION = {
    "starting_worker": 0.02, "planning": 0.05, "action:plan": 0.05, "start": 0.05,
    "conduct_research": 0.10, "conducting_research": 0.12,
    "searching": 0.25, "action:search": 0.25,
    "reading": 0.45, "action:scrape": 0.45, "research_completed": 0.65,
    "write_report": 0.72, "writing_report": 0.75,
    "writing": 0.80, "action:write": 0.80, "finalizing": 0.95, "report_completed": 0.98,
    "starting_mcp_server": 0.01, "clarify": 0.03, "brief": 0.08, "supervisor": 0.15,
    "supervisor_tools": 0.18, "researcher": 0.30, "researcher_tools": 0.35,
    "compress": 0.60, "final_report": 0.80,
}

ENGINE_PROGRESS_START = 10.0   # the research slice starts here (see ProgressPlan)
ENGINE_PROGRESS_END = 85.0
INGEST_PROGRESS_END = 99.0
CANCEL_GRACE_SECONDS = 10.0    # engine ignores the cooperative cancel → hard cancel


class _ReportingContext(RunContext):
    """RunContext that forwards every event to the adapter as it happens."""

    def __init__(self, run_id: str, request: RunRequest,
                 on_event: Callable[[Any], None]):
        super().__init__(run_id, request)
        self._on_event = on_event

    def emit(self, type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().emit(type, payload)
        try:
            self._on_event(self._events[-1])
        except RunCancelled:
            raise
        except Exception:
            pass


class EngineResearcher:
    """Researcher-compatible driver for one external engine run."""

    def __init__(self, engine: Engine, *, output_dir, language: str = "ja",
                 llm_client=None, search_client=None,
                 progress_callback: Optional[Callable[[str, Any], None]] = None,
                 cancel_check: Optional[Callable[[], None]] = None,
                 run_config: Optional[Dict[str, Any]] = None,
                 llm: Optional[LlmRunConfig] = None,
                 search: Optional[SearchRunConfig] = None,
                 proxy_env: Optional[Dict[str, str]] = None,
                 options: Optional[Dict[str, Any]] = None,
                 max_time_seconds: Optional[int] = 1800,
                 fetch_sources: bool = True,
                 parallel_max_workers: int = 4,
                 source_urls: Optional[List[str]] = None,
                 complement_urls: bool = True):
        self.engine = engine
        self.capabilities = engine.capabilities()
        self.engine_id = self.capabilities.engine_id
        self.engine_name = self.capabilities.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.llm = llm_client
        self.search = search_client
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.run_config: Dict[str, Any] = dict(run_config or {})
        self.llm_run_config = llm
        self.search_run_config = search
        self.proxy_env = dict(proxy_env or {})
        self.options = dict(options or {})
        self.max_time_seconds = int(max_time_seconds) if max_time_seconds else None
        self.fetch_sources = bool(fetch_sources)
        self.parallel_max_workers = max(1, int(parallel_max_workers or 1))
        # 指定 URL: handed to the engine as input_urls; complement_urls=False
        # means "these pages only" (no web search), True "these first"
        self.source_urls = [str(u) for u in (source_urls or []) if u]
        self.complement_urls = bool(complement_urls)
        if self.source_urls:
            self.options["complement_source_urls"] = self.complement_urls

        # Researcher surface
        self.session: Optional[ResearchSession] = None
        self.evidence_locker: Optional[EvidenceLocker] = None
        self.resume_plan: Optional[Dict[str, Any]] = None
        self.current_activity: Optional[Dict[str, Any]] = None
        self.timings = None
        self.content_extractor = None
        self.fast_crawler = None
        self.page_reuse_hits = 0
        self._checkpoint_lock = threading.Lock()

        # engine run state
        self.result: Optional[RunResult] = None
        self.events: List[Any] = []
        self.engine_warnings: List[str] = []
        self.engine_artifacts: Dict[str, str] = {}
        self.ingest_stats: Dict[str, int] = {}
        self.engine_elapsed: Optional[float] = None
        self._ctx: Optional[_ReportingContext] = None
        self._last_pct = 0.0

    # ------------------------------------------------------------------
    # Researcher surface
    # ------------------------------------------------------------------
    def get_session(self) -> Optional[ResearchSession]:
        return self.session

    def get_evidence_locker(self) -> Optional[EvidenceLocker]:
        return self.evidence_locker

    def expand_section_content(self, **kwargs) -> Dict[str, int]:
        """Length shortfalls are not researched by the engine path."""
        return {"characters_added": 0, "new_sources": 0}

    def conduct_research(self, query: str, requirements: str = "",
                         additional_context: str = "",
                         additional_documents: Optional[List[Dict[str, str]]] = None
                         ) -> ResearchSession:
        if additional_documents:
            raise ValueError(f"{self.engine_name} は手元の資料を受け付けません"
                             "（使用する資料を「Web」にするか、内蔵エンジンを選んでください）")
        return self._conduct(query, requirements, additional_context, session_id=None)

    def resume_research(self, session_path, additional_iterations: int = 2
                        ) -> ResearchSession:
        """A completed engine run is reused as is; an unfinished one is run
        again from the start (engines keep no partial state) under the
        same session id, so the stages after it resume normally."""
        session = ResearchSession.load(Path(session_path))
        self.session = session
        self.evidence_locker = self._load_locker(session.session_id)
        if session.state == ResearchState.COMPLETED and session.section_contents:
            sections = sorted(session.section_contents, key=_section_sort_key)
            self.resume_plan = {"reused": sections, "to_process": [],
                                "engine": self.engine_id}
            self._report(f"Reusing completed sections ({self.engine_name}): "
                         f"{len(sections)} 章", 99.0)
            return session
        self.resume_plan = {"reused": [], "to_process": ["*"],
                            "engine": self.engine_id, "engine_rerun": True}
        self._report(f"Planning ({self.engine_name}): 前回は完了していないため "
                     "エンジンを最初から実行します", 1.0)
        return self._conduct(session.query, session.requirements or "", "",
                             session_id=session.session_id)

    def _checkpoint(self, stage: str, reason: str = "") -> Dict[str, str]:
        """Persist session + evidence atomically (same contract as Researcher)."""
        with self._checkpoint_lock:
            saved: Dict[str, str] = {}
            if self.session is None:
                return saved
            self.session.stage = stage
            self.session.run_config = dict(self.run_config)
            self.session.checkpoints.append({
                "stage": stage, "reason": reason,
                "at": datetime.now().isoformat(),
                "completed_sections": list(self.session.completed_sections),
            })
            if self.evidence_locker is not None:
                try:
                    ev_path = self.evidence_locker.export_to_json()
                    if Path(ev_path).exists():
                        saved["evidence_json"] = str(ev_path)
                except Exception as e:
                    print(f"[Checkpoint] evidence export failed: {e}")
            session_path = self.output_dir / f"session_{self.session.session_id}.json"
            saved["session"] = str(session_path)
            self.session.saved_artifacts = dict(saved)
            try:
                self.session.save(session_path)
            except Exception as e:
                print(f"[Checkpoint] session save failed: {e}")
                saved.pop("session", None)
                self.session.saved_artifacts = dict(saved)
            return saved

    # ------------------------------------------------------------------
    # the run
    # ------------------------------------------------------------------
    def _conduct(self, query: str, requirements: str, additional_context: str,
                 session_id: Optional[str]) -> ResearchSession:
        self.session = ResearchSession(query=query, requirements=requirements or "",
                                       state=ResearchState.PLANNING)
        if session_id:
            self.session.session_id = session_id
        self.evidence_locker = EvidenceLocker(
            research_id=self.session.session_id,
            output_dir=self.output_dir / "evidence")
        self.result = None
        self.events = []
        self.engine_warnings = []
        self.current_activity = {"phase": "engine", "engine": self.engine_name,
                                 "stage": "starting", "stage_label": "準備"}
        self._report(f"Planning ({self.engine_name}): エンジンへの依頼を準備しています", 2.0)
        request = self._build_request(query, requirements, additional_context)
        self.run_config.update({
            "research_engine": self.engine_id,
            "engine_version": self.capabilities.version,
            "engine_options": dict(self.options),
        })
        self.session.state = ResearchState.RESEARCHING
        self._report(f"Researching ({self.engine_name}): 開始", ENGINE_PROGRESS_START)
        result = self._run_engine(request)
        self.result = result
        self.engine_warnings = list(result.warnings or [])
        self._ingest(result, query, requirements, request)
        return self.session

    def _build_request(self, query: str, requirements: str,
                       additional_context: str) -> RunRequest:
        return RunRequest(
            client_run_id=self.session.session_id,
            engine_id=self.engine_id,
            input=RunInput(topic=query, objective=requirements or None,
                           instructions=additional_context or None,
                           language=self.language,
                           input_urls=list(self.source_urls)),
            options=dict(self.options),
            llm=self.llm_run_config,
            search=self.search_run_config,
            proxy_env=dict(self.proxy_env),
            max_time_seconds=self.max_time_seconds,
        )

    def _run_engine(self, request: RunRequest) -> RunResult:
        ctx = _ReportingContext(request.client_run_id, request, self._on_event)
        self._ctx = ctx
        started = time.monotonic()
        try:
            result = asyncio.run(self._drive(ctx))
        except CancelledByUser:
            self._fail(ResearchState.CANCELLED, "cancelled by user")
            raise RunCancelled("cancelled by user")
        except RunCancelled:
            self._fail(ResearchState.CANCELLED, "cancelled by user")
            raise
        except asyncio.CancelledError:
            if ctx.cancel_requested:
                self._fail(ResearchState.CANCELLED, "cancelled by user")
                raise RunCancelled("cancelled by user")
            msg = f"{self.engine_name} の実行が中断されました"
            self._fail(ResearchState.ERROR, msg)
            raise RuntimeError(msg)
        except asyncio.TimeoutError:
            msg = (f"{self.engine_name} が制限時間 {self.max_time_seconds} 秒以内に"
                   "完了しませんでした（engine_max_time_seconds）")
            self._fail(ResearchState.ERROR, msg)
            raise RuntimeError(msg)
        except Exception as e:  # noqa: BLE001 - every engine error becomes a run error
            msg = f"{self.engine_name} の実行に失敗しました: {e}"
            self._fail(ResearchState.ERROR, msg)
            raise RuntimeError(msg) from e
        finally:
            self.engine_elapsed = round(time.monotonic() - started, 3)
            self.events = list(ctx.events)
        if not isinstance(result, RunResult):
            msg = f"{self.engine_name} が結果を返しませんでした"
            self._fail(ResearchState.ERROR, msg)
            raise RuntimeError(msg)
        return result

    async def _drive(self, ctx: _ReportingContext) -> RunResult:
        run_task = asyncio.ensure_future(self.engine.run(ctx))
        watcher = asyncio.ensure_future(self._watch_cancel(ctx, run_task))
        try:
            if self.max_time_seconds:
                return await asyncio.wait_for(run_task, timeout=self.max_time_seconds)
            return await run_task
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    async def _watch_cancel(self, ctx: _ReportingContext, run_task) -> None:
        """Turn the tool's cancel into the engine's cooperative cancel; an
        engine that keeps running after that is cancelled hard."""
        while not run_task.done():
            try:
                if self.cancel_check is not None:
                    self.cancel_check()
            except Exception:
                ctx.request_cancel()
                deadline = time.monotonic() + CANCEL_GRACE_SECONDS
                while not run_task.done() and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                if not run_task.done():
                    run_task.cancel()
                return
            await asyncio.sleep(0.25)

    def _fail(self, state: ResearchState, message: str) -> None:
        if self.session is None:
            return
        self.session.state = state
        self.session.error_message = message
        self.session.completed_at = datetime.now().isoformat()
        self.current_activity = None
        stage = "cancelled" if state == ResearchState.CANCELLED else "error"
        self._checkpoint(stage, message)

    # ------------------------------------------------------------------
    # events → progress
    # ------------------------------------------------------------------
    def _on_event(self, event) -> None:
        etype = event.type
        payload = event.payload or {}
        name = self.engine_name
        if etype == "stage":
            stage = str(payload.get("stage") or "")
            label = STAGE_LABELS.get(stage, stage)
            if label == stage and stage.startswith("action:"):
                label = stage.split(":", 1)[1]
            step = payload.get("step")
            total = payload.get("total")
            pct = None
            band = ENGINE_PROGRESS_END - ENGINE_PROGRESS_START
            try:
                if total and int(total) > 0:
                    frac = max(0.0, min(1.0, (float(step) - 1.0) / float(total)))
                    pct = ENGINE_PROGRESS_START + frac * band
                elif stage in STAGE_FRACTION:
                    pct = ENGINE_PROGRESS_START + STAGE_FRACTION[stage] * band
            except (TypeError, ValueError):
                pct = None
            self.current_activity = {"phase": "engine", "engine": name,
                                     "stage": stage, "stage_label": label,
                                     "index": step, "total": total}
            counts = f"（{step}/{total}）" if total and step else ""
            self._report(f"Researching ({name}): {label}{counts}", pct)
        elif etype == "search":
            q = payload.get("query") or ""
            self._report(f"Researching ({name}): 検索 {q}", None)
        elif etype == "source_found":
            t = payload.get("title") or payload.get("url") or ""
            self._report(f"Researching ({name}): 情報源 {t}", None)
        elif etype in ("log", "status"):
            if payload.get("content") == "report_chunk":
                return                      # streamed body chunks: not a log line each
            msg = " ".join(str(payload.get("message") or payload.get("status") or "").split())
            if msg:
                self._report(f"Researching ({name}): {msg[:200]}", None)
        elif etype == "warning":
            msg = str(payload.get("message") or payload)
            self.engine_warnings.append(msg)
            self._report(f"Researching ({name}): 警告 {msg}", None)
        elif etype == "error":
            msg = str(payload.get("message") or payload)
            self._report(f"Researching ({name}): エラー {msg}", None)

    def _report(self, message: str, pct: Optional[float]) -> None:
        if pct is not None:
            # never move backwards; the stages after us start at 100 of ours
            pct = max(self._last_pct, min(float(pct), INGEST_PROGRESS_END))
            self._last_pct = pct
        if self.progress_callback is not None:
            self.progress_callback(message, pct)

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _ingest(self, result: RunResult, query: str, requirements: str,
                request: RunRequest) -> None:
        name = self.engine_name
        sid = self.session.session_id
        self.current_activity = {"phase": "engine", "engine": name,
                                 "stage": "ingest", "stage_label": "結果の取り込み"}
        self._report(f"Researching ({name}): 結果を取り込んでいます", 86.0)
        try:
            session, sources, section_sources = build_session_from_result(
                result, query=query, requirements=requirements,
                language=self.language, engine_id=self.engine_id,
                engine_name=name, session_id=sid)
        except ValueError as e:
            self._fail(ResearchState.ERROR, str(e))
            raise RuntimeError(str(e)) from e
        session.started_at = self.session.started_at
        self.session = session
        self._report(f"Researching ({name}): 情報源 {len(sources)} 件のページを取得しています", 88.0)
        self.current_activity["stage_label"] = f"情報源の取得（{len(sources)} 件）"
        stats = seed_evidence_from_sources(
            session, self.evidence_locker, sources, section_sources,
            search_client=self.search, engine_id=self.engine_id,
            engine_name=name, topic=query,
            parallel_max_workers=self.parallel_max_workers,
            fetch=self.fetch_sources, cancel_check=self.cancel_check)
        self.ingest_stats = stats
        if stats.get("excerpt_only"):
            self.engine_warnings.append(
                f"{stats['excerpt_only']} 件の情報源はページを取得できず、"
                "エンジンの抜粋のみを根拠として登録しました")
        try:
            summary = {"engine_id": self.engine_id,
                       "engine_version": self.capabilities.version,
                       "topic": query, "language": self.language,
                       "options": dict(self.options),
                       "max_time_seconds": self.max_time_seconds,
                       "llm": {"api": request.llm.api, "endpoint": request.llm.endpoint,
                               "model": request.llm.model,
                               "embedding_model": request.llm.embedding_model}
                       if request.llm else None,
                       "search": request.search.model_dump() if request.search else None}
            self.engine_artifacts = save_engine_artifacts(
                self.output_dir, sid, self.engine_id, result, self.events, summary)
        except Exception as e:  # noqa: BLE001 - the audit copy must not fail the run
            self.engine_warnings.append(f"エンジン出力の保存に失敗: {e}")
            self.engine_artifacts = {}
        self.run_config.update({
            "engine_metrics": result.metrics.model_dump(mode="json"),
            "engine_elapsed_seconds": self.engine_elapsed,
            "engine_ingest": dict(stats),
            "engine_artifacts": dict(self.engine_artifacts),
        })
        session.completed_at = datetime.now().isoformat()
        session.state = ResearchState.COMPLETED
        self.resume_plan = None
        self.current_activity = None
        self._checkpoint("completed", "engine result ingested")
        self._report(f"Researching ({name}): 完了（{len(session.section_contents)} 章・"
                     f"情報源 {stats.get('sources', 0)} 件、取得 {stats.get('fetched', 0)} 件）",
                     INGEST_PROGRESS_END)

    def _load_locker(self, session_id: str) -> EvidenceLocker:
        ev_dir = self.output_dir / "evidence"
        ev_path = ev_dir / f"evidence_{session_id}.json"
        if ev_path.exists():
            try:
                return EvidenceLocker.load_from_json(ev_path)
            except Exception:
                pass
        return EvidenceLocker(research_id=session_id, output_dir=ev_dir)


def _section_sort_key(sid: str):
    try:
        return tuple(int(p) for p in str(sid).split("."))
    except ValueError:
        return (10 ** 6, str(sid))
