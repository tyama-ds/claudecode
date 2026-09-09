"""
Web UI server for Deep Research Tool.

Dependency-free (Python stdlib http.server) local web app:
- Serves the single-page UI from webui/static/index.html
- POST /api/research     starts a research job in a background thread
- GET  /api/status       returns job progress / log / result
- GET  /api/reports      lists generated report files
- GET  /api/report-file  downloads a report file (restricted to output dir)

Start with:  deep-research webui  (or python -m deep_research_tool.webui.server)
"""

import copy
import json
import os
import threading
import time
import traceback
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .. import __version__

STATIC_DIR = Path(__file__).parent / "static"

# UI parameter name -> create_config kwarg (pass-through numeric/str/bool)
_CONFIG_PARAM_MAP = {
    "provider": "provider",
    "model": "model",
    "openai_api_key": "openai_api_key",
    "anthropic_api_key": "anthropic_api_key",
    "openai_base_url": "openai_base_url",
    "anthropic_base_url": "anthropic_base_url",
    "local_base_url": "local_base_url",
    "local_api_key": "local_api_key",
    "local_backend": "local_backend",
    "search_method": "search_method",
    "browser": "browser",
    "driver_path": "driver_path",
    "waf_mitigation": "waf_mitigation",
    "per_domain_delay": "per_domain_delay",
    "http_proxy": "http_proxy",
    "https_proxy": "https_proxy",
    "verify_ssl": "verify_ssl",
    "iterations": "research_iterations",
    "parallel_max_workers": "parallel_max_workers",
    "output_format": "output_format",
    "output_dir": "output_dir",
    "language": "language",
    "source_mode": "source_mode",
    "crawl_mode": "crawl_mode",
    "ai_crawl_max_pages": "ai_crawl_max_total_pages",
    "ai_crawl_site_depth": "ai_crawl_site_depth",
    "max_pages_per_query": "max_pages_per_query",
    "gap_fill_rounds": "max_gap_fill_rounds",
    "importance_threshold": "importance_threshold",
    "report_version": "report_generator_version",
    "v2_writing_style": "v2_writing_style",
    "v2_enable_polish": "v2_enable_polish",
    "live_report_word": "live_report_word",
    "chart_library": "chart_library",
    "auto_figures": "auto_figures",
    "enable_verification": "enable_verification",
    "plan_review": "plan_review",
    "plan_review_timeout": "plan_review_timeout",
    # Advanced feature toggles (settings modal 機能 tab)
    "deep_think": "deep_think",
    "fermi_estimation": "fermi_estimation",
    "multilingual": "multilingual",
    "search_languages": "search_languages",
    "search_regions": "search_regions",
    "use_enhanced_synthesis": "use_enhanced_synthesis",
    "figure_max_workers": "figure_max_workers",
    "deep_think_max_workers": "deep_think_max_workers",
    # Adaptive length + finalization loop settings (all 12)
    "length_mode": "length_mode",
    "preferred_body_chars": "preferred_body_chars",
    "hard_min_body_chars": "hard_min_body_chars",
    "hard_max_body_chars": "hard_max_body_chars",
    "length_tolerance": "length_tolerance",
    "max_final_research_rounds": "max_final_research_rounds",
    "max_final_revision_rounds": "max_final_revision_rounds",
    "max_no_improvement_rounds": "max_no_improvement_rounds",
    "min_score_improvement": "min_score_improvement",
    "min_new_independent_sources": "min_new_independent_sources",
    "min_claim_support_score": "min_claim_support_score",
    "required_critical_coverage": "required_critical_coverage",
    # Verification profile (fast / balanced / strict / custom)
    "verification_profile": "verification_profile",
    "verification_max_workers": "verification_max_workers",
    "verification_batch_size": "verification_batch_size",
    "verification_cache_enabled": "verification_cache_enabled",
    "verification_timeout_seconds": "verification_timeout_seconds",
    "verification_minor_claim_sample_rate":
        "verification_minor_claim_sample_rate",
    # Adaptive coverage / audit log / Local LLM role routing
    "adaptive_coverage": "adaptive_coverage",
    "requirement_max_search_attempts": "requirement_max_search_attempts",
    "max_stall_rounds": "max_stall_rounds",
    "audit_log_enabled": "audit_log_enabled",
    "local_llm_role": "local_llm_role",
    "local_timeout": "local_timeout",
    "local_concurrency": "local_concurrency",
    # reuse / freshness
    "cache_reuse": "cache_reuse",
    "refresh_fetched": "refresh_fetched",
    "cache_dir": "cache_dir",
    # research presets (Web UI 短時間/標準/詳細) map to concrete knobs
    "iterations": "research_iterations",
    "max_iterations": "max_iterations",
    "deep_think": "deep_think",
    "v2_enable_polish": "v2_enable_polish",
    # settings the Tk GUI used to offer (the Web UI is now the GUI)
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "max_results": "max_results",
    "target_pages": "target_pages",
    "search_region": "search_region",
    "include_images": "include_images",
    "include_citations": "include_citations",
    "include_toc": "include_toc",
}


def _int_ge0(value, name):
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer (got {value!r})")
    if n < 0:
        raise ValueError(f"{name} must be >= 0 (got {n})")
    return n


def _float_range(lo, hi, inclusive_hi=True):
    def convert(value, name):
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number (got {value!r})")
        ok = (lo <= f <= hi) if inclusive_hi else (lo <= f < hi)
        if not ok:
            op = "<=" if inclusive_hi else "<"
            raise ValueError(
                f"{name} must satisfy {lo} <= value {op} {hi} (got {f})")
        return f
    return convert


def _length_mode(value, name):
    v = str(value).strip().lower()
    if v not in ("adaptive", "fixed"):
        raise ValueError(f"{name} must be 'adaptive' or 'fixed' (got {value!r})")
    return v


def _strict_workers(value, name):
    """Strict app-wide worker validation (1..16; bool/float/junk rejected;
    NEVER clamped — invalid input is a 400, not a silently adjusted value)."""
    from ..utils.concurrency import validate_parallel_max_workers
    return validate_parallel_max_workers(value, source=name)


# Server-side type conversion & validation for length/loop settings.
# Empty strings are dropped BEFORE conversion (=> config defaults / None);
# invalid values raise ValueError -> HTTP 400 before the job starts.
def _profile(value, name):
    v = str(value).strip().lower()
    if v not in ("fast", "balanced", "strict", "custom"):
        raise ValueError(f"{name} must be fast/balanced/strict/custom "
                         f"(got {value!r})")
    return v


def _local_llm_role(value, name):
    v = str(value).strip().lower()
    if v not in ("off", "verify", "draft", "all"):
        raise ValueError(f"{name} must be off/verify/draft/all "
                         f"(got {value!r})")
    return v


def _int_range(lo, hi):
    def convert(value, name):
        try:
            if isinstance(value, bool) or isinstance(value, float):
                raise ValueError
            n = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer (got {value!r})")
        if not (lo <= n <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi} "
                             f"(got {n})")
        return n
    return convert


_PARAM_CONVERTERS = {
    "parallel_max_workers": _strict_workers,
    "verification_profile": _profile,
    "verification_max_workers": _int_range(1, 16),
    "verification_batch_size": _int_range(1, 32),
    "verification_timeout_seconds": _int_range(0, 86400),
    "verification_minor_claim_sample_rate": _float_range(0.0, 1.0),
    "requirement_max_search_attempts": _int_range(1, 10),
    "max_stall_rounds": _int_range(1, 10),
    "local_llm_role": _local_llm_role,
    "local_timeout": _int_range(1, 3600),
    "local_concurrency": _int_range(1, 16),
    "ai_crawl_max_pages": _int_range(1, 200),
    "ai_crawl_site_depth": _int_range(0, 10),      # 0 = seeds only
    "gap_fill_rounds": _int_range(0, 10),
    "max_pages_per_query": _int_range(1, 50),
    "per_domain_delay": _float_range(0.0, 60.0),
    "plan_review_timeout": _int_range(0, 3600),    # 0 = explicit approval
    "length_mode": _length_mode,
    "preferred_body_chars": _int_ge0,
    "hard_min_body_chars": _int_ge0,
    "hard_max_body_chars": _int_ge0,
    "length_tolerance": _float_range(0.0, 1.0, inclusive_hi=False),
    "max_final_research_rounds": _int_ge0,
    "max_final_revision_rounds": _int_ge0,
    "max_no_improvement_rounds": _int_ge0,
    "min_score_improvement": _float_range(0.0, 1.0),
    "min_new_independent_sources": _int_ge0,
    "min_claim_support_score": _float_range(0.0, 1.0),
    "required_critical_coverage": _float_range(0.0, 1.0),
    "temperature": _float_range(0.0, 2.0),
    "max_tokens": _int_range(1, 1_000_000),
    "max_results": _int_range(1, 100),
    "target_pages": _int_range(1, 500),
}


class FieldError(ValueError):
    """A parameter validation error that names the offending UI field so
    the form can show the message NEXT TO that field."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def build_config_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map UI request parameters to create_config kwargs.

    Unknown keys are ignored; empty strings are dropped so config defaults
    apply (e.g. an empty hard_max_body_chars becomes None). Length/loop
    settings are type-converted and validated server-side; an invalid
    value raises ValueError (the API returns HTTP 400 without starting a
    job). Per-stage LLM overrides arrive as params["stage_llm"] =
    {stage: {"provider": ..., "model": ..., "api_key": ...}} and are passed
    through with empty entries removed.
    """
    kwargs: Dict[str, Any] = {}
    for ui_key, config_key in _CONFIG_PARAM_MAP.items():
        if ui_key not in params:
            continue
        value = params[ui_key]
        if value is None or value == "":
            continue
        converter = _PARAM_CONVERTERS.get(ui_key)
        if converter is not None:
            try:
                value = converter(value, ui_key)
            except ValueError as e:
                raise FieldError(ui_key, str(e)) from e
        kwargs[config_key] = value

    # cross-field check mirrored from Config.validate()
    hard_min = kwargs.get("hard_min_body_chars")
    hard_max = kwargs.get("hard_max_body_chars")
    if hard_min is not None and hard_max is not None and hard_min > hard_max:
        raise ValueError(
            f"hard_min_body_chars ({hard_min}) must be <= "
            f"hard_max_body_chars ({hard_max})")

    # source_mode 'local' without web never touches the network; documents
    # themselves are passed to tool.run, not create_config
    stage_llm = params.get("stage_llm") or {}
    cleaned_stages = {}
    for stage, spec in stage_llm.items():
        if not isinstance(spec, dict):
            continue
        cleaned = {k: v for k, v in spec.items() if v not in (None, "")}
        if cleaned.get("provider") or cleaned.get("model"):
            cleaned_stages[stage] = cleaned
    if cleaned_stages:
        kwargs["stage_llm"] = cleaned_stages

    return kwargs


SUPPORTED_DOC_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".pptx"}


def precheck_documents(paths) -> list:
    """Read each document ONCE and report whether it is usable.

    Returns [{"path","name","size","readable","chars","error"}] — the UI
    shows this before the run so unreadable files are fixed up front
    instead of silently dropping out of the research.
    """
    from ..utils.document_reader import DocumentReader
    reader = DocumentReader()
    out = []
    expanded = []
    for raw in (paths or []):
        if not raw:
            continue
        rp = Path(raw)
        if rp.is_dir():
            expanded.extend(expand_document_paths([raw]))
        else:
            expanded.append(raw)          # missing files are REPORTED, not dropped
    for raw in expanded:
        p = Path(raw)
        entry = {"path": str(p), "name": p.name,
                 "size": p.stat().st_size if p.exists() else 0,
                 "readable": False, "chars": 0, "error": None}
        if not p.exists():
            entry["error"] = "ファイルが見つかりません"
        elif p.suffix.lower() not in SUPPORTED_DOC_SUFFIXES:
            entry["error"] = f"未対応の形式です ({p.suffix})"
        else:
            try:
                doc = reader.read_document(p)
                if doc.error:
                    entry["error"] = doc.error
                else:
                    entry["chars"] = len(doc.content or "")
                    entry["readable"] = entry["chars"] > 0
                    if not entry["readable"]:
                        entry["error"] = "本文を抽出できませんでした（画像のみのPDF等）"
            except Exception as e:
                entry["error"] = str(e)[:200]
        out.append(entry)
    return out


def expand_document_paths(raw_paths) -> list:
    """
    Expand a list of file/directory paths into document file paths.

    Directories are scanned (non-recursive) for supported document types.
    Nonexistent paths are skipped.
    """
    files = []
    for raw in raw_paths or []:
        raw = str(raw).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_DOC_SUFFIXES:
                    files.append(str(f))
        elif path.is_file():
            files.append(str(path))
    return files


# keys that must never be persisted, echoed, or copied into "duplicate
# these conditions" payloads
_SECRET_KEYS = {"openai_api_key", "anthropic_api_key", "local_api_key",
                "api_key", "proxy_password", "proxy_username"}


def scrub_secrets(params: Any) -> Any:
    """Deep-copy ``params`` with every secret key removed (nested too)."""
    if isinstance(params, dict):
        return {k: scrub_secrets(v) for k, v in params.items()
                if k not in _SECRET_KEYS}
    if isinstance(params, list):
        return [scrub_secrets(v) for v in params]
    return copy.deepcopy(params)


TERMINAL_STATES = ("completed", "error", "cancelled", "interrupted")


class ResearchJob:
    """State of one research run, shared between worker and HTTP threads."""

    def __init__(self, job_id: str, query: str, params: Dict[str, Any] = None):
        self.job_id = job_id
        self.query = query
        # queued / running / plan_review / completed / error / cancelled
        self.state = "running"
        self.progress = 0.0
        self.message = "開始しています..."
        self.log = deque(maxlen=300)
        self._log_seq = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        # fixed when the job reaches a terminal state: elapsed time stops
        self.finished_at: Optional[float] = None
        self.queue_position: Optional[int] = None
        # non-secret parameter snapshot (history / "duplicate conditions")
        self.params_summary: Dict[str, Any] = scrub_secrets(params or {})
        self._lock = threading.Lock()
        # Live report preview (WebUILiveSink), set by the worker thread
        self.live_sink = None
        # callable returning the run's VerificationProgress (or None)
        self.verification_source = None
        # callable cancelling the WHOLE run (tool.request_cancel)
        self.run_cancel = None
        # callable returning the artifacts the researcher has ACTUALLY
        # written so far (checkpoints) — the execution panel shows these
        # while the run is still going, never a guessed "saved"
        self.artifacts_source = None
        # Plan review state (state == "plan_review")
        self.plan: Optional[Dict[str, Any]] = None
        self.plan_review_deadline: Optional[float] = None
        self._plan_event: Optional[threading.Event] = None
        self._plan_response: Optional[Dict[str, str]] = None

    def verification_snapshot(self):
        source = self.verification_source
        if source is None:
            return None
        try:
            progress = source()
            return progress.snapshot() if progress is not None else None
        except Exception:
            return None

    def live_saved_artifacts(self) -> Dict[str, str]:
        """Artifacts written so far (finished jobs: from the result)."""
        if self.result and isinstance(self.result.get("saved_artifacts"),
                                      dict):
            return dict(self.result["saved_artifacts"])
        source = self.artifacts_source
        if source is None:
            return {}
        try:
            return dict(source() or {})
        except Exception:
            return {}

    def cancel_verification(self) -> bool:
        """Request a SAFE verification cancel (idempotent)."""
        source = self.verification_source
        if source is None:
            return False
        try:
            progress = source()
        except Exception:
            return False
        if progress is None:
            return False
        progress.cancel()
        return True

    def cancel_run(self) -> bool:
        """Cancel the WHOLE research run (safe checkpoints, idempotent).

        Propagates a real cancel token into the worker: no new
        LLM/search/fetch work starts after the next checkpoint, permits
        are released normally, and the job ends in the terminal
        ``cancelled`` state with its partial artifacts. A job waiting in
        plan review is released immediately with action="cancel" (a
        cancel is never mistaken for an approval).
        """
        released = False
        with self._lock:
            if self.state == "plan_review" and self._plan_event is not None:
                self._plan_response = {"action": "cancel", "instructions": ""}
                self._plan_event.set()
                released = True
            self.cancel_requested = True
            self.message = "停止要求を受け付けました。実行中のリクエストの終了を待っています…"
        fn = self.run_cancel
        if fn is None:
            return released
        try:
            fn()
            return True
        except Exception:
            return released

    def update(self, message: str, percentage: float) -> None:
        with self._lock:
            if percentage >= 0:
                self.progress = max(self.progress, min(percentage, 100.0))
            self.message = message
            self._log_seq += 1
            self.log.append({
                "seq": self._log_seq,        # monotonic: UI diffs by seq
                "t": round(time.time() - self.started_at, 1),
                "pct": round(self.progress, 1),
                "msg": message,
            })

    def finish(self, state: str) -> None:
        """Enter a TERMINAL state and freeze the elapsed time."""
        with self._lock:
            self.state = state
            if self.finished_at is None:
                self.finished_at = time.time()
            self.queue_position = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(max(0.0, end - self.started_at), 1)

    def begin_plan_review(self, plan_dict: Dict[str, Any],
                          timeout: Optional[float]):
        """Enter plan-review state and block until a response (or timeout).

        ``timeout`` None/0 = EXPLICIT mode: wait until the user approves,
        revises or cancels — never auto-start. A positive timeout keeps
        the opt-in auto-start behavior (None response = auto-approve).
        Returns {"action": "approve"|"revise"|"cancel", "instructions"}.
        """
        explicit = not timeout or timeout <= 0
        with self._lock:
            self.plan = plan_dict
            self.plan_review_deadline = None if explicit \
                else time.time() + timeout
            self._plan_event = threading.Event()
            self._plan_response = None
            self.state = "plan_review"
        self._plan_event.wait(None if explicit else timeout)
        with self._lock:
            response = self._plan_response
            self.state = "running"
            self.plan_review_deadline = None
            self._plan_event = None
            self._plan_response = None
        return response

    def respond_plan_review(self, action: str, instructions: str = "") -> bool:
        """Deliver the user's plan-review response. False if not reviewing."""
        with self._lock:
            if self.state != "plan_review" or self._plan_event is None:
                return False
            self._plan_response = {"action": action, "instructions": instructions}
            self._plan_event.set()
            return True

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            verification = self.verification_snapshot()
            data = {
                "job_id": self.job_id,
                "query": self.query,
                "state": self.state,
                "verification": verification,
                "progress": round(self.progress, 1),
                "message": self.message,
                "log": list(self.log)[-50:],
                "log_seq": self._log_seq,
                "result": self.result,
                "error": self.error,
                "elapsed_seconds": self.elapsed_seconds(),
                "finished_at": self.finished_at,
                "created_at": self.created_at,
                "queue_position": self.queue_position,
                "cancel_requested": bool(getattr(self, "cancel_requested",
                                                 False)),
                "plan": self.plan,
                "params_summary": self.params_summary,
                "saved_artifacts": self.live_saved_artifacts(),
            }
            if self.state == "plan_review":
                data["plan_review_remaining"] = (
                    max(0, round(self.plan_review_deadline - time.time(), 1))
                    if self.plan_review_deadline else None)
            return data


class JobManager:
    """Runs research jobs in background threads (several in parallel).

    - job ids are PERSISTENTLY unique (timestamp + uuid): a restart never
      reuses output/<job-id>/ and never overwrites another report's
      figures;
    - jobs beyond MAX_CONCURRENT wait in a FIFO QUEUE (state "queued",
      position visible, cancellable, parameters retained) instead of
      being rejected;
    - every job is recorded in a JSON LEDGER under the output directory
      (history survives restarts; secrets are never written).
    """

    MAX_CONCURRENT = 3   # simultaneous research runs (LLM rate-limit guard)
    MAX_KEPT = 20        # finished jobs kept in memory for the job list
    LEDGER_NAME = "jobs_ledger.json"

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.jobs: "Dict[str, ResearchJob]" = {}
        self._queue: "deque" = deque()          # (job, params) waiting
        self._lock = threading.Lock()
        self._ledger: Dict[str, Dict[str, Any]] = {}
        self._ledger_lock = threading.Lock()
        self._load_ledger()

    # --- persistent history ledger ------------------------------------

    def _ledger_path(self) -> Path:
        return Path(self.output_dir) / self.LEDGER_NAME

    def _load_ledger(self) -> None:
        try:
            path = self._ledger_path()
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._ledger = {str(k): v for k, v in
                                    (data.get("jobs") or {}).items()}
        except Exception as e:
            print(f"[JobManager] ledger load failed: {e}")
            self._ledger = {}
        # a record left non-terminal by a dead process (server restarted
        # mid-run) can never finish: mark it so it is not shown as running
        changed = False
        for rec in self._ledger.values():
            if isinstance(rec, dict) and rec.get("state") in (
                    "queued", "running", "plan_review"):
                rec["state"] = "interrupted"
                rec["error"] = rec.get("error") or \
                    "サーバーの再起動により完了前に中断されました"
                rec["finished_at"] = rec.get("finished_at") or \
                    rec.get("started_at")
                changed = True
        if changed:
            self._write_ledger()

    def _write_ledger(self) -> None:
        path = self._ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps({"version": 1, "jobs": self._ledger},
                                      ensure_ascii=False, indent=1,
                                      default=str),
                           encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:
            print(f"[JobManager] ledger write failed: {e}")

    def record(self, job: "ResearchJob", output_dir: str = None) -> None:
        """Upsert the job's ledger record (no secrets, no log bodies)."""
        with self._ledger_lock:
            prev = self._ledger.get(job.job_id, {})
            entry = {
                "job_id": job.job_id,
                "query": job.query,
                "state": job.state,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "elapsed_seconds": job.elapsed_seconds(),
                "output_dir": output_dir or prev.get("output_dir"),
                "params_summary": scrub_secrets(job.params_summary),
                "result": scrub_secrets(job.result) if job.result else None,
                "error": (job.error or "")[:500] or None,
            }
            self._ledger[job.job_id] = entry
            self._write_ledger()

    def history(self):
        """Ledger records newest first (includes runs from before a
        restart; live jobs are merged with their current state)."""
        with self._ledger_lock:
            records = [dict(r) for r in self._ledger.values()]   # copies
        live = {j.job_id: j for j in self.jobs.values()}
        for rec in records:
            job = live.get(rec["job_id"])
            if job is not None:
                rec["state"] = job.state
                rec["queue_position"] = job.queue_position
        records.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
        return records

    def forget(self, job_id: str) -> bool:
        with self._ledger_lock:
            removed = self._ledger.pop(job_id, None) is not None
            if removed:
                self._write_ledger()
        return removed

    @staticmethod
    def new_job_id() -> str:
        return f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # --- queries -------------------------------------------------------

    def running_jobs(self):
        return [j for j in self.jobs.values()
                if j.state in ("running", "plan_review")]

    def is_running(self) -> bool:
        return bool(self.running_jobs())

    def get(self, job_id: str) -> Optional[ResearchJob]:
        return self.jobs.get(job_id)

    def list_jobs(self):
        """All jobs, newest first (queued jobs carry their position)."""
        with self._lock:
            for pos, (job, _params) in enumerate(self._queue, 1):
                job.queue_position = pos
        return sorted(self.jobs.values(),
                      key=lambda j: j.started_at, reverse=True)

    def queued_jobs(self):
        with self._lock:
            return [job for job, _p in self._queue]

    @property
    def current(self) -> Optional[ResearchJob]:
        """Most relevant job (running first, else newest) — legacy accessor."""
        running = self.running_jobs()
        if running:
            return max(running, key=lambda j: j.started_at)
        jobs = self.list_jobs()
        return jobs[0] if jobs else None

    # --- lifecycle -----------------------------------------------------

    def start(self, params: Dict[str, Any]) -> ResearchJob:
        """Create a job; run it now or QUEUE it when at capacity."""
        # Isolate each job's artifacts so parallel runs can't clobber each
        # other's figures / evidence / session files. The id is unique
        # across restarts, so output/<job-id>/ is never reused.
        params = dict(params)
        job = ResearchJob(self.new_job_id(), params.get("query", ""),
                          params=params)
        base = params.get("output_dir") or self.output_dir
        if params.get("resume_from"):
            # a resumed run writes into the ORIGINAL job's directory so
            # its session / evidence / reports stay together
            params["output_dir"] = params.get("resume_output_dir") or \
                str(Path(params["resume_from"]).parent)
        elif params.get("regenerate"):
            params["output_dir"] = params.get("regenerate_output_dir") or \
                str(Path(params["regenerate"]["session_path"]).parent)
        else:
            params["output_dir"] = str(Path(base) / job.job_id)
        # ONE page/extraction cache shared by every job of this server
        params.setdefault("cache_dir", str(Path(base) / ".cache"))
        with self._lock:
            self.jobs[job.job_id] = job
            self._trim_finished()
            at_capacity = len(self.running_jobs()) >= self.MAX_CONCURRENT
            if at_capacity:
                job.state = "queued"
                self._queue.append((job, params))
                job.queue_position = len(self._queue)
                job.update(f"実行待ち（{job.queue_position}番目）: 同時実行数"
                           f"{self.MAX_CONCURRENT}件の空きを待っています", 0)
        self.record(job, output_dir=params["output_dir"])
        if not at_capacity:
            self._launch(job, params)
        return job

    def _launch(self, job: ResearchJob, params: Dict[str, Any]) -> None:
        job.started_at = time.time()      # queue wait is not run time
        job.state = "running"
        job.queue_position = None
        thread = threading.Thread(
            target=self._run, args=(job, params), daemon=True,
        )
        thread.start()

    def _pump_queue(self) -> None:
        """Start queued jobs while capacity allows (FIFO)."""
        to_launch = []
        with self._lock:
            while self._queue and \
                    len(self.running_jobs()) < self.MAX_CONCURRENT:
                job, params = self._queue.popleft()
                if job.state != "queued":       # cancelled while waiting
                    continue
                job.state = "running"
                to_launch.append((job, params))
        for job, params in to_launch:
            self._launch(job, params)

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued OR running job."""
        job = self.jobs.get(job_id)
        if job is None:
            return False
        with self._lock:
            if job.state == "queued":
                self._queue = deque((j, p) for j, p in self._queue
                                    if j.job_id != job_id)
                job.finish("cancelled")
                job.update("実行待ちのままキャンセルしました", -1)
                cancelled = True
            else:
                cancelled = False
        if cancelled:
            self.record(job)
            return True
        ok = job.cancel_run()
        job.cancel_verification()          # also stop verification work
        return ok

    def _trim_finished(self) -> None:
        """Drop the oldest finished jobs beyond MAX_KEPT (callers hold _lock).
        Cancelled jobs are TERMINAL and are cleaned up like any other; the
        ledger keeps their history."""
        finished = [j for j in self.jobs.values() if j.is_terminal]
        excess = len(finished) - self.MAX_KEPT
        if excess > 0:
            for job in sorted(finished, key=lambda j: j.started_at)[:excess]:
                self.jobs.pop(job.job_id, None)

    def _run_fermi(self, job: ResearchJob, params: Dict[str, Any],
                   job_warnings) -> None:
        """Fermi estimation as a lightweight job (one LLM call, no web):
        the browser-based replacement for the former Tk fermi_gui."""
        from ..api import get_client
        from ..thinking import FermiEstimator
        spec = params["fermi"]
        question = (spec.get("question") or "").strip()
        provider = params.get("provider") or "openai"
        key_name = {"openai": "openai_api_key", "anthropic": "anthropic_api_key",
                    "local": "local_api_key"}.get(provider, "openai_api_key")
        base_url = {"openai": "openai_base_url", "anthropic": "anthropic_base_url",
                    "local": "local_base_url"}.get(provider)
        job.update("フェルミ推定を実行しています（LLM呼び出し 1回）…", 20)
        llm = get_client(
            provider=provider, api_key=params.get(key_name) or None,
            model=params.get("model") or None,
            http_proxy=params.get("http_proxy") or None,
            https_proxy=params.get("https_proxy") or None,
            verify_ssl=params.get("verify_ssl", True) is not False,
            base_url=params.get(base_url) or None if base_url else None,
            backend=params.get("local_backend") or None,
        )
        estimator = FermiEstimator(llm_client=llm,
                                   language=spec.get("language") or "ja")
        estimate = estimator.estimate(
            question=question, context=spec.get("context") or "",
            known_values=spec.get("known_values") or None)
        out_dir = Path(params.get("output_dir") or self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"fermi_{job.job_id}"
        saved: Dict[str, str] = {}
        md_path = stem.with_suffix(".md")
        md_path.write_text(estimate.to_markdown(), encoding="utf-8")
        saved["fermi_md"] = str(md_path)
        json_path = stem.with_suffix(".json")
        json_path.write_text(json.dumps(estimate.to_dict(), ensure_ascii=False,
                                        indent=1), encoding="utf-8")
        saved["fermi_json"] = str(json_path)
        for ext, fn in (("docx", estimate.save_docx), ("pdf", estimate.save_pdf)):
            try:
                path = fn(stem.with_suffix("." + ext))
                saved[f"fermi_{ext}"] = str(path)
            except Exception as e:            # optional dependency missing
                job_warnings.add(job_warnings.LOW, "Fermi",
                                 f"{ext} 出力を省略しました: {e}")
        job.result = {
            "kind": "fermi",
            "fermi": estimate.to_dict(),
            "fermi_markdown": estimate.to_markdown(),
            "report_path": str(md_path),
            "saved_artifacts": saved,
            "warnings": job_warnings.to_dict_list(),
            "warning_count": job_warnings.count(),
            "status": {"process": "completed", "verification": "skipped",
                       "quality": "unverified"},
            "run_status": "completed",
            "output_dir": str(out_dir),
        }
        job.progress = 100.0
        job.finish("completed")
        job.update("フェルミ推定が完了しました（推定値は検証していません）", 100)

    def _run(self, job: ResearchJob, params: Dict[str, Any]) -> None:
        from ..utils.helpers import ResearchWarnings
        # PER-JOB warning collector bound to this thread's context; every
        # parallel stage inherits it (ContextThreadPoolExecutor), so other
        # jobs' warnings never leak into this job's report/result
        job_warnings = ResearchWarnings.bind()
        ResearchWarnings.begin_run()
        tool = None
        try:
            from ..config import create_config
            from ..main import DeepResearchTool

            if params.get("regenerate"):
                # re-output from the SAVED research: no search, no LLM
                from ..main import regenerate_from_session
                spec = params["regenerate"]
                job.update("保存済みの調査結果から再出力しています…", 10)
                out = regenerate_from_session(
                    spec["session_path"],
                    output_format=spec.get("output_format", "markdown"),
                    mode=spec.get("mode", "full"))
                job.result = {
                    "report_path": out["report_path"],
                    "regenerated_from": out["source"],
                    "mode": out["mode"],
                    "warnings": job_warnings.to_dict_list(),
                    "warning_count": job_warnings.count(),
                    # the axes stored WITH the frozen body (a cancelled or
                    # timed-out verification stays "not verified")
                    "status": out.get("status") or {
                        "process": "completed", "verification": "skipped",
                        "quality": "unverified"},
                    "decision": out.get("decision"),
                    "run_status": "completed",
                    "saved_artifacts": {"report": out["report_path"]},
                    "output_dir": params.get("output_dir"),
                }
                job.progress = 100.0
                job.finish("completed")
                job.update("再出力が完了しました", 100)
                return

            if params.get("fermi"):
                self._run_fermi(job, params, job_warnings)
                return

            config = create_config(**build_config_kwargs(params))
            job.update("設定を構築しました。ツールを初期化中...", 1)

            tool = DeepResearchTool(config)
            # verification progress + safe cancel become pollable; the
            # run-level cancel propagates into the worker
            job.verification_source = (
                lambda: getattr(tool, "verification_progress", None))
            job.run_cancel = tool.request_cancel
            job.artifacts_source = tool.current_saved_artifacts
            job.update("調査を開始します", 2)

            documents = expand_document_paths(params.get("local_documents"))
            if params.get("source_mode") == "local" and not documents:
                raise ValueError(
                    "ローカル文献モードには文書パスの指定が必要です"
                )
            if documents:
                job.update(f"ローカル文書 {len(documents)} 件を読み込みます", 1)

            # Plan review: pause after plan generation so the user can
            # inspect / revise it in the UI. Auto-start after a timeout is
            # OPT-IN (plan_review_timeout > 0); the default waits for an
            # explicit approve / revise / cancel.
            plan_cb = None
            if params.get("plan_review", True):
                raw_timeout = params.get("plan_review_timeout")
                try:
                    plan_timeout = float(raw_timeout) if raw_timeout else 0.0
                except (TypeError, ValueError):
                    plan_timeout = 0.0

                def plan_cb(plan, revise_fn):
                    from ..utils.cancellation import RunCancelled
                    current = plan
                    changed = False
                    for _ in range(5):  # revision round limit
                        if plan_timeout > 0:
                            job.update(
                                f"調査計画のレビュー待ちです（{int(plan_timeout)}秒以内に"
                                f"応答がなければこのまま開始します）", 10)
                        else:
                            job.update("調査計画のレビュー待ちです（承認・修正・"
                                       "中止のいずれかを選ぶまで開始しません）", 10)
                        response = job.begin_plan_review(
                            current.to_dict(), plan_timeout)
                        if response is None:
                            job.update("応答がないため、この計画で調査を開始します", 10)
                            return current if changed else None
                        if response.get("action") == "cancel":
                            job.update("計画レビュー中に中止されました", -1)
                            raise RunCancelled("cancelled during plan review")
                        if response.get("action") != "revise":
                            job.update("計画が承認されました。調査を開始します", 10)
                            return current if changed else None
                        instructions = (response.get("instructions") or "").strip()
                        if not instructions:
                            return current if changed else None
                        job.update("計画を修正しています...", 10)
                        try:
                            current = revise_fn(current, instructions)
                            changed = True
                        except Exception as e:
                            job.update(f"計画の修正に失敗しました（{e}）。"
                                       f"現在の計画で開始します", 10)
                            return current if changed else None
                    job.update("修正回数の上限に達しました。この計画で開始します", 10)
                    return current if changed else None

            # Live report preview: the UI polls /api/live-report for the
            # chapters/figures as they are written
            from ..report.live_report import WebUILiveSink
            job.live_sink = WebUILiveSink()

            result = tool.run(
                query=params.get("query", ""),
                requirements=params.get("requirements", ""),
                additional_documents=documents or None,
                progress_callback=job.update,
                plan_review_callback=plan_cb,
                live_sink=job.live_sink,
                resume_from=params.get("resume_from"),
            )

            job.result = {
                "report_path": result.get("report_path"),
                "evidence_json": result.get("evidence_json"),
                "evidence_csv": result.get("evidence_csv"),
                "verification_html": result.get("verification_html"),
                "session_id": result.get("session_id"),
                "token_usage": result.get("token_usage"),
                # this job's OWN warnings (context-bound collector)
                "warnings": job_warnings.to_dict_list(),
                "warning_count": job_warnings.count(),
                "verification_summary": result.get("verification_summary"),
                # process / verification / quality — three separate facts
                "status": result.get("status") or {},
                "run_status": result.get("run_status", "completed"),
                # artifacts that were ACTUALLY written (checkpoints)
                "saved_artifacts": result.get("saved_artifacts") or {},
                "semantic_artifact_check": result.get("semantic_artifact_check"),
                "output_dir": params.get("output_dir"),
                "timings": result.get("timings"),
                "resume_plan": result.get("resume_plan"),
            }
            job.progress = 100.0
            if result.get("verification_cancelled"):
                job.finish("cancelled")
                job.update("検証をキャンセルしました（保存済みの成果物のみ"
                           "表示しています）", 100)
            elif result.get("run_status", "completed") != "completed":
                # semantic mismatch etc.: never a normal completion
                job.error = (f"run_status={result['run_status']}: "
                             f"成果物検査に失敗したため正常完了ではありません")
                job.finish("error")
                job.update(f"検査失敗（{result['run_status']}）", -1)
            else:
                job.finish("completed")
                axes = job.result.get("status") or {}
                if axes.get("verification") == "performed" and \
                        axes.get("quality") == "passed":
                    job.update("完了しました（検証済み・品質基準達成）", 100)
                elif axes.get("verification") == "performed":
                    job.update("完了しました（検証済み・要確認事項あり）", 100)
                else:
                    job.update("処理が完了しました（未検証）", 100)
        except Exception as e:
            from ..utils.cancellation import RunCancelled
            if isinstance(e, RunCancelled):
                # a cancelled run reports ONLY the artifacts that were
                # actually written by the researcher's checkpoint
                saved = dict(getattr(tool, "partial_artifacts", {}) or {})
                job.result = {
                    "saved_artifacts": saved,
                    "session_id": getattr(tool, "partial_session_id", None),
                    "warnings": job_warnings.to_dict_list(),
                    "warning_count": job_warnings.count(),
                    "status": {"process": "cancelled",
                               "verification": "skipped",
                               "quality": "unverified"},
                    "run_status": "cancelled",
                    "output_dir": params.get("output_dir"),
                }
                job.finish("cancelled")
                if saved:
                    job.update(f"中止しました。保存済み: "
                               f"{', '.join(sorted(saved))}", -1)
                else:
                    job.update("中止しました（保存済みの成果物はありません）", -1)
            else:
                job.error = f"{e}\n{traceback.format_exc(limit=3)}"
                job.finish("error")
                job.update(f"エラー: {e}", -1)
        finally:
            ResearchWarnings.end_run()
            ResearchWarnings.unbind()
            try:
                self.record(job, output_dir=params.get("output_dir"))
            except Exception:
                pass
            self._pump_queue()


class WebUIHandler(BaseHTTPRequestHandler):
    """HTTP handler; job manager and output dir are set on the server."""

    server_version = "DeepResearchWebUI/1.0"

    # ---- helpers ----

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str, download: bool = False) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"',
            )
        self.end_headers()
        self.wfile.write(data)

    def _output_dir(self) -> Path:
        return Path(getattr(self.server, "output_dir", "./output")).resolve()

    def _safe_output_path(self, raw: str) -> Optional[Path]:
        """Resolve a path and require it under the output directory."""
        try:
            path = Path(raw).resolve()
            path.relative_to(self._output_dir())
            return path if path.is_file() else None
        except (ValueError, OSError):
            return None

    def log_message(self, fmt, *args):  # quiet default logging
        pass

    def _history_record(self, manager, params):
        """Resolve a resume/regenerate target to a ledger record with
        a session_path (from the job's saved artifacts or output dir)."""
        job_id = params.get("job_id") or ""
        rec = None
        if job_id:
            for r in manager.history():
                if r.get("job_id") == job_id:
                    rec = dict(r)
                    break
            if rec is None:
                return None
        else:
            sp = params.get("session_path")
            if not sp:
                return None
            safe = self._safe_output_path(sp)
            if safe is None:
                return None
            rec = {"query": "", "output_dir": str(safe.parent),
                   "params_summary": {}}
            rec["session_path"] = str(safe)
            return rec
        result = rec.get("result") or {}
        saved = result.get("saved_artifacts") or {}
        session_path = saved.get("session")
        if not session_path and rec.get("output_dir"):
            candidates = sorted(Path(rec["output_dir"]).glob("session_*.json"))
            if candidates:
                session_path = str(candidates[-1])
        rec["session_path"] = session_path
        return rec

    def _handle_upload(self):
        """multipart/form-data upload of the BROWSER's local files into
        <output>/uploads/<token>/; each file is read back once so the
        UI can show whether it is usable before the run starts."""
        import email
        from email import policy
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "multipart/form-data required"}, 400)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 200 * 1024 * 1024:
            self._send_json({"error": "upload too large or empty"}, 413)
            return
        body = self.rfile.read(length)
        msg = email.message_from_bytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body,
            policy=policy.default)
        token = uuid.uuid4().hex[:10]
        target_dir = self._output_dir() / "uploads" / token
        target_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for part in msg.iter_parts():
            filename = part.get_filename()
            if not filename:
                continue
            safe_name = Path(filename).name.replace("..", "_") or "file"
            dest = target_dir / safe_name
            dest.write_bytes(part.get_payload(decode=True) or b"")
            saved.append(str(dest))
        self._send_json({"files": precheck_documents(saved),
                         "upload_dir": str(target_dir)})

    # ---- routes ----

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            index = STATIC_DIR / "index.html"
            if index.is_file():
                self._send_file(index, "text/html; charset=utf-8")
            else:
                self._send_json({"error": "index.html not found"}, 500)
        elif route == "/api/version":
            self._send_json({"version": __version__})
        elif route == "/api/jobs":
            manager: JobManager = self.server.job_manager
            self._send_json({
                "version": __version__,
                "max_concurrent": manager.MAX_CONCURRENT,
                "queued": len(manager.queued_jobs()),
                "jobs": [j.to_dict() for j in manager.list_jobs()],
            })
        elif route == "/api/history":
            manager: JobManager = self.server.job_manager
            self._send_json({"history": manager.history()})
        elif route == "/api/status":
            manager: JobManager = self.server.job_manager
            query = parse_qs(parsed.query)
            job_id = (query.get("job_id") or [""])[0]
            manager.list_jobs()          # refreshes queue positions
            job = manager.get(job_id) if job_id else manager.current
            if job is None:
                self._send_json({"state": "idle", "version": __version__})
            else:
                data = job.to_dict()
                data["version"] = __version__
                self._send_json(data)
        elif route == "/api/live-report":
            manager: JobManager = self.server.job_manager
            query = parse_qs(parsed.query)
            job_id = (query.get("job_id") or [""])[0]
            job = manager.get(job_id) if job_id else manager.current
            if job is None or job.live_sink is None:
                self._send_json({"available": False})
            else:
                data = job.live_sink.snapshot()
                data["available"] = True
                data["job_id"] = job.job_id
                data["state"] = job.state
                self._send_json(data)
        elif route == "/api/reports":
            # Recursive: parallel jobs write under output/<job-id>/reports/
            base = self._output_dir()
            files = []
            if base.is_dir():
                for f in base.rglob("*"):
                    if f.is_file() and f.parent.name == "reports":
                        files.append(f)
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            self._send_json({"reports": [
                {
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "mtime": f.stat().st_mtime,
                }
                for f in files[:50]
            ]})
        elif route == "/api/report-file":
            query = parse_qs(parsed.query)
            raw = (query.get("path") or [""])[0]
            path = self._safe_output_path(raw)
            if path is None:
                self._send_json({"error": "invalid path"}, 400)
                return
            content_type = {
                ".md": "text/markdown; charset=utf-8",
                ".html": "text/html; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
            }.get(path.suffix.lower(), "application/octet-stream")
            download = (query.get("download") or ["0"])[0] == "1"
            self._send_file(path, content_type, download=download)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._handle_upload()
            return
        if parsed.path not in ("/api/research", "/api/plan-review",
                               "/api/cancel-verification",
                               "/api/cancel-run", "/api/precheck",
                               "/api/resume", "/api/regenerate",
                               "/api/history/forget", "/api/fermi"):
            self._send_json({"error": "not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        manager: JobManager = self.server.job_manager

        if parsed.path == "/api/precheck":
            # readability pre-check of SERVER-side paths (never confused
            # with the browser's local files, which go through /api/upload)
            self._send_json({"files": precheck_documents(
                params.get("paths") or [])})
            return

        if parsed.path == "/api/fermi":
            question = (params.get("question") or "").strip()
            if not question:
                self._send_json({"error": "question is required",
                                 "field": "fermi_question"}, 400)
                return
            known = params.get("known_values") or {}
            if not isinstance(known, dict):
                self._send_json({"error": "known_values must be an object",
                                 "field": "fermi_known"}, 400)
                return
            cleaned = {}
            for name, value in known.items():
                try:
                    cleaned[str(name)] = float(str(value).replace(",", ""))
                except (TypeError, ValueError):
                    self._send_json({"error": f"known value '{name}' is not a number",
                                     "field": "fermi_known"}, 400)
                    return
            job_params = {k: v for k, v in params.items()
                          if k not in ("question", "context", "known_values",
                                       "language")}
            job_params["query"] = f"フェルミ推定: {question}"
            job_params["fermi"] = {
                "question": question, "context": params.get("context") or "",
                "known_values": cleaned or None,
                "language": params.get("language") or "ja",
            }
            job = manager.start(job_params)
            self._send_json({"job_id": job.job_id, "state": job.state}, 202)
            return

        if parsed.path == "/api/history/forget":
            ok = manager.forget(params.get("job_id") or "")
            self._send_json({"ok": ok}, 200 if ok else 404)
            return

        if parsed.path in ("/api/resume", "/api/regenerate"):
            rec = self._history_record(manager, params)
            if rec is None:
                self._send_json({"error": "no such job / session"}, 404)
                return
            session_path = rec.get("session_path")
            if not session_path or not Path(session_path).is_file():
                self._send_json({"error": "saved session not found — "
                                          "nothing to resume/re-output"}, 409)
                return
            if parsed.path == "/api/resume":
                base_params = dict(rec.get("params_summary") or {})
                base_params.update(params.get("overrides") or {})
                base_params["resume_from"] = session_path
                base_params["resume_output_dir"] = rec.get("output_dir")
                # secrets come from the CURRENT request only (never stored)
                for k in ("openai_api_key", "anthropic_api_key",
                          "local_api_key", "stage_llm"):
                    if params.get(k) is not None:
                        base_params[k] = params[k]
                job = manager.start(base_params)
            else:
                fmt = params.get("output_format", "markdown")
                mode = params.get("mode", "full")
                if fmt not in ("markdown", "docx", "pdf", "html"):
                    self._send_json({"error": f"unsupported output_format: {fmt}",
                                     "field": "output_format"}, 400)
                    return
                if mode not in ("full", "summary"):
                    self._send_json({"error": f"mode must be 'full' or 'summary' (got {mode})",
                                     "field": "mode"}, 400)
                    return
                job = manager.start({
                    "query": f"再出力: {rec.get('query', '')}",
                    "regenerate": {
                        "session_path": session_path,
                        "output_format": fmt,
                        "mode": mode,
                    },
                    "regenerate_output_dir": rec.get("output_dir"),
                })
            self._send_json({"job_id": job.job_id, "state": job.state}, 202)
            return

        if parsed.path == "/api/cancel-verification":
            job_id = params.get("job_id") or ""
            job = manager.get(job_id) if job_id else manager.current
            if job is None:
                self._send_json({"error": "no such job"}, 404)
                return
            ok = job.cancel_verification()
            # idempotent: repeated cancels simply return ok/false
            self._send_json({"ok": ok}, 200 if ok else 409)
            return

        if parsed.path == "/api/cancel-run":
            # SAFE whole-run cancellation: the worker stops at its next
            # checkpoint; the job ends as terminal "cancelled" with its
            # partial artifacts. Idempotent; 409 when nothing to cancel
            # (the UI restores the cancel button on non-200).
            job_id = params.get("job_id") or ""
            job = manager.get(job_id) if job_id else manager.current
            if job is None:
                self._send_json({"error": "no such job"}, 404)
                return
            ok = manager.cancel(job.job_id)   # queued or running
            self._send_json({"ok": ok, "state": job.state},
                            200 if ok else 409)
            return

        if parsed.path == "/api/plan-review":
            action = params.get("action")
            if action not in ("approve", "revise"):
                self._send_json({"error": "action must be 'approve' or 'revise'"}, 400)
                return
            job_id = params.get("job_id") or ""
            if job_id:
                job = manager.get(job_id)
            else:
                # Legacy clients without job_id: target the only reviewing job
                reviewing = [j for j in manager.running_jobs()
                             if j.state == "plan_review"]
                job = reviewing[0] if len(reviewing) == 1 else None
            if job is None or job.state != "plan_review":
                self._send_json({"error": "no job awaiting plan review"}, 409)
                return
            ok = job.respond_plan_review(
                action, params.get("instructions", "") or "")
            self._send_json({"ok": ok}, 200 if ok else 409)
            return

        if not (params.get("query") or "").strip():
            self._send_json({"error": "query is required"}, 400)
            return

        # Server-side validation BEFORE the job starts: invalid length /
        # loop settings are rejected with 400, not a failed job
        try:
            build_config_kwargs(params)
        except FieldError as e:
            self._send_json({"error": f"invalid parameter: {e}",
                             "field": e.field}, 400)
            return
        except ValueError as e:
            self._send_json({"error": f"invalid parameter: {e}"}, 400)
            return

        try:
            job = manager.start(params)
        except RuntimeError as e:
            self._send_json({"error": str(e)}, 409)
            return
        except Exception as e:
            self._send_json({"error": f"failed to start: {e}"}, 500)
            return

        self._send_json({"job_id": job.job_id, "state": job.state,
                         "queue_position": job.queue_position}, 202)


def run_server(host: str = "127.0.0.1", port: int = 8765,
               output_dir: str = "./output", open_browser: bool = False,
               fragment: str = "") -> None:
    """Start the Web UI server (blocking). With ``open_browser`` the
    default browser is opened on the UI (this is the GUI — no Tk)."""
    from ..utils.helpers import ensure_utf8_output
    ensure_utf8_output()  # avoid cp932 print crashes on Windows
    server = ThreadingHTTPServer((host, port), WebUIHandler)
    server.job_manager = JobManager(output_dir=output_dir)
    server.output_dir = output_dir
    url = f"http://{host}:{port}/{('#' + fragment) if fragment else ''}"
    print(f"Deep Research Tool v{__version__} Web UI: {url}")
    print("Ctrl+C で終了")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_server()
