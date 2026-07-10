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

import json
import threading
import time
import traceback
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
    "chart_library": "chart_library",
    "auto_figures": "auto_figures",
    "enable_verification": "enable_verification",
    "plan_review": "plan_review",
    "plan_review_timeout": "plan_review_timeout",
}


def build_config_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map UI request parameters to create_config kwargs.

    Unknown keys are ignored; empty strings are dropped so config defaults
    apply. Per-stage LLM overrides arrive as params["stage_llm"] =
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
        kwargs[config_key] = value

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


class ResearchJob:
    """State of one research run, shared between worker and HTTP threads."""

    def __init__(self, job_id: str, query: str):
        self.job_id = job_id
        self.query = query
        self.state = "running"  # running / plan_review / completed / error
        self.progress = 0.0
        self.message = "開始しています..."
        self.log = deque(maxlen=300)
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.started_at = time.time()
        self._lock = threading.Lock()
        # Plan review state (state == "plan_review")
        self.plan: Optional[Dict[str, Any]] = None
        self.plan_review_deadline: Optional[float] = None
        self._plan_event: Optional[threading.Event] = None
        self._plan_response: Optional[Dict[str, str]] = None

    def update(self, message: str, percentage: float) -> None:
        with self._lock:
            if percentage >= 0:
                self.progress = max(self.progress, min(percentage, 100.0))
            self.message = message
            self.log.append({
                "t": round(time.time() - self.started_at, 1),
                "pct": round(self.progress, 1),
                "msg": message,
            })

    def begin_plan_review(self, plan_dict: Dict[str, Any], timeout: float):
        """Enter plan-review state and block until a response or timeout.

        Returns {"action": "approve"|"revise", "instructions": str} or None
        when the timeout elapsed with no response (= auto-approve).
        """
        with self._lock:
            self.plan = plan_dict
            self.plan_review_deadline = time.time() + timeout
            self._plan_event = threading.Event()
            self._plan_response = None
            self.state = "plan_review"
        self._plan_event.wait(timeout)
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
            data = {
                "job_id": self.job_id,
                "query": self.query,
                "state": self.state,
                "progress": round(self.progress, 1),
                "message": self.message,
                "log": list(self.log)[-50:],
                "result": self.result,
                "error": self.error,
                "elapsed_seconds": round(time.time() - self.started_at, 1),
                "plan": self.plan,
            }
            if self.state == "plan_review" and self.plan_review_deadline:
                data["plan_review_remaining"] = max(
                    0, round(self.plan_review_deadline - time.time(), 1)
                )
            return data


class JobManager:
    """Runs one research job at a time in a background thread."""

    def __init__(self):
        self.current: Optional[ResearchJob] = None
        self._counter = 0
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return (self.current is not None
                and self.current.state in ("running", "plan_review"))

    def start(self, params: Dict[str, Any]) -> ResearchJob:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A research job is already running")
            self._counter += 1
            job = ResearchJob(f"job-{self._counter}", params.get("query", ""))
            self.current = job

        thread = threading.Thread(
            target=self._run, args=(job, params), daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: ResearchJob, params: Dict[str, Any]) -> None:
        try:
            from ..config import create_config
            from ..main import DeepResearchTool

            config = create_config(**build_config_kwargs(params))
            job.update("設定を構築しました。ツールを初期化中...", 1)

            tool = DeepResearchTool(config)
            job.update("調査を開始します", 2)

            documents = expand_document_paths(params.get("local_documents"))
            if params.get("source_mode") == "local" and not documents:
                raise ValueError(
                    "ローカル文献モードには文書パスの指定が必要です"
                )
            if documents:
                job.update(f"ローカル文書 {len(documents)} 件を読み込みます", 1)

            # Plan review: pause after plan generation so the user can
            # inspect / revise it in the UI; auto-continue on timeout
            plan_cb = None
            if params.get("plan_review", True):
                plan_timeout = float(params.get("plan_review_timeout") or 60)

                def plan_cb(plan, revise_fn):
                    current = plan
                    changed = False
                    for _ in range(5):  # revision round limit
                        job.update(
                            f"調査計画のレビュー待ちです（{int(plan_timeout)}秒以内に"
                            f"応答がなければこのまま開始します）", 10)
                        response = job.begin_plan_review(
                            current.to_dict(), plan_timeout)
                        if response is None:
                            job.update("応答がないため、この計画で調査を開始します", 10)
                            return current if changed else None
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

            result = tool.run(
                query=params.get("query", ""),
                requirements=params.get("requirements", ""),
                additional_documents=documents or None,
                progress_callback=job.update,
                plan_review_callback=plan_cb,
            )

            job.result = {
                "report_path": result.get("report_path"),
                "evidence_json": result.get("evidence_json"),
                "evidence_csv": result.get("evidence_csv"),
                "verification_html": result.get("verification_html"),
                "session_id": result.get("session_id"),
                "token_usage": result.get("token_usage"),
            }
            job.progress = 100.0
            job.state = "completed"
            job.update("完了しました", 100)
        except Exception as e:
            job.error = f"{e}\n{traceback.format_exc(limit=3)}"
            job.state = "error"
            job.update(f"エラー: {e}", -1)


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
        elif route == "/api/status":
            manager: JobManager = self.server.job_manager
            if manager.current is None:
                self._send_json({"state": "idle", "version": __version__})
            else:
                data = manager.current.to_dict()
                data["version"] = __version__
                self._send_json(data)
        elif route == "/api/reports":
            reports_dir = self._output_dir() / "reports"
            files = []
            if reports_dir.is_dir():
                for f in sorted(reports_dir.iterdir(),
                                key=lambda p: p.stat().st_mtime, reverse=True):
                    if f.is_file():
                        files.append({
                            "name": f.name,
                            "path": str(f),
                            "size": f.stat().st_size,
                            "mtime": f.stat().st_mtime,
                        })
            self._send_json({"reports": files[:50]})
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
        if parsed.path not in ("/api/research", "/api/plan-review"):
            self._send_json({"error": "not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        manager: JobManager = self.server.job_manager

        if parsed.path == "/api/plan-review":
            action = params.get("action")
            if action not in ("approve", "revise"):
                self._send_json({"error": "action must be 'approve' or 'revise'"}, 400)
                return
            job = manager.current
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

        try:
            job = manager.start(params)
        except RuntimeError as e:
            self._send_json({"error": str(e)}, 409)
            return
        except Exception as e:
            self._send_json({"error": f"failed to start: {e}"}, 500)
            return

        self._send_json({"job_id": job.job_id}, 202)


def run_server(host: str = "127.0.0.1", port: int = 8765,
               output_dir: str = "./output") -> None:
    """Start the Web UI server (blocking)."""
    from ..utils.helpers import ensure_utf8_output
    ensure_utf8_output()  # avoid cp932 print crashes on Windows
    server = ThreadingHTTPServer((host, port), WebUIHandler)
    server.job_manager = JobManager()
    server.output_dir = output_dir
    print(f"Deep Research Tool v{__version__} Web UI: http://{host}:{port}")
    print("Ctrl+C で終了")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_server()
