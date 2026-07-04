"""A dependency-free web server (REST + Server-Sent Events).

Built on the standard library's ``http.server`` so it runs with zero installs
and no native binaries — open it in a browser and go. Live updates use SSE
(server -> browser), which is a perfect fit for streaming a collaboration
transcript; browser -> server actions are ordinary JSON POSTs.
"""

from __future__ import annotations

import json
import os
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

from ..adapters import build_adapter, catalog
from ..config import get_settings
from ..orchestrator import (
    SessionManager,
    start_session,
    strategy_metadata,
)
from ..orchestrator.strategies import get_strategy, load_references

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}

# Shared, process-wide session registry.
MANAGER = SessionManager()


def _ensure_workspace_dir(path: str) -> str:
    """Create the workspace directory if it doesn't exist (no git involved).

    Returns "created" if it was newly made, or "exists" if it was already there.
    """
    existed = os.path.isdir(path)
    os.makedirs(path, exist_ok=True)
    return "exists" if existed else "created"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AgentOrchestrator/0.1"

    # -- low-level helpers -------------------------------------------------

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename: str) -> None:
        safe = os.path.basename(filename) or "index.html"
        path = os.path.join(_STATIC_DIR, safe)
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, status=404)
            return
        ext = os.path.splitext(safe)[1]
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):  # quieter logging
        return

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_static("index.html")
        elif path.startswith("/static/"):
            self._send_static(path[len("/static/"):])
        elif path == "/api/catalog":
            self._send_json({"agents": catalog(), "strategies": strategy_metadata()})
        elif path == "/api/settings":
            self._send_json(self._settings_payload())
        elif path == "/api/sessions":
            self._send_json({"sessions": self._sessions_payload()})
        elif path.startswith("/api/stream/"):
            self._stream(path[len("/api/stream/"):])
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/run":
            self._run()
        elif path == "/api/settings":
            get_settings().apply_overrides(self._read_body())
            self._send_json(self._settings_payload())
        elif path.startswith("/api/stop/"):
            self._stop(path[len("/api/stop/"):])
        elif path.startswith("/api/finish/"):
            self._finish(path[len("/api/finish/"):])
        else:
            self._send_json({"error": "not found"}, status=404)

    # -- endpoints ---------------------------------------------------------

    def _sessions_payload(self) -> list:
        """Newest-first summary of every session this server has run.

        The event bus keeps each session's full history, so the UI can reopen
        any of these (running or finished) and replay the whole transcript.
        """
        out = []
        for s in sorted(MANAGER.all(), key=lambda s: s.created, reverse=True):
            task = s.task if len(s.task) <= 100 else s.task[:100] + "…"
            out.append({
                "id": s.id, "task": task, "strategy": s.strategy,
                "rounds": s.rounds, "status": s.status, "created": s.created,
            })
        return out

    def _settings_payload(self) -> dict:
        """Non-secret view of current settings for the UI (keys never echoed)."""
        s = get_settings()

        def prov(model, base, key, env_var):
            return {
                "model": model,
                "base_url": base,
                "key_set": bool(key),
                "key_from_env": bool(os.environ.get(env_var)),
            }

        proxy_env = any(os.environ.get(v) for v in (
            "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"))
        return {
            "proxy": s.proxy or "",
            "proxy_from_env": proxy_env,
            "providers": {
                "anthropic": prov(s.anthropic_model, s.anthropic_base_url,
                                  s.anthropic_api_key, "ANTHROPIC_API_KEY"),
                "openai": prov(s.openai_model, s.openai_base_url,
                               s.openai_api_key, "OPENAI_API_KEY"),
                "local": {
                    "model": s.local_model, "base_url": s.local_base_url,
                    "key_set": bool(s.local_api_key and s.local_api_key != "local"),
                    "key_from_env": bool(os.environ.get("LOCAL_LLM_API_KEY")),
                    "use_proxy": s.local_use_proxy,
                },
            },
        }

    def _run(self) -> None:
        body = self._read_body()
        task = (body.get("task") or "").strip()
        strategy_name = body.get("strategy") or ""
        try:
            rounds = int(body.get("rounds", 2))  # 0 is meaningful: unlimited
        except (TypeError, ValueError):
            rounds = 2
        roles_spec = body.get("roles") or {}

        if not task:
            self._send_json({"error": "task is required"}, status=400)
            return
        try:
            strategy = get_strategy(strategy_name)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        # rounds == 0 means "no limit" (run until DONE / a human presses Finish)
        # for strategies with a natural end; others get their default back.
        if rounds <= 0:
            rounds = 0 if strategy.supports_unlimited else strategy.default_rounds
        else:
            rounds = min(rounds, 8)

        # Human feedback on a previous session's result: fold it into the task
        # so the team reworks with full knowledge of what was rejected and why.
        feedback = (body.get("feedback") or "").strip()
        parent = MANAGER.get((body.get("parent_session") or "").strip())
        if feedback:
            prev = ""
            if parent and parent.result:
                prev = f"\n\n[PREVIOUS RESULT — the attempt being reworked]\n{parent.result[:4000]}"
            task = (f"{task}\n\n[HUMAN FEEDBACK — the previous attempt was not "
                    f"satisfactory; rework it accordingly]\n{feedback}{prev}")

        # Roles are fixed by the strategy, except for strategies with a dynamic
        # role set (the user-defined "custom" panel, the conductor's team) whose
        # participants come from the request.
        if strategy.custom or strategy.dynamic_roles:
            order = body.get("role_order") or list(roles_spec.keys())
            role_list = [(k, k) for k in order]
            if len(role_list) < 2:
                self._send_json(
                    {"error": "this strategy needs at least 2 participants"}, status=400
                )
                return
        else:
            role_list = strategy.roles

        # Build one adapter per role; capture per-role model + persona overrides.
        agents = {}
        personas = {}
        unavailable = []
        for role_key, _label in role_list:
            spec = roles_spec.get(role_key) or {"id": "mock"}
            adapter = build_adapter(
                {"id": spec.get("id", "mock"), "name": role_key, "model": spec.get("model")}
            )
            sys_override = (spec.get("system") or "").strip()
            if sys_override:
                personas[role_key] = sys_override
            ok, reason = adapter.available()
            if not ok:
                unavailable.append({"role": role_key, "id": spec.get("id", "mock"), "reason": reason})
            agents[role_key] = adapter

        if unavailable:
            self._send_json(
                {"error": "selected agents are unavailable", "details": unavailable},
                status=400,
            )
            return

        session = MANAGER.create(task, strategy_name, rounds, agents)
        session.personas = personas
        session.role_order = [k for k, _ in role_list]
        if feedback and parent:
            session.parent_id = parent.id

        # Orchestrator-provided tools the agents may call (opt-in, allowlisted).
        tools = body.get("tools") or []
        if isinstance(tools, list):
            session.tools = [t for t in tools
                             if t in ("list_files", "read_file", "run", "http_get")]

        # Chain of command for org_team: role -> supervisor role.
        sups = body.get("supervisors") or {}
        if isinstance(sups, dict):
            roles_set = set(session.role_order)
            session.supervisors = {
                str(k): str(v) for k, v in sups.items()
                if str(k) in roles_set and str(v) in roles_set and str(k) != str(v)
            }

        # Optional read-only reference directory (any strategy): load its files
        # into the agents' shared context.
        ref = (body.get("reference_dir") or "").strip()
        if ref:
            ref_real = os.path.realpath(ref)
            if not os.path.isdir(ref_real):
                self._send_json({"error": f"reference directory not found: {ref}"}, status=400)
                return
            session.reference_dir = ref_real
            session.references = load_references(ref_real)

        if strategy_name == "workspace_build":
            # Default to the directory the server was launched from; a request may
            # override it. Edits are confined to this root (see _safe_join).
            ws = (body.get("workspace") or "").strip() or os.getcwd()
            session.workspace = os.path.realpath(ws)
            if body.get("create_dir"):
                session.workspace_created = _ensure_workspace_dir(session.workspace)
            # Optional auto-verification command, run in the workspace each round.
            session.test_command = (body.get("test_command") or "").strip()[:400]
            # CLI backends (Claude Code / Codex) run *inside* the workspace and
            # edit files natively with their own tools; other backends keep the
            # <FILE> protocol.
            for adapter in agents.values():
                if adapter.kind == "cli":
                    adapter.workdir = session.workspace

        start_session(session)
        self._send_json({"session_id": session.id})

    def _stop(self, session_id: str) -> None:
        session = MANAGER.get(session_id)
        if not session:
            self._send_json({"error": "no such session"}, status=404)
            return
        session.stop_requested = True
        self._send_json({"ok": True})

    def _finish(self, session_id: str) -> None:
        """Graceful human-initiated wrap-up: the strategy stops looping and
        produces its final deliverable (unlike stop, which aborts)."""
        session = MANAGER.get(session_id)
        if not session:
            self._send_json({"error": "no such session"}, status=404)
            return
        session.finish_requested = True
        self._send_json({"ok": True})

    def _stream(self, session_id: str) -> None:
        session = MANAGER.get(session_id)
        if not session:
            self._send_json({"error": "no such session"}, status=404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        q = session.bus.subscribe()
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # heartbeat keeps the socket alive
                    self.wfile.flush()
                    continue
                if session.bus.is_closed_sentinel(item):
                    break
                payload = json.dumps(item.to_dict())
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away
        finally:
            session.bus.unsubscribe(q)


def run(host: Optional[str] = None, port: Optional[int] = None,
        open_browser: bool = False) -> None:
    """Start the web server (blocking)."""
    settings = get_settings()
    host = host or settings.host
    port = port or settings.port
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Agent Orchestrator UI running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
