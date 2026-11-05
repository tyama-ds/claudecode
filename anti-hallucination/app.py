"""CLI and loopback-only browser UI for the numerical fact-checking pilot."""
from __future__ import annotations

import argparse
import io
import json
import re
import secrets
import socket
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
MAX_BODY = 20 * 1024 * 1024
SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


class LocalHTTPServer(ThreadingHTTPServer):
    # Windows otherwise permits two HTTP servers to share a listening port.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSONのキーが重複しています: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"JSONに有限でない数値は使えません: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8-sig"))


def preflight(reports, gold, config):
    from factcheck.dataset import validate_dataset
    from factcheck.providers import validate_models
    dataset = validate_dataset(reports, gold)
    cfg = validate_models(config, check_credentials=True)
    n = len(dataset["items"])
    m = len(cfg["models"])
    repeats = cfg.get("repeats", 1)
    fills = n * m * repeats
    pool_max = n + sum(len(x.get("candidates", [])) for x in dataset["items"]) + fills
    return {
        "reports": len(dataset["reports"]), "items": n, "models": m,
        "repeats": repeats, "fill_calls": fills,
        "check_calls_max": pool_max * m * repeats,
        "total_calls_max": fills + pool_max * m * repeats,
        "demo": all(x["provider"] == "demo" for x in cfg["models"]),
        "destinations": [
            {"id": x["id"], "provider": x["provider"], "model": x["model"],
             "base_url": x.get("base_url", "")}
            for x in cfg["models"]
        ],
    }


class PilotState:
    def __init__(self, runs_dir):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.jobs = {}

    def list_runs(self):
        result = []
        for path in sorted(self.runs_dir.glob("*/summary.json"), reverse=True):
            try:
                summary = read_json(path)
                result.append({"id": path.parent.name, **{
                    k: summary.get(k) for k in ("status", "demo", "run_mode", "created_at", "counts")}})
            except (OSError, ValueError):
                continue
        return result

    def start(self, reports, gold, config):
        preflight(reports, gold, config)
        with self.lock:
            if any(x["status"] == "running" for x in self.jobs.values()):
                raise ValueError("実験を実行中です。完了または停止してから開始してください。")
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            job = {"id": run_id, "status": "running", "progress": {
                "stage": "prepare", "completed": 0, "total": 0, "message": "準備しています"},
                "result": None, "error": None, "stop": threading.Event()}
            self.jobs[run_id] = job

        def update(progress):
            with self.lock:
                job["progress"] = progress

        def execute():
            from factcheck.experiment import run_experiment
            try:
                result = run_experiment(reports, gold, config, self.runs_dir / run_id,
                                        progress=update, cancel=job["stop"].is_set)
                with self.lock:
                    job["result"] = result
                    job["status"] = result.get("status", "completed")
            except Exception as exc:
                with self.lock:
                    job["status"] = "failed"
                    job["error"] = f"{type(exc).__name__}: {exc}"
        threading.Thread(target=execute, daemon=True).start()
        return run_id

    def snapshot(self, run_id):
        with self.lock:
            job = self.jobs.get(run_id)
            if job is None:
                return None
            return {k: v for k, v in job.items() if k != "stop"}

    def active_job(self):
        with self.lock:
            for job in self.jobs.values():
                if job["status"] == "running":
                    return {k: v for k, v in job.items() if k != "stop"}
        return None


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FactCheckPilot/0.1"

        def log_message(self, fmt, *args):
            # Do not print request bodies, API responses or credentials.
            if args and str(args[0]).startswith("GET /api/jobs/"):
                return
            super().log_message(fmt, *args)

        def send(self, status, body, content_type="application/json; charset=utf-8", filename=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def trusted_host(self):
            host = self.headers.get("Host", "")
            port = self.server.server_address[1]
            return host in (f"127.0.0.1:{port}", f"localhost:{port}")

        def valid_write(self):
            if not self.trusted_host():
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                return False
            return secrets.compare_digest(self.headers.get("X-Pilot-Token", ""), state.token)

        def do_GET(self):
            if not self.trusted_host():
                self.send(403, {"error": "Localhost access only"})
                return
            path = urlsplit(self.path).path
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in static:
                file, mime = static[path]
                self.send(200, (ROOT / "web" / file).read_bytes(), mime)
                return
            if path == "/api/bootstrap":
                self.send(200, {"token": state.token, "samples": {
                    "reports": read_json(ROOT / "samples/reports.json"),
                    "gold": read_json(ROOT / "samples/gold.json"),
                    "config": read_json(ROOT / "samples/models.demo.json"),
                    "live_config": read_json(ROOT / "samples/models.live.example.json")},
                    "runs": state.list_runs(), "active_job": state.active_job()})
                return
            if path == "/api/runs":
                self.send(200, state.list_runs())
                return
            match = re.fullmatch(r"/api/jobs/([a-zA-Z0-9_-]+)", path)
            if match:
                job = state.snapshot(match[1])
                self.send(200 if job else 404, job or {"error": "実行が見つかりません"})
                return
            match = re.fullmatch(r"/api/runs/([a-zA-Z0-9_-]+)(/download)?", path)
            if match:
                folder = state.runs_dir / match[1]
                if not folder.is_dir():
                    self.send(404, {"error": "結果が見つかりません"})
                    return
                if match[2]:
                    archive = io.BytesIO()
                    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
                        for file in sorted(folder.iterdir()):
                            if file.is_file() and not file.is_symlink() and file.suffix in (".json", ".jsonl", ".csv", ".txt", ".html"):
                                out.write(file, file.name)
                    self.send(200, archive.getvalue(), "application/zip", match[1] + ".zip")
                elif (folder / "summary.json").is_file():
                    self.send(200, read_json(folder / "summary.json"))
                else:
                    self.send(404, {"error": "集計はまだ保存されていません"})
                return
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_write():
                # Drain a bounded body before closing to avoid a TCP reset
                # discarding the 403 response on Windows.
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if 0 < length <= MAX_BODY:
                        self.connection.settimeout(2)
                        self.rfile.read(length)
                except (ValueError, OSError):
                    pass
                self.send(403, {"error": "ページを再読み込みしてください。外部サイトからの操作はできません。"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("入力サイズは20 MB以下にしてください。")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON形式で送信してください。")
                body = parse_json(self.rfile.read(length).decode("utf-8-sig"))
                path = urlsplit(self.path).path
                if path in ("/api/validate", "/api/run"):
                    reports, gold, config = body["reports"], body["gold"], body["config"]
                    if path == "/api/validate":
                        self.send(200, preflight(reports, gold, config))
                    else:
                        run_id = state.start(reports, gold, config)
                        self.send(202, {"id": run_id})
                    return
                match = re.fullmatch(r"/api/jobs/([a-zA-Z0-9_-]+)/cancel", path)
                if match:
                    with state.lock:
                        job = state.jobs.get(match[1])
                        if not job:
                            raise ValueError("実行が見つかりません")
                        job["stop"].set()
                    self.send(200, {"message": "現在のAPI呼び出しが終わり次第停止します。"})
                    return
                self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                self.send(500, {"error": "処理できませんでした。入力形式と保存先を確認してください。"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="数値穴埋め・ファクトチェック比較の予備実験")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="ブラウザー画面を起動")
    serve.add_argument("--port", type=int, default=8788)
    serve.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    run = sub.add_parser("run", help="JSON入力から実験を実行")
    run.add_argument("--reports", type=Path, required=True)
    run.add_argument("--gold", type=Path, required=True)
    run.add_argument("--models", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.command in (None, "serve"):
        port = getattr(args, "port", 8788)
        state = PilotState(getattr(args, "runs_dir", ROOT / "runs"))
        try:
            server = LocalHTTPServer(("127.0.0.1", port), make_handler(state))
        except OSError as exc:
            print(f"ポート {port} で起動できません。--port 8789 などで別のポートを指定してください。\n{exc}", file=sys.stderr)
            sys.exit(2)
        print(f"Fact-check pilot: http://127.0.0.1:{server.server_address[1]}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    try:
        reports, gold, config = read_json(args.reports), read_json(args.gold), read_json(args.models)
        plan = preflight(reports, gold, config)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        if args.validate_only:
            return
        from factcheck.experiment import run_experiment
        output = args.output or ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        result = run_experiment(reports, gold, config, output,
                                progress=lambda p: print(p.get("message", p.get("stage", "")), flush=True))
        print(f"Saved: {output.resolve()}")
        print(f"Status: {result['status']}")
        if result["status"] != "completed" or result.get("errors"):
            sys.exit(1)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
