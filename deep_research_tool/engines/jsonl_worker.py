"""JSONL worker subprocess driver (ported from the orchestrator's gptr_engine).

A real engine (GPT Researcher, Open Deep Research) runs in ITS OWN Python
(its own virtualenv, possibly a newer interpreter) as a subprocess:

- the request payload goes to the worker's stdin as one JSON document
- the worker prints JSONL to stdout: ``{"kind": "event", "type", "payload"}``
  lines are forwarded to ``ctx.emit``; one ``{"kind": "result", ...}`` line
  carries the RunResult; ``{"kind": "fatal", "error"}`` aborts
- a cancel request terminates the worker (SIGTERM, then SIGKILL after a
  grace period)
- API keys travel ONLY through the environment and are masked in every
  error message
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, Dict, List, Optional

from .runner_core.engine import CancelledByUser, RunContext
from .runner_core.models import RunResult

_STDOUT_LIMIT = 2 ** 25  # 32 MiB per line: long reports stay on one JSONL line


class EngineConfigError(ValueError):
    """The request cannot be run with the given configuration (fail fast,
    before any subprocess starts)."""


def parse_worker_line(line: str) -> Optional[Dict[str, Any]]:
    """worker stdout の 1 行を JSONL としてパースする。プロトコル外の行は None。"""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "kind" not in obj:
        return None
    return obj


def redact_secrets(text: str, secrets: List[Optional[str]]) -> str:
    """エラーメッセージ等に混入した API キーをマスクする。"""
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "***")
    return text


def result_from_worker_payload(payload: Dict[str, Any]) -> RunResult:
    """worker の {"kind":"result", ...} 行から RunResult を構築する。"""
    data = {k: v for k, v in payload.items() if k != "kind"}
    return RunResult.model_validate(data)


async def run_jsonl_worker(
    ctx: RunContext,
    *,
    argv: List[str],
    env: Dict[str, str],
    payload: Dict[str, Any],
    cwd: str,
    secrets: List[Optional[str]],
    term_grace_seconds: float = 5.0,
) -> Dict[str, Any]:
    """JSONL プロトコルの worker サブプロセスを実行し、result payload を返す。

    - event は ctx.emit に転送
    - キャンセル要求で SIGTERM → SIGKILL
    - fatal / 異常終了は RuntimeError（API キーはマスク）
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
        limit=_STDOUT_LIMIT,
    )
    stderr_tail: deque = deque(maxlen=100)

    async def _drain_stderr() -> None:
        assert proc.stderr is not None
        while True:
            try:
                chunk = await proc.stderr.readline()
            except (ValueError, asyncio.LimitOverrunError):
                continue  # 長すぎる stderr 行は捨てる
            if not chunk:
                return
            stderr_tail.append(chunk.decode("utf-8", errors="replace").rstrip())

    stderr_task = asyncio.create_task(_drain_stderr())
    result_payload: Optional[Dict[str, Any]] = None
    fatal_error: Optional[str] = None
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        while True:
            if ctx.cancel_requested:
                raise CancelledByUser()
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if not raw:
                break  # EOF
            msg = parse_worker_line(raw.decode("utf-8", errors="replace"))
            if msg is None:
                continue
            kind = msg.get("kind")
            if kind == "event":
                event_payload = msg.get("payload")
                ctx.emit(
                    str(msg.get("type") or "log"),
                    event_payload if isinstance(event_payload, dict) else {},
                )
            elif kind == "result":
                result_payload = msg
                break
            elif kind == "fatal":
                fatal_error = str(msg.get("error") or "unknown worker error")
                break
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=term_grace_seconds)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

    if fatal_error is not None:
        raise RuntimeError(redact_secrets(f"engine worker 失敗: {fatal_error}", secrets))
    if result_payload is None:
        tail = redact_secrets("\n".join(list(stderr_tail)[-20:]), secrets)
        raise RuntimeError(
            f"engine worker が結果を返さずに終了しました (exit={proc.returncode})。"
            f" stderr 末尾:\n{tail}"
        )
    return result_payload
