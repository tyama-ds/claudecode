"""
Audit log — append-only JSONL trail of the adaptive research run.

Records WHAT the pipeline decided and measured (rounds, transitions,
queries, verdict counts) so a run can be reconstructed afterwards.

Security rules:
- values under keys matching the secret pattern are masked BEFORE
  serialization (api_key, token, authorization, password, secret);
- long strings are truncated — prompts and page bodies never land in
  the log;
- the log is a LOCAL file; nothing is transmitted anywhere.
"""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|authorization|credential)",
    re.IGNORECASE)
_MAX_STR = 300


def _mask(value: Any, key: str = "") -> Any:
    if key and _SECRET_KEY_RE.search(key):
        return "***"
    if isinstance(value, dict):
        return {str(k): _mask(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mask(v) for v in value]
    if isinstance(value, str) and len(value) > _MAX_STR:
        return value[:_MAX_STR] + f"…(+{len(value) - _MAX_STR} chars)"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:_MAX_STR]


class AuditLog:
    """Thread-safe JSONL audit writer (no-op when disabled)."""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True,
                 session_id: str = ""):
        self.enabled = bool(enabled) and path is not None
        self.path = Path(path) if path is not None else None
        self.session_id = session_id
        self._lock = threading.Lock()
        self._count = 0
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.enabled = False

    def event(self, event_type: str, **fields) -> None:
        """Append one audit record; failures never break the pipeline."""
        if not self.enabled:
            return
        record = {
            "ts": round(time.time(), 3),
            "session": self.session_id,
            "event": str(event_type),
        }
        record.update({str(k): _mask(v, str(k))
                       for k, v in fields.items()})
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self._count += 1
            except Exception:
                self.enabled = False    # disk problems disable, not crash

    @property
    def count(self) -> int:
        with self._lock:
            return self._count
