"""
Local disk cache for fetched pages and LLM extraction results.

Purpose: cut network waits and LLM calls when the same page / the same
extraction is needed again (another section, a re-run, a resumed run).

Keys (deterministic, content-addressed):
- pages:       normalized URL  -> {text, title, fetched_at, etag,
               last_modified, content_hash}
- extractions: sha256(content_hash | section_context | research_query |
               model | prompt_version) -> ExtractedContent dict

Freshness: a cached page older than ``max_age_hours`` is stale and is
re-fetched; callers that DEMAND fresh information (recency markers in
the topic, or the user's "更新確認" choice) pass ``refresh=True`` and the
cache is bypassed for reads (writes still happen). Both hit counters are
exposed so the run report can state how much work was reused.

Everything is plain JSON under ``<cache_dir>/pages`` and
``<cache_dir>/extractions``; writes are atomic (tmp + os.replace) and
any read error is treated as a miss. No network access here.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

EXTRACTION_PROMPT_VERSION = "extract-v1"


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:32]


class FetchCache:
    """Disk-backed page + extraction cache (thread-safe, atomic writes)."""

    def __init__(self, cache_dir: Path, max_age_hours: float = 72.0,
                 enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max(0.0, float(max_age_hours)) * 3600.0
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self.page_hits = 0
        self.page_misses = 0
        self.extract_hits = 0
        self.extract_misses = 0
        self.refreshed = 0          # stale/forced re-fetches
        if self.enabled:
            try:
                (self.cache_dir / "pages").mkdir(parents=True, exist_ok=True)
                (self.cache_dir / "extractions").mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False

    # ------------------------------------------------------------------
    # low-level
    # ------------------------------------------------------------------

    @staticmethod
    def _key(*parts: str) -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(str(part).encode("utf-8", "replace"))
            h.update(b"\x1f")
        return h.hexdigest()[:40]

    def _path(self, kind: str, key: str) -> Path:
        return self.cache_dir / kind / f"{key}.json"

    def _read(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(self, path: Path, data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            pass                        # a cache write failure is never fatal

    def _fresh(self, entry: Dict[str, Any]) -> bool:
        if self.max_age_seconds <= 0:
            return True
        return (time.time() - float(entry.get("fetched_at", 0))) <= \
            self.max_age_seconds

    # ------------------------------------------------------------------
    # pages (keyed by normalized URL)
    # ------------------------------------------------------------------

    def get_page(self, norm_url: str, refresh: bool = False
                 ) -> Optional[Dict[str, Any]]:
        """Cached page dict or None (miss / stale / refresh demanded)."""
        if not self.enabled or not norm_url:
            return None
        entry = self._read(self._path("pages", self._key("page", norm_url)))
        with self._lock:
            if entry is None:
                self.page_misses += 1
                return None
            if refresh or not self._fresh(entry):
                self.refreshed += 1
                self.page_misses += 1
                return None
            self.page_hits += 1
        return entry

    def put_page(self, norm_url: str, text: str, title: str = "",
                 etag: str = "", last_modified: str = "",
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entry = {
            "url": norm_url, "title": title or "", "text": text or "",
            "content_hash": content_hash(text),
            "fetched_at": time.time(), "etag": etag or "",
            "last_modified": last_modified or "",
        }
        if extra:
            entry["extra"] = extra
        self._write(self._path("pages", self._key("page", norm_url)), entry)
        return entry

    # ------------------------------------------------------------------
    # extractions (content-addressed)
    # ------------------------------------------------------------------

    def extraction_key(self, text: str, section_context: str,
                       research_query: str, model: str,
                       prompt_version: str = EXTRACTION_PROMPT_VERSION) -> str:
        return self._key("extract", content_hash(text), section_context,
                         research_query, model, prompt_version)

    def get_extraction(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        entry = self._read(self._path("extractions", key))
        with self._lock:
            if entry is None:
                self.extract_misses += 1
            else:
                self.extract_hits += 1
        return entry

    def put_extraction(self, key: str, extracted: Dict[str, Any]) -> None:
        self._write(self._path("extractions", key),
                    {"saved_at": time.time(), "extracted": extracted})

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "page_hits": self.page_hits, "page_misses": self.page_misses,
                "extract_hits": self.extract_hits,
                "extract_misses": self.extract_misses,
                "refreshed": self.refreshed,
            }
