"""Pages named by the user (使用する資料: 指定 URL) as research documents.

The URL source modes do not add a research path of their own: the pages
are fetched through the search client (HTML, PDF and Office files alike,
as any web page is) and registered in the same local document store the
手元の資料 modes use. "指定 URL のみ" is then the local-only path and
"指定 URL を重視＋Web" the hybrid path with the URL-origin evidence boosted.

``fetch_url_documents`` returns the documents plus a report of what was
and was not read, so the result can say "3 URLs, 2 fetched" instead of
silently researching from fewer pages than the user named.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from ..utils.concurrency import ContextThreadPoolExecutor, effective_workers

MIN_CHARS = 200          # shorter pages are reported as unreadable (same as 手元の資料)
MAX_URLS = 50            # named URLs per run
CRAWL_PAGES_CAP = 100    # hard cap on pages read through same-site links

_SKIP_LINK_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".css", ".js",
                       ".zip", ".mp4", ".mp3", ".ico", ".woff", ".woff2")


def normalize_urls(urls) -> List[str]:
    """Trimmed, de-duplicated, http(s)-only, order kept; a string is split
    on newlines / commas / whitespace."""
    if urls is None:
        return []
    if isinstance(urls, str):
        parts = [p for chunk in urls.replace(",", "\n").split("\n")
                 for p in chunk.split()]
    else:
        parts = [str(u) for u in urls if u not in (None, "")]
    out: List[str] = []
    seen = set()
    for raw in parts:
        u = raw.strip().strip("<>\"'")
        if not u:
            continue
        if "://" not in u:
            u = "https://" + u
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.netloc:
            continue
        key = u.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out[:MAX_URLS]


def normalize_domains(domains) -> List[str]:
    """Hostnames only (a URL or "site:example.com" is reduced to its host)."""
    if domains is None:
        return []
    if isinstance(domains, str):
        parts = [p for chunk in domains.replace(",", "\n").split("\n") for p in chunk.split()]
    else:
        parts = [str(d) for d in domains if d not in (None, "")]
    out: List[str] = []
    for raw in parts:
        d = raw.strip().lower()
        if d.startswith("site:"):
            d = d[5:]
        if "://" in d:
            d = urlparse(d).netloc
        d = d.split("/")[0].strip(" .")
        if d and "." in d and d not in out:
            out.append(d)
    return out[:10]


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _same_site(a: str, b: str) -> bool:
    ha, hb = _host(a), _host(b)
    if not ha or not hb:
        return False
    strip = lambda h: h[4:] if h.startswith("www.") else h  # noqa: E731
    return strip(ha) == strip(hb)


def fetch_one(url: str, search_client, min_chars: int = MIN_CHARS) -> Dict[str, Any]:
    """One page through the search client: {url, ok, title, chars, kind,
    error, content, links, metadata}. Never raises."""
    out: Dict[str, Any] = {"url": url, "ok": False, "title": "", "chars": 0,
                           "kind": "page", "error": None, "content": "", "links": [],
                           "metadata": {}}
    low = url.lower().split("?")[0]
    for ext, kind in ((".pdf", "pdf"), (".docx", "docx"), (".xlsx", "xlsx"),
                      (".pptx", "pptx"), (".csv", "csv"), (".md", "markdown"), (".txt", "text")):
        if low.endswith(ext):
            out["kind"] = kind
    if search_client is None or not hasattr(search_client, "get_page_content"):
        out["error"] = "ページを取得するクライアントがありません"
        return out
    try:
        page = search_client.get_page_content(url)
    except Exception as e:  # noqa: BLE001 - a bad URL must not stop the run
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    meta = dict(getattr(page, "metadata", {}) or {})
    text = (getattr(page, "text_content", "") or "").strip()
    out["title"] = (getattr(page, "title", "") or "").strip() or url
    out["metadata"] = meta
    out["links"] = list(getattr(page, "links", []) or [])
    if meta.get("error"):
        out["error"] = str(meta["error"])[:300]
        return out
    if len(text) < min_chars:
        out["error"] = f"本文が短すぎます（{len(text)} 文字）"
        out["chars"] = len(text)
        return out
    out["ok"] = True
    out["chars"] = len(text)
    out["content"] = text
    return out


def _fetch_many(urls: List[str], search_client, *, parallel_max_workers: int,
                cancel_check: Optional[Callable[[], None]]) -> List[Dict[str, Any]]:
    def _one(u):
        if cancel_check is not None:
            cancel_check()
        return fetch_one(u, search_client)
    serial = bool(getattr(search_client, "requires_serial_access", False))
    workers = 1 if serial else effective_workers(parallel_max_workers, 4, len(urls))
    if workers > 1 and len(urls) > 1:
        with ContextThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_one, urls))
    return [_one(u) for u in urls]


def fetch_url_documents(urls, search_client, *, crawl_depth: int = 0,
                        max_pages: int = 30, parallel_max_workers: int = 4,
                        cancel_check: Optional[Callable[[], None]] = None,
                        progress: Optional[Callable[[str], None]] = None
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The named URLs (and, with crawl_depth=1, pages they link to on the
    same site, up to max_pages in total) as documents for the local store.

    Returns (documents, report). documents: {path, title, content, metadata}
    with metadata.source == "url". report: {requested, fetched, failed:
    [{url, error}], crawled, pages, urls: [{url, ok, title, chars, kind,
    error, depth}]}."""
    seeds = normalize_urls(urls)
    max_pages = max(1, min(int(max_pages or 30), CRAWL_PAGES_CAP))
    report: Dict[str, Any] = {"requested": len(seeds), "fetched": 0, "failed": [],
                              "crawled": 0, "pages": 0, "urls": []}
    docs: List[Dict[str, Any]] = []
    if not seeds:
        return docs, report
    if progress:
        progress(f"指定 URL {len(seeds)} 件を取得しています")
    results = _fetch_many(seeds, search_client, parallel_max_workers=parallel_max_workers,
                          cancel_check=cancel_check)
    now = datetime.now().isoformat()
    seen = {urldefrag(u)[0].rstrip("/").lower() for u in seeds}

    def _add(res: Dict[str, Any], depth: int, seed: str) -> None:
        entry = {k: res[k] for k in ("url", "ok", "title", "chars", "kind", "error")}
        entry["depth"] = depth
        report["urls"].append(entry)
        if not res["ok"]:
            report["failed"].append({"url": res["url"], "error": res["error"]})
            return
        docs.append({
            "path": res["url"], "title": res["title"], "content": res["content"],
            "metadata": {"source": "url", "source_url": res["url"], "seed_url": seed,
                         "depth": depth, "fetched_at": now, "kind": res["kind"],
                         **{k: v for k, v in res["metadata"].items()
                            if k in ("published_date", "publisher", "source_type")}},
        })
        if depth == 0:
            report["fetched"] += 1
        else:
            report["crawled"] += 1

    for seed, res in zip(seeds, results):
        _add(res, 0, seed)

    if int(crawl_depth or 0) >= 1 and len(docs) < max_pages:
        # one level of same-site links from each readable seed, breadth
        # first, until max_pages pages are read in total
        candidates: List[Tuple[str, str]] = []
        for seed, res in zip(seeds, results):
            if not res["ok"]:
                continue
            for link in res["links"]:
                href = link.get("url") if isinstance(link, dict) else str(link)
                if not href:
                    continue
                href = urldefrag(urljoin(seed, href))[0]
                key = href.rstrip("/").lower()
                if (not href.startswith(("http://", "https://")) or key in seen
                        or not _same_site(seed, href)
                        or key.split("?")[0].endswith(_SKIP_LINK_SUFFIXES)):
                    continue
                seen.add(key)
                candidates.append((seed, href))
        room = max_pages - len(docs)
        candidates = candidates[:max(0, room)]
        if candidates:
            if progress:
                progress(f"同一サイト内のリンク先 {len(candidates)} 件を取得しています")
            crawled = _fetch_many([h for _, h in candidates], search_client,
                                  parallel_max_workers=parallel_max_workers,
                                  cancel_check=cancel_check)
            for (seed, _href), res in zip(candidates, crawled):
                _add(res, 1, seed)
    report["pages"] = len(docs)
    return docs, report


def precheck_urls(urls, search_client) -> List[Dict[str, Any]]:
    """Reachability / size / title of each named URL, for the form's
    「取得を確認」(no crawling, nothing kept)."""
    out = []
    for res in _fetch_many(normalize_urls(urls), search_client,
                           parallel_max_workers=4, cancel_check=None):
        out.append({k: res[k] for k in ("url", "ok", "title", "chars", "kind", "error")})
    return out
