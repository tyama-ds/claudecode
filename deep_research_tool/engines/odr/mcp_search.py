"""The tool's own search, served to Open Deep Research over MCP.

open-deep-research 0.0.16 knows three hosted search APIs (Tavily, OpenAI,
Anthropic) or ``none``; it has no DuckDuckGo / SearXNG retriever. What it
does have is ``mcp_config``: a streamable-HTTP MCP server whose tools the
research agents may call. So each run starts THIS server on 127.0.0.1 (a
free port, a daemon thread inside the tool's process) and the engine is
told ``search_api="none"`` plus ``mcp_config={"url": ..., "tools": [...]}``.

The server speaks the stateless subset of MCP's streamable HTTP transport
(JSON-RPC 2.0 over POST ``/mcp``; ``initialize``, ``notifications/
initialized``, ``ping``, ``tools/list``, ``tools/call``) with the standard
library only — no ``mcp`` dependency on the tool's side — and verified
against the real ``mcp`` 1.28 client (see tests). The tools:

- ``web_search(queries, max_results?)``  the search the tool is configured
  for (DuckDuckGo / browser) or SearXNG, with a short page excerpt per
  result so the agent writes from page text, not only from snippets
- ``fetch_page(url)``  one page's text (指定 URL, or a result to read fully)

Every search and fetch is logged (counts, URLs → titles) for the engine's
result and reported to the run as ``search`` / ``source_found`` events.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Sequence

from ... import __version__

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL = "2025-03-26"
SERVER_NAME = "deep-research-tool-search"

TOOL_SEARCH = "web_search"
TOOL_FETCH = "fetch_page"
ALL_TOOLS = (TOOL_SEARCH, TOOL_FETCH)

MAX_QUERIES_PER_CALL = 5
DEFAULT_MAX_RESULTS = 8
EXCERPT_CHARS = 1500          # page excerpt per search result
FETCH_CHARS = 8000            # fetch_page text limit
SNIPPET_CHARS = 400
# the MCP client (langchain-mcp-adapters → mcp) waits 30 s for a tools/call
# response by default; every call stays well under that
SEARCH_BUDGET_SECONDS = 22.0
EXCERPT_FETCH_SECONDS = 10.0
EXCERPT_FETCH_TOP = 5          # results whose page is read for an excerpt
MAX_BODY_BYTES = 4 * 1024 * 1024


# ----------------------------------------------------------------------
# search backends
# ----------------------------------------------------------------------

class SearchBackend:
    """What the tools call. ``search`` returns dicts {title, url, snippet};
    ``fetch`` returns (title, text) or raises."""

    name = "backend"

    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def fetch(self, url: str) -> tuple:
        raise NotImplementedError


class ClientSearchBackend(SearchBackend):
    """The tool's own search client (DuckDuckGo / Selenium browser)."""

    def __init__(self, search_client, name: str = "duckduckgo"):
        self.client = search_client
        self.name = name

    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        results = self.client.search(query, max_results=max_results) or []
        out = []
        for r in results:
            url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else None)
            if not url:
                continue
            title = getattr(r, "title", None) or (r.get("title") if isinstance(r, dict) else None)
            snippet = getattr(r, "snippet", None) or (r.get("snippet") if isinstance(r, dict) else None)
            out.append({"title": title or "", "url": str(url), "snippet": snippet or ""})
            if len(out) >= max_results:
                break
        return out

    def fetch(self, url: str) -> tuple:
        page = self.client.get_page_content(url)
        if page is None:
            raise RuntimeError("no content")
        text = getattr(page, "text_content", None) or getattr(page, "content", None) or ""
        title = getattr(page, "title", None) or ""
        if not str(text).strip():
            raise RuntimeError("empty page")
        return title, str(text)


class SearxSearchBackend(SearchBackend):
    """SearXNG's JSON API, called directly (an internal service: no proxy).
    Pages are read with the tool's search client when one is given."""

    name = "searxng"

    def __init__(self, endpoint: str, *, timeout: float = 20.0, search_client=None):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout or 20.0)
        self.client = search_client

    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        import requests
        session = requests.Session()
        session.trust_env = False
        resp = session.get(f"{self.endpoint}/search", params={"q": query, "format": "json"},
                           headers={"Accept": "application/json"}, timeout=self.timeout)
        resp.raise_for_status()
        return format_searx_items(resp.json(), max_results)

    def fetch(self, url: str) -> tuple:
        if self.client is not None:
            return ClientSearchBackend(self.client).fetch(url)
        import requests
        resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return "", resp.text


def format_searx_items(payload: Any, max_results: int) -> List[Dict[str, Any]]:
    """SearXNG ``/search?format=json`` → [{title, url, snippet}]."""
    out: List[Dict[str, Any]] = []
    results = payload.get("results") if isinstance(payload, dict) else None
    for item in results or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("href")
        if not isinstance(url, str) or not url:
            continue
        title = item.get("title") if isinstance(item.get("title"), str) else ""
        snippet = item.get("content") or item.get("snippet") or ""
        out.append({"title": title, "url": url,
                    "snippet": " ".join(str(snippet).split())})
        if len(out) >= max(1, int(max_results)):
            break
    return out


# ----------------------------------------------------------------------
# the tools
# ----------------------------------------------------------------------

def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


class SearchTools:
    """Tool implementations + the log the engine reads afterwards."""

    def __init__(self, backend: SearchBackend, *, max_results: int = DEFAULT_MAX_RESULTS,
                 excerpt_chars: int = EXCERPT_CHARS, fetch_chars: int = FETCH_CHARS,
                 excerpt_top: int = EXCERPT_FETCH_TOP,
                 on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 fetch_workers: int = 4):
        self.backend = backend
        self.max_results = max(1, int(max_results or DEFAULT_MAX_RESULTS))
        self.excerpt_chars = int(excerpt_chars)
        self.fetch_chars = int(fetch_chars)
        self.excerpt_top = max(0, int(excerpt_top))
        self.on_event = on_event
        self.fetch_workers = max(1, int(fetch_workers))
        self.lock = threading.Lock()
        self.log: Dict[str, Any] = {"searches": 0, "search_errors": 0, "results": 0,
                                    "fetches": 0, "fetch_errors": 0, "urls": {},
                                    "queries": []}

    # --- events ---
    def _emit(self, type_: str, payload: Dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(type_, payload)
        except Exception:
            pass

    def _note_url(self, url: str, title: str) -> None:
        with self.lock:
            if url not in self.log["urls"]:
                self.log["urls"][url] = title or ""
                self._emit("source_found", {"url": url, "title": title or None})

    # --- web_search ---
    def web_search(self, queries: Any, max_results: Any = None) -> str:
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, (list, tuple)):
            return "(invalid queries: expected a list of strings)"
        try:
            limit = int(max_results) if max_results is not None else self.max_results
        except (TypeError, ValueError):
            limit = self.max_results
        limit = max(1, min(limit, 20))
        started = time.monotonic()
        blocks: List[str] = []
        for query in list(queries)[:MAX_QUERIES_PER_CALL]:
            if not isinstance(query, str) or not query.strip():
                continue
            query = query.strip()
            if time.monotonic() - started > SEARCH_BUDGET_SECONDS:
                blocks.append(f'Search results for "{query}":\n\n(skipped: time budget exhausted)')
                continue
            blocks.append(self._search_one(query, limit, started))
        if not blocks:
            return "(no valid queries)"
        return "\n\n---\n\n".join(blocks)

    def _search_one(self, query: str, limit: int, started: float) -> str:
        with self.lock:
            self.log["searches"] += 1
            self.log["queries"].append(query)
        self._emit("search", {"query": query})
        try:
            items = self.backend.search(query, limit)
        except Exception as e:  # noqa: BLE001 - one failed query is reported, not fatal
            with self.lock:
                self.log["search_errors"] += 1
            return f'Search results for "{query}":\n\n(search failed: {type(e).__name__}: {e})'
        with self.lock:
            self.log["results"] += len(items)
        for it in items:
            self._note_url(it["url"], it.get("title") or "")
        excerpts = self._excerpts([it["url"] for it in items[:self.excerpt_top]],
                                  budget=max(1.0, SEARCH_BUDGET_SECONDS - (time.monotonic() - started)))
        lines = [f'Search results for "{query}":']
        for i, it in enumerate(items, 1):
            title = it.get("title") or "(no title)"
            entry = f"{i}. {title}\n   URL: {it['url']}"
            if it.get("snippet"):
                entry += f"\n   {_clip(it['snippet'], SNIPPET_CHARS)}"
            ex = excerpts.get(it["url"])
            if ex:
                entry += f"\n   Page excerpt: {ex}"
            lines.append(entry)
        if len(items) == 0:
            lines.append("(no results)")
        return "\n\n".join(lines)

    def _excerpts(self, urls: Sequence[str], budget: float) -> Dict[str, str]:
        """Page excerpts for the top results, fetched in parallel within
        the time budget; a page that fails or is slow simply has none."""
        if not urls or self.excerpt_top <= 0 or self.excerpt_chars <= 0:
            return {}
        budget = min(float(budget), EXCERPT_FETCH_SECONDS)
        out: Dict[str, str] = {}
        pool = ThreadPoolExecutor(max_workers=min(self.fetch_workers, len(urls)))
        futures = {pool.submit(self._fetch_quiet, u): u for u in urls}
        try:
            done, _ = wait(list(futures), timeout=budget)
            for fut in done:
                res = fut.result()
                if res:
                    out[futures[fut]] = _clip(res[1], self.excerpt_chars)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return out

    def _fetch_quiet(self, url: str):
        try:
            title, text = self.backend.fetch(url)
        except Exception:  # noqa: BLE001
            with self.lock:
                self.log["fetch_errors"] += 1
            return None
        with self.lock:
            self.log["fetches"] += 1
        if title:
            self._note_url(url, title)
        return title, text

    # --- fetch_page ---
    def fetch_page(self, url: Any) -> str:
        url = str(url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return "(invalid url: must start with http:// or https://)"
        res = self._fetch_quiet(url)
        if res is None:
            return f"(could not fetch {url})"
        title, text = res
        self._note_url(url, title)
        head = f"{title}\nURL: {url}\n\n" if title else f"URL: {url}\n\n"
        return head + _clip(text, self.fetch_chars)


TOOL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    TOOL_SEARCH: {
        "name": TOOL_SEARCH,
        "description": (
            "Search the web with the research tool's configured search engine. "
            "Returns, per query, numbered results with title, URL, snippet and a "
            "short excerpt of the page text. Cite the URLs you use."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"},
                            "description": "One to five search queries (keywords or natural language, "
                                           "in the language of the sources you expect)"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20,
                                "description": "Results per query (default 8)"},
            },
            "required": ["queries"],
        },
    },
    TOOL_FETCH: {
        "name": TOOL_FETCH,
        "description": ("Read one web page (HTML or PDF) and return its text, truncated. "
                        "Use it for the starting URLs you were given and to read a search "
                        "result in full."),
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL"}},
            "required": ["url"],
        },
    },
}


# ----------------------------------------------------------------------
# the server (stateless streamable HTTP, JSON-RPC 2.0)
# ----------------------------------------------------------------------

def _rpc_error(id_: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _rpc_result(id_: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


class McpSearchServer:
    """Serve ``SearchTools`` on 127.0.0.1:<free port> in a daemon thread.
    ``tools`` limits what ``tools/list`` offers (指定 URL のみ → fetch_page
    only)."""

    def __init__(self, tools: SearchTools, *, host: str = "127.0.0.1", port: int = 0,
                 enabled_tools: Sequence[str] = ALL_TOOLS):
        self.tools = tools
        self.host = host
        self._port = int(port)
        self.enabled_tools = [t for t in enabled_tools if t in TOOL_DEFINITIONS] or list(ALL_TOOLS)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.requests_seen: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        if self._httpd is not None:
            return self._httpd.server_address[1]
        return self._port

    @property
    def url(self) -> str:
        """The base URL the engine is given (it appends ``/mcp``)."""
        return f"http://{self.host}:{self.port}"

    @property
    def mcp_url(self) -> str:
        return self.url + "/mcp"

    def start(self) -> "McpSearchServer":
        if self._httpd is not None:
            return self
        server = self
        handler = type("McpHandler", (_Handler,), {"server_ref": server})
        self._httpd = ThreadingHTTPServer((self.host, self._port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="drt-mcp-search", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "McpSearchServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- JSON-RPC dispatch (also used directly by tests) ---
    def handle_message(self, msg: Any) -> Optional[Dict[str, Any]]:
        """One JSON-RPC message → response dict, or None for a notification."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return _rpc_error(msg.get("id") if isinstance(msg, dict) else None,
                              -32600, "Invalid Request")
        method = str(msg.get("method"))
        params = msg.get("params") or {}
        id_ = msg.get("id")
        with self._lock:
            self.requests_seen[method] = self.requests_seen.get(method, 0) + 1
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            requested = str((params or {}).get("protocolVersion") or DEFAULT_PROTOCOL)
            version = requested if requested in PROTOCOL_VERSIONS else DEFAULT_PROTOCOL
            return _rpc_result(id_, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
                "instructions": ("Web search and page reading provided by Deep Research Tool. "
                                 "Ground every finding in the returned pages and cite their URLs."),
            })
        if method == "ping":
            return _rpc_result(id_, {})
        if method == "tools/list":
            return _rpc_result(id_, {"tools": [TOOL_DEFINITIONS[t] for t in self.enabled_tools]})
        if method == "tools/call":
            name = str((params or {}).get("name") or "")
            args = (params or {}).get("arguments") or {}
            if name not in self.enabled_tools:
                return _rpc_error(id_, -32602, f"Unknown tool: {name}")
            try:
                if name == TOOL_SEARCH:
                    text = self.tools.web_search(args.get("queries"), args.get("max_results"))
                else:
                    text = self.tools.fetch_page(args.get("url"))
                return _rpc_result(id_, {"content": [{"type": "text", "text": text}],
                                         "isError": False})
            except Exception as e:  # noqa: BLE001 - tool errors go back as isError
                return _rpc_result(id_, {"content": [{"type": "text",
                                                      "text": f"{type(e).__name__}: {e}"}],
                                         "isError": True})
        return _rpc_error(id_, -32601, f"Method not found: {method}")


class _Handler(BaseHTTPRequestHandler):
    server_ref: McpSearchServer = None  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - silence the default stderr log
        pass

    def _send(self, status: int, body: Optional[bytes] = None,
              content_type: str = "application/json", extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body or b"")))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _path_ok(self) -> bool:
        path = self.path.split("?", 1)[0].rstrip("/")
        return path in ("/mcp", "")

    def do_GET(self):  # noqa: N802
        # no server-initiated stream in the stateless subset
        self._send(405, b"", extra={"Allow": "POST"})

    def do_DELETE(self):  # noqa: N802
        self._send(405, b"", extra={"Allow": "POST"})

    def do_POST(self):  # noqa: N802
        if not self._path_ok():
            self._send(404, b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, json.dumps(_rpc_error(None, -32600, "empty or too large body")).encode())
            return
        raw = self.rfile.read(length)
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(400, json.dumps(_rpc_error(None, -32700, "Parse error")).encode())
            return
        server = type(self).server_ref
        if isinstance(msg, list):          # batch
            responses = [r for r in (server.handle_message(m) for m in msg) if r is not None]
            if not responses:
                self._send(202, b"")
                return
            self._send(200, json.dumps(responses, ensure_ascii=False).encode("utf-8"))
            return
        response = server.handle_message(msg)
        if response is None:
            self._send(202, b"")
            return
        self._send(200, json.dumps(response, ensure_ascii=False).encode("utf-8"))


def backend_for(provider: str, *, endpoint: Optional[str] = None, search_client=None,
                timeout: float = 20.0) -> SearchBackend:
    """The backend for a SearchRunConfig: SearXNG by URL, otherwise the
    tool's search client (which must be given)."""
    if provider == "searxng":
        if not (endpoint or "").strip():
            raise ValueError("SearXNG の URL（searx_url）が未設定です")
        return SearxSearchBackend(endpoint.strip(), timeout=timeout, search_client=search_client)
    if search_client is None:
        raise ValueError("検索クライアントがありません（Open Deep Research の検索は"
                         "このツールの検索クライアントを MCP 経由で使います）")
    return ClientSearchBackend(search_client, name=provider)


__all__ = [
    "ALL_TOOLS", "TOOL_SEARCH", "TOOL_FETCH", "TOOL_DEFINITIONS", "SearchBackend",
    "ClientSearchBackend", "SearxSearchBackend", "SearchTools", "McpSearchServer",
    "backend_for", "format_searx_items",
]
