"""Minimal OpenAI-compatible server that plays the LLM for open-deep-research
0.0.16 (no network): answers by the shape of each request.

- response_format json_schema ResearchQuestion / ClarifyWithUser → JSON content
- tools with ConductResearch → one ConductResearch call, then ResearchComplete
- tools with web_search / fetch_page (the MCP tools) → web_search, then
  fetch_page of the first URL seen, then ResearchComplete
- no tools: compression prompt → summary with the URLs; final report
  prompt → markdown report with a "### Sources" section
Every response carries usage so usage_metadata is populated.

    python3 fake_openai_odr.py <port>
"""
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS = []
_URL_RE = re.compile(r"https?://[^\s<>\)\]\"']+")


def _text(messages):
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            out.append(" ".join(str(p.get("text", "")) for p in c if isinstance(p, dict)))
        else:
            out.append(str(c or ""))
    return "\n".join(out)


def _content_text(c):
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def _urls_in(messages):
    seen = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            for u in _URL_RE.findall(_content_text(m.get("content"))):
                u = u.rstrip(".,;:")
                if u not in seen:
                    seen.append(u)
    return seen


def _tool_names(req):
    names = []
    for t in req.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(fn["name"])
        elif isinstance(t, dict) and t.get("name"):
            names.append(t["name"])
    return names


def _tool_call(name, args):
    return {"id": f"call_{int(time.time()*1000) % 100000}_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def decide(req):
    """→ (content or None, tool_calls or None)"""
    messages = req.get("messages") or []
    text = _text(messages)
    low = text.lower()
    rf = req.get("response_format") or {}
    schema_name = ""
    if isinstance(rf, dict):
        schema_name = str((rf.get("json_schema") or {}).get("name") or "")
        schema_props = ((rf.get("json_schema") or {}).get("schema") or {}).get("properties") or {}
    else:
        schema_props = {}
    names = _tool_names(req)

    if schema_name == "ResearchQuestion" or "research_brief" in schema_props:
        return json.dumps({"research_brief": "国内の炭素繊維市場の規模・主要メーカー・用途別需要を、"
                                             "2025 年の一次資料にもとづいて調べる。"}, ensure_ascii=False), None
    if schema_name == "ClarifyWithUser" or "need_clarification" in schema_props:
        return json.dumps({"need_clarification": False, "question": "", "verification": "了解しました。"}), None
    if "ResearchQuestion" in names:
        return None, [_tool_call("ResearchQuestion", {"research_brief": "炭素繊維市場の調査"})]

    if "ConductResearch" in names:
        prior = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
                    and any(tc.get("function", {}).get("name") == "ConductResearch"
                            for tc in (m.get("tool_calls") or [])))
        if prior == 0:
            return None, [_tool_call("ConductResearch", {
                "research_topic": "国内の炭素繊維市場の規模（2025 年）と主要メーカー（東レ・帝人・三菱ケミカル）の戦略を、"
                                  "公的統計と各社の公表資料から詳しく調べる。"})]
        return None, [_tool_call("ResearchComplete", {})]

    if "web_search" in names or "fetch_page" in names:
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        if not tool_msgs and "web_search" in names:
            return None, [_tool_call("web_search", {"queries": ["炭素繊維 市場規模 2025", "炭素繊維 メーカー 戦略"],
                                                    "max_results": 3})]
        if len(tool_msgs) == 1 and "fetch_page" in names:
            urls = _urls_in(messages)
            return None, [_tool_call("fetch_page", {"url": urls[0] if urls else "https://ex.test/none"})]
        if not tool_msgs and "fetch_page" in names:
            return None, [_tool_call("fetch_page", {"url": "https://ex.test/start"})]
        return None, [_tool_call("ResearchComplete", {})]

    urls = _urls_in(messages)
    if "clean up the findings" in low or "compress" in low[:4000]:
        return ("調査結果の要約: 国内の炭素繊維市場は 2025 年に拡大し、主要メーカーは増産投資を進めている。\n"
                + "\n".join(f"- 出典: {u}" for u in urls) + "\n\n### Sources\n"
                + "\n".join(f"[{i}] 資料{i}: {u}" for i, u in enumerate(urls, 1))), None
    # the final report (also any other plain prompt)
    if not urls:
        urls = _URL_RE.findall(text)
    urls = list(dict.fromkeys(u.rstrip(".,;:") for u in urls)) or ["https://ex.test/fallback"]
    body = ("# 国内の炭素繊維市場の動向\n\n"
            "## 1. 市場規模\n\n国内の炭素繊維市場は 2025 年に拡大を続けている [1]。航空機向け需要の回復が背景にある [2]。\n\n"
            "## 2. 主要メーカーの戦略\n\n主要メーカーは増産投資を進めている [1]。\n\n"
            "### Sources\n" + "\n".join(f"[{i}] 資料{i}: {u}" for i, u in enumerate(urls[:4], 1)) + "\n")
    return body, None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json({"error": f"unsupported path {self.path}"}, 404)
            return
        content, tool_calls = decide(req)
        CALLS.append({"tools": _tool_names(req), "rf": bool(req.get("response_format")),
                      "content": (content or "")[:60], "tool_calls": [tc["function"]["name"] for tc in (tool_calls or [])]})
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"
        resp = {"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
                "model": req.get("model", "fake-model"),
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = {"role": "assistant", "content": content}
            if tool_calls:
                delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(tool_calls)]
            chunk = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": req.get("model", "fake-model"),
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            end = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                   "model": req.get("model", "fake-model"),
                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                   "usage": resp["usage"]}
            self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
            return
        self._json(resp)


def serve(port):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8812
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"fake openai on http://127.0.0.1:{port}/v1", flush=True)
    httpd.serve_forever()
