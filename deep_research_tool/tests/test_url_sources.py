"""指定 URL source modes (urls / urls_web): URL normalisation, fetching with
a same-site crawl, the researcher on named pages (no web search in urls,
boosted URL evidence and site: queries in urls_web), the tool's result,
config / Web UI validation, the pre-check endpoint and the hand-off to an
external engine. No network, no LLM."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.config import CrawlMode, ResearchSourceMode, create_config
from deep_research_tool.research.researcher import Researcher, ResearchState
from deep_research_tool.research.url_sources import (
    MAX_URLS, fetch_url_documents, normalize_domains, normalize_urls, precheck_urls)
from deep_research_tool.tests.test_concurrency_visibility import _post, web  # noqa: F401
from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
from deep_research_tool.tests.test_finalization_integration import build_tool
from deep_research_tool.webui.server import FieldError, build_config_kwargs, engine_request_error

BODY = "炭素繊維の国内市場は拡大を続けている。航空機向け需要の回復が背景にある。" * 20


class LinkySearch(FakeSearchClient):
    """Pages with same-site links, one failing URL, one short page."""

    def get_page_content(self, url):
        self.fetch_calls += 1
        if "fail" in url:
            return SimpleNamespace(url=url, title="", text_content="", metadata={"error": "HTTP 404"}, links=[])
        if "short" in url:
            return SimpleNamespace(url=url, title="短い", text_content="短すぎる", metadata={}, links=[])
        if url.endswith("boom"):
            raise ConnectionError("boom")
        depth0 = url.count("/") <= 3
        links = ([{"url": url.rstrip("/") + "/sub1", "text": "s1"},
                  {"url": url.rstrip("/") + "/sub2#frag", "text": "s2"},
                  {"url": "https://other.example/x", "text": "other"},
                  {"url": url.rstrip("/") + "/pic.png", "text": "img"},
                  {"url": url, "text": "self"}] if depth0 else [])
        return SimpleNamespace(url=url, title="T " + url, text_content=BODY,
                               metadata={"published_date": "2026-01-01"}, links=links)


class TestNormalise:
    def test_urls(self):
        out = normalize_urls("https://A.example/x/\nhttp://b.example/y, c.example/z  ftp://no  <https://d.example>")
        assert out == ["https://A.example/x/", "http://b.example/y", "https://c.example/z", "https://d.example"]
        assert normalize_urls(["https://a/1", "https://a/1/", " ", None]) == ["https://a/1"]
        assert normalize_urls(None) == [] and normalize_urls("") == []
        assert len(normalize_urls([f"https://h/{i}" for i in range(MAX_URLS + 5)])) == MAX_URLS

    def test_domains(self):
        assert normalize_domains("https://meti.go.jp/x, site:JCMA.jp, bad, www.a.example/path") == \
            ["meti.go.jp", "jcma.jp", "www.a.example"]
        assert normalize_domains(["a.example", "a.example"]) == ["a.example"]
        assert normalize_domains(None) == []


class TestFetch:
    def test_fetch_report_and_crawl(self):
        s = LinkySearch()
        docs, rep = fetch_url_documents(
            ["https://a.example/x", "https://a.example/fail", "https://a.example/short",
             "https://a.example/boom"], s, crawl_depth=1, max_pages=10)
        assert [d["path"] for d in docs] == ["https://a.example/x", "https://a.example/x/sub1",
                                              "https://a.example/x/sub2"]
        assert docs[0]["metadata"]["source"] == "url" and docs[0]["metadata"]["depth"] == 0
        assert docs[1]["metadata"]["seed_url"] == "https://a.example/x" and docs[1]["metadata"]["depth"] == 1
        assert docs[0]["metadata"]["published_date"] == "2026-01-01"
        assert rep["requested"] == 4 and rep["fetched"] == 1 and rep["crawled"] == 2 and rep["pages"] == 3
        errors = {f["url"]: f["error"] for f in rep["failed"]}
        assert errors["https://a.example/fail"] == "HTTP 404"
        assert "短すぎます" in errors["https://a.example/short"]
        assert "ConnectionError" in errors["https://a.example/boom"]
        assert len(rep["urls"]) == 6 and rep["urls"][0]["depth"] == 0

    def test_max_pages_and_no_crawl(self):
        s = LinkySearch()
        docs, rep = fetch_url_documents(["https://a.example/x"], s, crawl_depth=1, max_pages=2)
        assert len(docs) == 2 and rep["crawled"] == 1
        s = LinkySearch()
        docs, rep = fetch_url_documents(["https://a.example/x"], s, crawl_depth=0)
        assert len(docs) == 1 and s.fetch_calls == 1 and rep["crawled"] == 0
        assert fetch_url_documents([], s) == ([], {"requested": 0, "fetched": 0, "failed": [],
                                                   "crawled": 0, "pages": 0, "urls": []})

    def test_precheck_and_missing_client(self):
        rows = precheck_urls(["https://a.example/x", "https://a.example/fail"], LinkySearch())
        assert rows[0]["ok"] and rows[0]["chars"] == len(BODY) and rows[0]["title"].startswith("T ")
        assert not rows[1]["ok"] and rows[1]["error"] == "HTTP 404"
        assert precheck_urls(["https://a.example/report.pdf"], LinkySearch())[0]["kind"] == "pdf"
        docs, rep = fetch_url_documents(["https://a.example/x"], None)
        assert not docs and "クライアント" in rep["failed"][0]["error"]

    def test_cancel_check(self):
        def cancel():
            raise RuntimeError("cancelled")
        with pytest.raises(RuntimeError, match="cancelled"):
            fetch_url_documents(["https://a.example/x"], LinkySearch(), cancel_check=cancel)


def _researcher(tmp_path, mode, search, **kw):
    kw.setdefault("source_urls", ["https://a.example/x", "https://a.example/fail"])
    return Researcher(llm_client=FakeLLM(), search_client=search, output_dir=tmp_path,
                      source_mode=mode, crawl_mode=CrawlMode.STANDARD,
                      use_enhanced_synthesis=False, filter_mode="none", max_gap_fill_rounds=0,
                      max_queries_per_iteration=2, max_pages_per_query=2, **kw)


class TestResearcher:
    def test_urls_only_never_searches(self, tmp_path):
        search = LinkySearch()
        r = _researcher(tmp_path, ResearchSourceMode.URLS, search)
        s = r.conduct_research(query="炭素繊維の市場調査", requirements="目次は2章構成でよい")
        assert s.state == ResearchState.COMPLETED
        assert search.search_calls == 0 and search.fetch_calls == 2      # the two named URLs only
        assert r.url_source_report["fetched"] == 1 and r.url_source_report["failed"][0]["url"].endswith("fail")
        ev = r.evidence_locker.get_all_evidence()
        assert ev and all(e.access_method == "url" and e.search_query == "source_urls" for e in ev)
        assert all(e.url == "https://a.example/x" for e in ev)          # the page itself, no pseudo URL
        assert all(e.section_reference for e in ev) and all(e.importance_score >= 0.8 for e in ev)
        assert all(e.evidence_type.value == "web_page" for e in ev)
        for sid in ("1", "2"):
            assert "[SOURCE" in s.section_contents[sid]["content"]
            assert s.section_contents[sid]["sources"] == ["https://a.example/x"]

    def test_urls_web_boosts_named_pages_and_searches_preferred_hosts_first(self, tmp_path):
        search = LinkySearch()
        r = _researcher(tmp_path, ResearchSourceMode.URLS_WEB, search,
                        preferred_domains=["meti.go.jp", "jcma.jp"])
        s = r.conduct_research(query="炭素繊維の市場調査", requirements="目次は2章構成でよい")
        assert s.state == ResearchState.COMPLETED and search.search_calls > 0
        executed = [q for it in s.iterations for q in it.queries_executed]
        site = [q for q in executed if q.startswith("site:")]
        assert site and site[0].startswith("site:meti.go.jp ") and any(q.startswith("site:jcma.jp ") for q in site)
        assert any(not q.startswith("site:") for q in executed)
        ev = r.evidence_locker.get_all_evidence()
        url_ev = [e for e in ev if e.access_method == "url"]
        web_ev = [e for e in ev if e.access_method != "url"]
        assert url_ev and web_ev
        assert min(e.importance_score for e in url_ev) >= 0.8
        assert r._queries_with_preferred_domains([]) == []
        r.preferred_domains = []
        assert r._queries_with_preferred_domains(["a", "b"]) == ["a", "b"]

    def test_urls_mode_requires_readable_pages(self, tmp_path):
        r = _researcher(tmp_path, ResearchSourceMode.URLS, LinkySearch(), source_urls=["https://a.example/fail"])
        with pytest.raises(ValueError, match="取得できませんでした"):
            r.conduct_research(query="q")
        r = _researcher(tmp_path, ResearchSourceMode.URLS, LinkySearch(), source_urls=[])
        with pytest.raises(ValueError, match="source_urls"):
            r.conduct_research(query="q")
        # the classic modes are untouched by the new properties
        assert ResearchSourceMode.WEB.uses_web_search and not ResearchSourceMode.WEB.uses_local_store
        assert ResearchSourceMode.LOCAL.uses_local_files and not ResearchSourceMode.LOCAL.uses_web_search
        assert ResearchSourceMode.URLS.uses_url_sources and not ResearchSourceMode.URLS.uses_local_files
        assert ResearchSourceMode.URLS_WEB.uses_web_search and ResearchSourceMode.URLS_WEB.uses_local_store


class TestToolAndConfig:
    def test_result_carries_url_sources(self, tmp_path):
        tool = build_tool(tmp_path, FakeLLM(), LinkySearch(), source_mode="urls_web",
                          source_urls=["https://a.example/x", "https://a.example/fail"],
                          preferred_domains=["meti.go.jp"], enable_verification=False,
                          research_iterations=1)
        tool.config.research.use_enhanced_synthesis = False
        tool.config.research.max_gap_fill_rounds = 0
        res = tool.run(query="炭素繊維の市場調査", requirements="目次は2章構成でよい")
        u = res["url_sources"]
        assert u["mode"] == "urls_web" and u["requested"] == 2 and u["fetched"] == 1
        assert u["failed"][0]["url"] == "https://a.example/fail" and u["evidence_count"] >= 1
        assert u["preferred_domains"] == ["meti.go.jp"] and u["engine"] == "builtin"
        snap = tool._run_config_snapshot()
        assert snap["source_urls"] == ["https://a.example/x", "https://a.example/fail"]
        assert any("取得できませんでした" in w["message"] for w in res["warnings"])
        tool = build_tool(tmp_path / "w", FakeLLM(), LinkySearch(), enable_verification=False)
        assert tool._url_sources_summary() is None

    def test_create_config(self):
        c = create_config(provider="openai", openai_api_key="x", source_mode="urls",
                          source_urls="https://a.example/x\nhttps://a.example/x/", url_crawl_depth=1,
                          url_max_pages=40, preferred_domains="https://meti.go.jp/x")
        rc = c.research
        assert rc.source_mode == ResearchSourceMode.URLS and rc.source_urls == ["https://a.example/x"]
        assert rc.url_crawl_depth == 1 and rc.url_max_pages == 40 and rc.preferred_domains == ["meti.go.jp"]
        for bad, msg in [(dict(source_mode="urls"), "requires source_urls"),
                         (dict(source_mode="urls_web", source_urls=["ftp://x"]), "requires source_urls"),
                         (dict(url_crawl_depth=2), "0 or 1"),
                         (dict(url_max_pages=0), "between 1 and 100"),
                         (dict(research_engine="mock-fast", source_mode="local"), "source_mode 'web', 'urls' or 'urls_web'")]:
            with pytest.raises(ValueError, match=msg):
                create_config(provider="openai", openai_api_key="x", **bad)
        c = create_config(provider="openai", openai_api_key="x", research_engine="mock-fast",
                          source_mode="urls_web", source_urls=["https://a/1"])
        assert c.research.source_mode == ResearchSourceMode.URLS_WEB


class TestWebUi:
    def test_param_conversion(self):
        out = build_config_kwargs({"source_mode": "urls_web",
                                   "source_urls": "https://a/1\nhttps://a/1/ , b.example/x",
                                   "url_crawl_depth": True, "url_max_pages": "40",
                                   "preferred_domains": "https://meti.go.jp/x, site:jcma.jp"})
        assert out["source_urls"] == ["https://a/1", "https://b.example/x"]
        assert out["url_crawl_depth"] == 1 and out["url_max_pages"] == 40
        assert out["preferred_domains"] == ["meti.go.jp", "jcma.jp"]
        assert build_config_kwargs({"url_crawl_depth": 0})["url_crawl_depth"] == 0
        for bad, field in [({"source_mode": "urls"}, "source_urls"),
                           ({"source_mode": "urls", "source_urls": ["ftp://x"]}, "source_urls"),
                           ({"url_max_pages": 0}, "url_max_pages"),
                           ({"url_crawl_depth": 3}, "url_crawl_depth"),
                           ({"source_urls": 5}, "source_urls")]:
            with pytest.raises(FieldError) as ei:
                build_config_kwargs(bad)
            assert ei.value.field == field

    def test_engine_request_allows_urls(self):
        assert engine_request_error({"research_engine": "mock-fast", "source_mode": "urls_web",
                                     "source_urls": ["https://a/1"]}) is None
        msg, field = engine_request_error({"research_engine": "mock-fast", "source_mode": "local"})
        assert field == "research_engine" and "手元の資料" in msg

    def test_precheck_endpoint(self, web, monkeypatch):
        import deep_research_tool.main as main_mod
        monkeypatch.setattr(main_mod, "build_search_client", lambda config: LinkySearch())
        status, d = _post(web, "/api/precheck-urls", {"urls": "https://a.example/x\nhttps://a.example/fail",
                                                      "provider": "openai", "openai_api_key": "k"})
        assert status == 200, d
        assert [u["ok"] for u in d["urls"]] == [True, False] and d["urls"][1]["error"] == "HTTP 404"
        status, d = _post(web, "/api/precheck-urls", {"urls": ""})
        assert status == 200 and d["urls"] == []
        status, d = _post(web, "/api/research", {"query": "q", "provider": "openai", "openai_api_key": "k",
                                                 "source_mode": "urls", "source_urls": []})
        assert status == 400 and d["field"] == "source_urls"


class TestEngineHandOff:
    def test_adapter_passes_input_urls_and_complement_flag(self, tmp_path):
        from deep_research_tool.engines import get_engine
        from deep_research_tool.engines.adapter import EngineResearcher
        from deep_research_tool.engines.gptr.engine import build_worker_payload
        from deep_research_tool.engines.runner_core.models import RunInput, RunRequest
        r = EngineResearcher(get_engine("mock-fast"), output_dir=tmp_path, options={"speed_factor": 0.01},
                             source_urls=["https://a/1", ""], complement_urls=False, fetch_sources=False,
                             progress_callback=lambda m, p: None)
        r.conduct_research("q")
        req = r._build_request("q", "", "")
        assert req.input.input_urls == ["https://a/1"] and req.options["complement_source_urls"] is False
        saved = json.loads(Path(r.engine_artifacts["engine_result"]).read_text(encoding="utf-8"))
        assert saved["request"]["options"]["complement_source_urls"] is False
        payload, _ = build_worker_payload(RunRequest(
            client_run_id="r", engine_id="gpt-researcher",
            input=RunInput(topic="t", input_urls=["https://a/1"]), options={"complement_source_urls": False}))
        assert payload["input_urls"] == ["https://a/1"] and payload["complement_source_urls"] is False
        payload, _ = build_worker_payload(RunRequest(client_run_id="r", engine_id="gpt-researcher",
                                                     input=RunInput(topic="t")))
        assert payload["complement_source_urls"] is True
        r2 = EngineResearcher(get_engine("mock-fast"), output_dir=tmp_path / "n", fetch_sources=False,
                              progress_callback=lambda m, p: None)
        assert "complement_source_urls" not in r2.options

    def test_tool_hands_urls_to_the_engine(self, tmp_path):
        tool = build_tool(tmp_path, FakeLLM(), LinkySearch(), research_engine="mock-fast",
                          engine_options={"speed_factor": 0.01}, source_mode="urls",
                          source_urls=["https://a/1"], enable_verification=False)
        res = tool.run(query="炭素繊維市場")
        assert res["run_status"] == "completed" and res["url_sources"]["engine"] == "mock-fast"
        assert res["url_sources"]["requested"] == 1 and "fetched" not in res["url_sources"]
        assert tool.researcher.source_urls == ["https://a/1"] and tool.researcher.complement_urls is False
        tool = build_tool(tmp_path / "h", FakeLLM(), LinkySearch(), research_engine="mock-fast",
                          engine_options={"speed_factor": 0.01}, enable_verification=False)
        tool.config.research.source_mode = ResearchSourceMode.HYBRID
        with pytest.raises(ValueError, match="手元の資料"):
            tool.run(query="x")
