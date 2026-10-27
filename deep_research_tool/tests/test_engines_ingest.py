"""Engine result ingestion (engines/ingest.py): URL canonicalization,
sources-section parsing, chapter assignment, the [SOURCE k] citation
bridge, evidence seeding (fetched / excerpt-only) and the raw artifacts.
No network, no LLM."""
import json
from types import SimpleNamespace

import pytest

from deep_research_tool.engines.ingest import (
    assign_sources_to_sections, bridge_citations, build_session_from_result,
    canonicalize_url, dedupe_sources, parse_sources_section,
    save_engine_artifacts, seed_evidence_from_sources)
from deep_research_tool.engines.runner_core.models import (RunEvent, RunResult,
                                                           SourceRecord)
from deep_research_tool.evidence.locker import EvidenceLocker
from deep_research_tool.research.researcher import ResearchState


def _src(url, title=None, excerpt=None):
    return SourceRecord(url=url, title=title, excerpt=excerpt)


class TestUrlAndSources:
    def test_canonicalize_url(self):
        assert canonicalize_url("HTTPS://Example.org/a/b/?utm_source=x&b=2&a=1#frag") == \
            "https://example.org/a/b?b=2&a=1"
        assert canonicalize_url("http://x.com/") == "http://x.com/"
        assert canonicalize_url("http://x.com/p?fbclid=1&gclid=2&ref_=3&mc_eid=4") == "http://x.com/p"
        assert canonicalize_url("not a url") == "not a url"
        assert canonicalize_url("http://H.com:8080/x/") == "http://h.com:8080/x"

    def test_parse_sources_section_reads_last_sources_heading(self):
        md = ("# T\n\n## 本文\nx [1]\n\n## 出典\n\n[1] 概況: https://a.example/overview\n"
              "- [2] **統計** - https://a.example/stats.\nno url line\n"
              "## 次の見出し\n[3] https://a.example/ignored")
        out = parse_sources_section(md)
        assert [s["url"] for s in out] == ["https://a.example/overview", "https://a.example/stats"]
        assert out[0]["title"] == "概況" and out[1]["title"] == "統計"
        assert parse_sources_section("# no sources") == []
        assert parse_sources_section(None) == []
        assert parse_sources_section("### Sources\n[1] A: https://x/1\n[2] B: https://x/1")[0]["url"] == "https://x/1"

    def test_dedupe_by_canonical_url_keeps_first_and_fills_blanks(self):
        unique, index_map, by_canon = dedupe_sources([
            _src("https://a.example/x?utm_source=m", title=None, excerpt="e1"),
            _src("https://A.example/x/", title="T", excerpt=None),
            _src("https://a.example/y", title="Y"),
            _src("", title="empty"),
        ])
        assert [s.url for s in unique] == ["https://a.example/x?utm_source=m", "https://a.example/y"]
        assert unique[0].title == "T" and unique[0].excerpt == "e1"
        assert index_map == {1: 0, 2: 0, 3: 1}
        assert by_canon["https://a.example/x"] == 0


class TestAssignmentAndBridge:
    SOURCES = [_src("https://s.example/one", "成長率レポート", "年平均成長率は12%"),
               _src("https://s.example/two", "規制の動向", "規制強化の動きがある"),
               _src("https://s.example/three", "無関係", "zzz")]

    def _maps(self):
        unique, index_map, by_canon = dedupe_sources(self.SOURCES)
        return unique, index_map, by_canon

    def test_explicit_citations_win_in_first_appearance_order(self):
        unique, index_map, by_canon = self._maps()
        sections = [{"title": "A", "content": "規制 [2] と成長 [1][2] と https://s.example/three ."},
                    {"title": "B", "content": "[link](https://s.example/one?utm_source=x) only"}]
        out = assign_sources_to_sections(sections, unique, index_map, by_canon)
        assert out[0] == [1, 0, 2]
        assert out[1] == [0]

    def test_uncited_source_goes_to_best_chapter_or_chapter_one(self):
        unique, index_map, by_canon = self._maps()
        sections = [{"title": "A", "content": "市場の概況。"},
                    {"title": "B", "content": "規制強化の動きがあるとされる。規制の動向を注視。"}]
        out = assign_sources_to_sections(sections, unique, index_map, by_canon)
        assert 1 in out[1]              # 規制の動向 matched chapter B by 2-grams
        assert 2 in out[0]              # 無関係 fell back to chapter 1
        assert 0 in out[0] or 0 in out[1]
        assert sorted(i for v in out.values() for i in v) == [0, 1, 2]   # nothing dropped

    def test_bridge_rewrites_numbers_and_links_per_chapter(self):
        unique, index_map, by_canon = self._maps()
        text = "規制 [2] と成長 [1] と [リンク](https://s.example/one) と [9] と [2025] と [SOURCE 7]"
        out = bridge_citations(text, [1, 0], index_map, by_canon)
        assert out == ("規制 [SOURCE 1] と成長 [SOURCE 2] と リンク [SOURCE 2] と [9] と [2025] "
                       "と [SOURCE 7]")
        # a number whose source is not in this chapter's list stays as is
        assert bridge_citations("x [2]", [0], index_map, by_canon) == "x [2]"


def _result(md, sources=None, summary="要約"):
    return RunResult(report_markdown=md, summary=summary, sources=sources or [])


class TestBuildSession:
    MD = ("# 炭素繊維 調査レポート\n\n## 市場規模\n\n規模は120億ドル [1]。\n\n## 規制\n\n"
          "規制強化 [2]、詳細は [分析](https://s.example/three)。\n\n## 出典\n\n"
          "[1] 概況: https://s.example/one\n[2] 統計: https://s.example/two\n"
          "[3] 分析: https://s.example/three\n")
    SOURCES = [_src("https://s.example/one", "概況", "e1"),
               _src("https://s.example/two", "統計", "e2"),
               _src("https://s.example/three", "分析", "e3")]

    def test_session_from_report_with_sources(self):
        session, unique, section_sources = build_session_from_result(
            _result(self.MD, self.SOURCES), query="炭素繊維", requirements="r",
            language="ja", engine_id="mock-fast", engine_name="Mock Fast", session_id="abc")
        assert session.session_id == "abc" and session.state == ResearchState.COMPLETED
        assert session.research_plan.title == "炭素繊維 調査レポート"
        assert session.research_plan.summary == "要約"
        assert "external engine: mock-fast" in session.research_plan.methodology_notes
        assert list(session.section_contents) == ["1", "2"]          # 出典 dropped
        assert session.completed_sections == ["1", "2"]
        assert session.section_contents["1"]["content"] == "規模は120億ドル [SOURCE 1]。"
        assert session.section_contents["2"]["content"] == \
            "規制強化 [SOURCE 1]、詳細は 分析 [SOURCE 2]。"
        assert section_sources == {"1": [0], "2": [1, 2]}
        assert session.section_contents["2"]["sources"] == \
            ["https://s.example/two", "https://s.example/three"]
        assert session.run_config["research_engine"] == "mock-fast"
        assert session.requirements == "r"

    def test_sources_fall_back_to_the_report_section(self):
        session, unique, _ = build_session_from_result(
            _result(self.MD), query="q", engine_id="e", engine_name="E")
        assert [s.url for s in unique] == [s.url for s in self.SOURCES]
        assert session.section_contents["1"]["content"].endswith("[SOURCE 1]。")

    def test_summary_only_and_empty(self):
        session, _, _ = build_session_from_result(
            _result("", summary="短い要約だけ"), query="q", engine_id="e", engine_name="E")
        assert list(session.section_contents) == ["1"]
        assert session.section_contents["1"]["content"] == "短い要約だけ"
        with pytest.raises(ValueError, match="本文を返しませんでした"):
            build_session_from_result(_result("", summary=""), query="q",
                                      engine_id="e", engine_name="E")
        with pytest.raises(ValueError, match="章が見つかりません"):
            build_session_from_result(_result("## 出典\n[1] a: https://x/1"),
                                      query="q", engine_id="e", engine_name="E")


class FakeSearch:
    requires_serial_access = False

    def __init__(self, fail=(), short=()):
        self.fail, self.short, self.calls = set(fail), set(short), []

    def get_page_content(self, url):
        self.calls.append(url)
        if url in self.fail:
            raise ConnectionError("boom")
        text = "短い" if url in self.short else f"本文 {url} " * 60
        return SimpleNamespace(url=url, title=f"T:{url}", text_content=text, metadata={})


class TestSeedEvidence:
    def _session(self, tmp_path):
        session, unique, section_sources = build_session_from_result(
            _result(TestBuildSession.MD, TestBuildSession.SOURCES), query="炭素繊維",
            engine_id="mock-fast", engine_name="Mock Fast", session_id="s1")
        locker = EvidenceLocker(research_id="s1", output_dir=tmp_path / "evidence")
        return session, unique, section_sources, locker

    def test_fetched_and_excerpt_only_in_citation_order(self, tmp_path):
        session, unique, section_sources, locker = self._session(tmp_path)
        search = FakeSearch(fail={"https://s.example/two"}, short={"https://s.example/three"})
        stats = seed_evidence_from_sources(
            session, locker, unique, section_sources, search_client=search,
            engine_id="mock-fast", engine_name="Mock Fast", topic="炭素繊維",
            parallel_max_workers=4)
        assert stats == {"sources": 3, "fetched": 1, "excerpt_only": 2}
        assert sorted(search.calls) == sorted(s.url for s in unique)   # each fetched once
        ec2 = session.section_contents["2"]["extracted_content"]
        # [SOURCE 1] of chapter 2 = the first extracted item = source two
        assert [e["url"] for e in ec2] == ["https://s.example/two", "https://s.example/three"]
        assert ec2[0]["excerpt_only"] is True and ec2[0]["raw_content"] == "e2"
        assert ec2[1]["excerpt_only"] is True          # too short → excerpt
        ec1 = session.section_contents["1"]["extracted_content"]
        assert ec1[0]["excerpt_only"] is False and ec1[0]["title"] == "T:https://s.example/one"
        by_url = {e.url: e for e in locker.get_all_evidence()}
        assert by_url["https://s.example/two"].metadata["excerpt_only"] is True
        assert "ConnectionError" in by_url["https://s.example/two"].metadata["fetch_error"]
        assert by_url["https://s.example/one"].metadata["excerpt_only"] is False
        assert by_url["https://s.example/one"].metadata["engine"] == "mock-fast"
        assert [e.url for e in locker.get_section_evidence("2")] == \
            ["https://s.example/two", "https://s.example/three"]
        assert by_url["https://s.example/one"].search_query == "Mock Fast: 炭素繊維"

    def test_fetch_disabled_and_serial_client(self, tmp_path):
        session, unique, section_sources, locker = self._session(tmp_path)
        search = FakeSearch()
        search.requires_serial_access = True
        stats = seed_evidence_from_sources(
            session, locker, unique, section_sources, search_client=search,
            engine_id="e", engine_name="E", topic="t", fetch=False)
        assert stats["excerpt_only"] == 3 and search.calls == []
        assert all(e.metadata["fetch_error"].startswith("取得を省略")
                   for e in locker.get_all_evidence())
        stats = seed_evidence_from_sources(
            session, locker, unique, section_sources, search_client=None,
            engine_id="e", engine_name="E", topic="t")
        assert stats["fetched"] == 0

    def test_cancel_check_is_honoured(self, tmp_path):
        session, unique, section_sources, locker = self._session(tmp_path)

        def cancel():
            raise RuntimeError("cancelled by user")

        with pytest.raises(RuntimeError, match="cancelled"):
            seed_evidence_from_sources(
                session, locker, unique, section_sources, search_client=FakeSearch(),
                engine_id="e", engine_name="E", topic="t", cancel_check=cancel,
                parallel_max_workers=1)


class TestArtifacts:
    def test_result_json_and_report_md(self, tmp_path):
        res = _result("# T\n\n## A\n\nx", summary="s")
        events = [RunEvent(seq=1, type="stage", payload={"stage": "planning"}, ts="t")]
        paths = save_engine_artifacts(tmp_path, "sid", "mock-fast", res, events,
                                      {"topic": "T"})
        data = json.loads(open(paths["engine_result"], encoding="utf-8").read())
        assert data["engine_id"] == "mock-fast" and data["request"] == {"topic": "T"}
        assert data["result"]["summary"] == "s" and data["events"][0]["type"] == "stage"
        assert open(paths["engine_report"], encoding="utf-8").read() == "# T\n\n## A\n\nx"
        assert "engine_sid" in paths["engine_result"]
