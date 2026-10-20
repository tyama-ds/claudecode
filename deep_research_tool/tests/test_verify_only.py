"""v1.6.0 — run modes: research WITH verification (default), research
ONLY, and VERIFY ONLY (a saved research or an external report goes
through the finalization loop without a research loop). No network, no
real LLM."""
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deep_research_tool.report.report_loader import (report_text_from_file,
                                                     session_from_report,
                                                     split_report_sections)
from deep_research_tool.tests.test_finalization_integration import (
    CLAIM_SENTENCE, SUPPORT_PHRASE, SUPPORTED_EVIDENCE, FakeSearch, build_tool,
    j, long_body, make_llm, make_locker, make_plan, make_session)
from deep_research_tool.tests.test_concurrency_visibility import _post, web  # noqa: F401

MD = """# 炭素繊維市場レポート

## 目次
- 1. 市場規模
- 2. メーカー

## 1. 市場規模の推移
国内市場は2,000億円に達した。

### 1.1 内訳
航空機向けが6割を占める。

## 第2章 主要メーカーの動向
東レが首位である。

## 参考文献
1. https://example.com/a
"""


class TestReportLoader:
    def test_markdown_headings_become_chapters(self, tmp_path):
        title, secs = split_report_sections(MD)
        assert title == "炭素繊維市場レポート"
        assert [s["title"] for s in secs] == ["市場規模の推移", "主要メーカーの動向"]
        assert "### 1.1 内訳" in secs[0]["content"] and "6割" in secs[0]["content"]
        p = tmp_path / "r.md"
        p.write_text(MD, encoding="utf-8")
        s = session_from_report(p, query="炭素繊維")
        assert sorted(s.section_contents) == ["1", "2"]
        toc = s.research_plan.table_of_contents
        assert [i.section for i in toc.items] == ["1", "2"]
        assert s.query == "炭素繊維" and s.run_config["source"] == "external_report"

    def test_level_one_only_and_plain_text(self, tmp_path):
        _t, secs = split_report_sections("# A\nx\n\n# B\ny\n", "doc")
        assert [s["title"] for s in secs] == ["A", "B"]
        t, secs = split_report_sections("本文だけ。\n二行目。", "memo")
        assert t == "memo" and len(secs) == 1 and secs[0]["content"].startswith("本文")
        with pytest.raises(ValueError):
            session_from_report(tmp_path / "missing.pdf")

    def test_docx_headings_tables_and_title(self, tmp_path):
        docx = pytest.importorskip("docx")
        doc = docx.Document()
        doc.add_heading("タイトル", 0)
        doc.add_heading("1. 市場規模", 1)
        doc.add_paragraph("本文A。")
        doc.add_heading("内訳", 2)
        doc.add_paragraph("本文B。")
        doc.add_heading("2. 動向", 1)
        doc.add_paragraph("本文C。")
        t = doc.add_table(rows=2, cols=2)
        t.rows[0].cells[0].text = "指標"
        t.rows[1].cells[1].text = "42"
        p = tmp_path / "r.docx"
        doc.save(p)
        title, text = report_text_from_file(p)
        assert title == "タイトル" and "## 1. 市場規模" in text and "### 内訳" in text
        s = session_from_report(p)
        assert s.research_plan.title == "タイトル"
        assert sorted(s.section_contents) == ["1", "2"]
        assert "本文B" in s.section_contents["1"]["content"]
        assert "| 指標 |" in s.section_contents["2"]["content"]


def _saved_research(tmp):
    """A finished research of this tool on disk: session + evidence JSON."""
    plan = make_plan([("1", "市場シェア", "A社のシェアを明らかにする")])
    session = make_session(plan, {"1": {"title": "市場シェア",
                                        "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                                        "evidence": SUPPORTED_EVIDENCE}})
    locker = make_locker(tmp, SUPPORTED_EVIDENCE)
    locker.export_to_json()                      # tmp/evidence/evidence_itest.json
    path = tmp / f"session_{session.session_id}.json"
    session.saved_artifacts = {"session": str(path)}
    session.save(path)
    return path


class TestVerifyOnly:
    def test_saved_research_is_verified_without_a_research_loop(self, tmp_path):
        session_path = _saved_research(tmp_path)
        out = tmp_path / "verify-job"
        llm = make_llm()
        search = FakeSearch()
        tool = build_tool(out, llm, search)
        result = tool.verify_report(session_path=str(session_path))
        assert result["kind"] == "verify"
        assert result["verify_source"]["mode"] == "session"
        assert result["verify_source"]["chapters"] == 1
        assert result["verify_source"]["evidence_items"] == 1
        assert result["status"]["verification"] == "performed"
        assert result["status"]["quality"] == "passed"
        assert result["run_status"] == "completed"
        assert tool.finalization_outcome["decision"] == "accept"
        assert llm.prompts("extract") and not search.searched        # no research loop
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        assert CLAIM_SENTENCE in report
        saved = result["saved_artifacts"]
        for key in ("session", "evidence_json", "final_body", "report", "verification_html"):
            assert Path(saved[key]).exists(), key
        # the verify run's artifacts live in ITS output dir, the original is untouched
        assert Path(saved["session"]).parent == out
        assert session_path.exists()
        plan = result["research_plan"]                   # outline for the plan viewer
        assert plan and "市場シェア" in json.dumps(plan, ensure_ascii=False)

    def test_external_report_gathers_evidence_then_verifies(self, tmp_path):
        report = tmp_path / "external.md"
        report.write_text("# 市場レポート\n\n## 1. 市場シェア\n\n"
                          + long_body("1", "市場シェア", CLAIM_SENTENCE).split("\n\n", 1)[1]
                          .replace(" [SOURCE 1]", ""),
                          encoding="utf-8")
        new_url = "https://new.example.com/share"

        def extract(prompt):
            # a correct extractor: the [SOURCE N] numbers that really sit
            # next to the claim (none in the external body at first)
            import re
            m = re.search(re.escape(CLAIM_SENTENCE) + r"((?:\s*\[SOURCE \d+\])*)", prompt)
            if not m:
                return j({"claims": []})
            nums = sorted({int(n) for n in re.findall(r"\[SOURCE (\d+)\]", m.group(1))})
            return j({"claims": [{"claim": CLAIM_SENTENCE, "importance": "critical",
                                  "source_numbers": nums}]})
        llm = make_llm().on("extract", extract).on(
            "edit", lambda p: f"## 1. 市場シェア\n\n{CLAIM_SENTENCE} [SOURCE 1]。"
                              + "根拠を反映した本文として市場の構造と需要動向を述べる。" * 40)
        search = FakeSearch(results={"": [new_url]},
                            pages={new_url: f"統計によればA社の{SUPPORT_PHRASE}である。" * 20})
        tool = build_tool(tmp_path / "out", llm, search)
        result = tool.verify_report(report_path=str(report), query="A社の市場シェア")
        assert result["verify_source"]["mode"] == "report"
        assert result["verify_source"]["evidence_items"] == 0      # nothing at start
        assert result["verify_source"]["seeded_pages"] == 1        # candidate sources collected
        assert search.searched and new_url in search.fetched
        assert result["status"]["verification"] == "performed"
        assert tool.finalization_outcome["decision"] == "accept"    # rewritten with a citation
        assert "[1]" in tool.finalization_outcome["chapters"]["1"]
        ev = json.loads(Path(result["evidence_json"]).read_text(encoding="utf-8"))
        assert any(new_url in json.dumps(e, ensure_ascii=False) for e in
                   (ev.get("evidence") or ev.get("items") or ev.values() if isinstance(ev, dict) else ev))
        assert Path(result["report_path"]).exists()
        assert Path(result["verification_html"]).exists()

    def test_requires_a_source_and_a_body(self, tmp_path):
        tool = build_tool(tmp_path, make_llm(), FakeSearch())
        with pytest.raises(ValueError):
            tool.verify_report()
        empty = tmp_path / "empty.md"
        empty.write_text("# only a title\n", encoding="utf-8")
        with pytest.raises(ValueError):
            tool.verify_report(report_path=str(empty))


class TestResearchOnlyMode:
    def test_enable_verification_false_skips_the_finalization(self, tmp_path):
        from deep_research_tool.tests.test_finalization_integration import run_tool
        llm = make_llm()
        plan = make_plan([("1", "市場シェア", "d")])
        session = make_session(plan, {"1": {"title": "市場シェア",
                                            "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                                            "evidence": SUPPORTED_EVIDENCE}})
        locker = make_locker(tmp_path, SUPPORTED_EVIDENCE)
        tool = build_tool(tmp_path, llm, FakeSearch(), enable_verification=False)
        result = run_tool(tool, session, locker)
        assert result["status"]["verification"] == "skipped"
        assert result["status"]["quality"] == "unverified"
        assert not llm.prompts("extract") and result["verification_html"] is None
        assert Path(result["report_path"]).exists()


class _FakeTool:
    """Stands in for DeepResearchTool inside the Web UI job."""
    calls = []

    def __init__(self, config):
        self.config = config
        self.verification_progress = None
        self.researcher = None

    def request_cancel(self):
        pass

    def current_saved_artifacts(self):
        return {}

    def concurrency_snapshot(self):
        return {}

    def llm_routing_snapshot(self):
        return {}

    def progress_snapshot(self):
        return None

    def current_plan(self):
        return None

    def verify_report(self, **kw):
        _FakeTool.calls.append((self.config.enable_verification, kw))
        return {"kind": "verify", "verify_source": {"mode": "session", "chapters": 1,
                                                     "evidence_items": 2},
                "report_path": kw.get("session_path"), "session_id": "s1",
                "status": {"process": "completed", "verification": "performed",
                           "quality": "passed"}, "run_status": "completed",
                "saved_artifacts": {"report": kw.get("session_path")},
                "research_plan": None}


class TestVerifyEndpoint:
    def _wait(self, web, job_id):
        import time
        import urllib.request
        port = web.server_address[1]
        for _ in range(100):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/jobs", timeout=10) as r:
                jobs = json.loads(r.read())["jobs"]
            job = next((x for x in jobs if x["job_id"] == job_id), None)
            if job and job["state"] in ("completed", "error", "cancelled"):
                return job
            time.sleep(0.1)
        raise AssertionError("job did not finish")

    def test_validation(self, web, tmp_path):
        status, d = _post(web, "/api/verify", {"provider": "openai"})
        assert status == 400 and d["field"] == "verify_source"
        status, d = _post(web, "/api/verify", {"job_id": "job-nope"})
        assert status == 404
        outside = tmp_path.parent / "outside.md"
        outside.write_text("# x\n\n## 1. a\nb\n", encoding="utf-8")
        status, d = _post(web, "/api/verify", {"report_path": str(outside)})
        assert status == 400
        bad = tmp_path / "uploads" / "t" / "r.pdf"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"%PDF")
        status, d = _post(web, "/api/verify", {"report_path": str(bad)})
        assert status == 400 and "md / txt / docx" in d["error"]

    def test_uploaded_report_runs_a_verify_job(self, web, tmp_path):
        rp = tmp_path / "uploads" / "abc" / "report.md"
        rp.parent.mkdir(parents=True)
        rp.write_text("# 題\n\n## 1. 章\n本文。\n", encoding="utf-8")
        _FakeTool.calls.clear()
        with patch("deep_research_tool.main.DeepResearchTool", _FakeTool):
            status, d = _post(web, "/api/verify", {
                "report_path": str(rp), "provider": "openai",
                "openai_api_key": "sk-test", "enable_verification": False})
            assert status == 202, d
            job = self._wait(web, d["job_id"])
        assert job["state"] == "completed"
        assert job["query"].startswith("検証のみ: report.md")
        res = job["result"]
        assert res["kind"] == "verify" and res["verify_source"]["chapters"] == 1
        assert res["status"]["verification"] == "performed"
        enabled, kw = _FakeTool.calls[-1]
        assert enabled is True                       # forced on for this mode
        assert kw["report_path"] == str(rp) and kw["session_path"] is None
        assert "検証のみ" in job["message"]

    def test_history_job_is_verified_again(self, web, tmp_path):
        from deep_research_tool.webui.server import ResearchJob
        manager = web.job_manager
        session_path = tmp_path / "old-job" / "session_x.json"
        session_path.parent.mkdir(parents=True)
        session_path.write_text("{}", encoding="utf-8")
        old = ResearchJob("job-old", "元の調査", params={"query": "元の調査"})
        old.result = {"saved_artifacts": {"session": str(session_path)}}
        old.finish("completed")
        manager.jobs[old.job_id] = old
        manager.record(old, output_dir=str(session_path.parent))
        _FakeTool.calls.clear()
        with patch("deep_research_tool.main.DeepResearchTool", _FakeTool):
            status, d = _post(web, "/api/verify", {"job_id": "job-old", "provider": "openai",
                                                    "openai_api_key": "sk-test"})
            assert status == 202, d
            job = self._wait(web, d["job_id"])
        assert job["state"] == "completed" and job["query"] == "検証のみ: 元の調査"
        _enabled, kw = _FakeTool.calls[-1]
        assert kw["session_path"] == str(session_path)
        # the new job writes into its OWN directory, never the old one
        assert job["result"]["output_dir"] != str(session_path.parent)


class TestVerifyOnlyBodySource:
    def test_frozen_raw_chapters_are_preferred_over_display_text(self, tmp_path):
        """After a verified run the session holds display-numbered text
        ([1]); the verifier needs the frozen [SOURCE N] form kept in
        final_body_<id>.json."""
        session_path = _saved_research(tmp_path)
        from deep_research_tool.research.researcher import ResearchSession
        session = ResearchSession.load(session_path)
        display = session.section_contents["1"]["content"].replace("[SOURCE 1]", "[1]")
        session.section_contents["1"]["content"] = display
        session.save(session_path)
        (tmp_path / f"final_body_{session.session_id}.json").write_text(json.dumps({
            "session_id": session.session_id,
            "chapters": {"1": display},
            "raw_chapters": {"1": long_body("1", "市場シェア", CLAIM_SENTENCE), "要旨": "x"},
        }, ensure_ascii=False), encoding="utf-8")
        llm = make_llm()
        tool = build_tool(tmp_path / "verify", llm, FakeSearch())
        result = tool.verify_report(session_path=str(session_path))
        assert result["verify_source"]["body"] == "final_body_raw"
        assert "[SOURCE 1]" in llm.prompts("extract")[0]      # the frozen form was verified
        assert result["status"]["verification"] == "performed"
        assert tool.finalization_outcome["decision"] == "accept"
