"""
chartkit — charts as spec + fixed renderer:
- data layer: canonical grouping, one display scale per series (re-scaled
  values tagged 換算), primary vs secondary sources, best point wins,
  section attribution through the locker, no values in the LLM catalog
- spec: the handoff's validate cases + the value gate; parsing; prompt
- renderers: a self-contained HTML page (no external script), PNGs for
  every form with the same rules
- pipeline: DeepResearchTool._spec_charts with a fake LLM — Figures with
  PNGs and chart_data, charts.html, the per-chapter limit, and a chapter
  whose spec fails never breaks the others
No paid API: fake clients only.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.chartkit import (build_messages, build_section_catalog,
                                         classify_source_basis, panel_semantics,
                                         parse_panels, parse_unit, render_html,
                                         render_png, validate_panels)
from deep_research_tool.chartkit.spec import SpecError, resolve, validate
from deep_research_tool.evidence.numerical_extractor import (NumericalDataPoint,
                                                             NumericalDataStore)

SAMPLE = Path(__file__).resolve().parent.parent / "chartkit" / "samples" / "zyn"


def _load(name):
    return json.loads((SAMPLE / name).read_text(encoding="utf-8"))


def _dp(subject, metric, year, value, unit, url="https://example.com/a", **kw):
    kw.setdefault("raw_text", f"{value}{unit}")
    return NumericalDataPoint(value=value, unit=unit,
                              metric_name=metric, subject=subject, year=year,
                              source_url=url, source_title=url.split("/")[2] if url else "",
                              **kw)


# --------------------------------------------------------------------------
# data layer
# --------------------------------------------------------------------------

class TestDataLayer:

    def test_unit_parsing_and_source_basis(self):
        assert parse_unit("billion USD") == ("ドル", 1e9, "10億")
        assert parse_unit("百万缶") == ("缶", 1e6, "百万")
        assert parse_unit("%") == ("%", 1.0, "")
        assert parse_unit("GPa") == ("gpa", 1.0, "")
        assert classify_source_basis("https://www.meti.go.jp/x") == "公表"
        assert classify_source_basis("https://www.sec.gov/Archives/x.htm") == "公表"
        assert classify_source_basis("https://corp.example.com/ir/results/2025") == "公表"
        assert classify_source_basis("https://news.example.com/article") == "二次"
        assert classify_source_basis("") == "二次"
        assert classify_source_basis("https://x.example", ["x.example"]) == "公表"

    def test_series_are_grouped_scaled_tagged_and_deduplicated(self):
        gov, news, ir = "https://www.stat.go.jp/x", "https://news.example.com/y", "https://toray.example.com/ir/2024"
        points = [
            _dp("炭素繊維市場", "市場規模", 2022, 50, "億円", gov, section_id="1"),
            _dp("炭素繊維市場", "市場規模", 2023, 5500, "百万円", news, section_id="1"),
            _dp("炭素繊維市場", "市場規模", 2024, 60, "億円", gov, section_id="1"),
            _dp("炭素繊維市場", "市場規模", 2024, 58, "億円", news, section_id="1"),   # conflicting
            _dp("Toray", "売上高", 2024, 300, "億円", ir, section_id="", evidence_id="ev-ir"),
            _dp("Teijin", "売上高", 2024, 200, "億円", news, section_id="1"),
            _dp("", "", 2024, 7, "", news, section_id="1"),                         # no metric: dropped
            _dp("炭素繊維市場", "市場規模", 2026, 70, "億円", news, section_id="2"),   # other chapter
        ]
        cat = build_section_catalog(points, "1", section_evidence_ids=["ev-ir"])
        assert not cat.is_empty()
        ts = next(s for s in cat.series.values() if s.kind == "time")
        assert cat.labels[ts.labels_ref] == ["2022", "2023", "2024"]
        assert ts.unit == "億円" and ts.values == [50.0, 55.0, 60.0]
        assert ts.basis == ["公表", "換算", "公表"]           # 5500 百万円 re-scaled
        assert "換算" in ts.note
        assert ts.sources[2]["url"] == gov                   # the primary source won
        cmp = next(s for s in cat.series.values() if s.kind == "comparison")
        assert cat.labels[cmp.labels_ref] == ["toray", "teijin"]
        assert cmp.values == [300.0, 200.0] and cmp.basis == ["公表", "二次"]
        assert cmp.year == 2024
        assert any("指標名なし" in d for d in cat.dropped)
        data = cat.to_data()
        assert data[ts.labels_ref] == ["2022", "2023", "2024"]
        assert data[ts.key]["values"] == [50.0, 55.0, 60.0]
        prompt_items = cat.catalog_for_prompt()
        assert prompt_items and all("values" not in it for it in prompt_items)
        assert {it["values_ref"] for it in prompt_items} == set(cat.series)

    def test_derived_and_interpolated_points_get_their_tags(self):
        points = [
            _dp("X", "売上高", 2022, 10, "億円", "https://x.go.jp/a", section_id="1"),
            _dp("X", "売上高", 2023, 12, "億円", "https://x.go.jp/a", section_id="1",
                is_derived=True, raw_text="Interpolated for 2023"),
            _dp("X", "売上高", 2024, 15, "億円", "https://x.go.jp/a", section_id="1",
                is_derived=True, raw_text="CAGR 2022-2024"),
        ]
        cat = build_section_catalog(points, "1")
        ts = next(iter(cat.series.values()))
        assert ts.basis == ["公表", "推定", "導出"]

    def test_empty_when_nothing_belongs_to_the_chapter(self):
        points = [_dp("X", "売上高", 2022, 10, "億円", section_id="9")]
        assert build_section_catalog(points, "1").is_empty()


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------

class TestSpec:

    def test_the_zyn_sample_validates_completely(self):
        valid, rejected = validate_panels(_load("panels.json"), _load("data.json"))
        assert [p["id"] for p in valid] == ["us_shipments", "offtake_growth", "us_share", "us_composition",
                                            "global_share", "global_volume", "market_estimates", "capacity"]
        assert rejected == []
        sem = panel_semantics(valid[0])
        assert sem["series"]["出荷"] == [202, 190, 206, 196, 155, 193]
        assert sem["series_detail"][0]["basis"][1] == "換算"
        assert sem["labels"] == ["25Q1", "25Q2", "25Q3", "25Q4", "26Q1", "26Q2"]
        # the manifest text (what the verifier reads) carries the numbers
        from deep_research_tool.report.semantic_manifest import figure_semantics_markdown
        from deep_research_tool.report.figure_table_generator import (Figure, FigureTableCollection,
                                                                      FigureType)
        coll = FigureTableCollection()
        coll.charts.append(Figure(figure_id="x", figure_type=FigureType.CHART, title="t", caption="c",
                                  section_id="1", chart_data=sem))
        md = figure_semantics_markdown(coll)
        assert "出荷:25Q1=202" in md and "見るべき点" in md

    def test_validate_rejects_the_handoff_cases(self):
        data = _load("data.json")
        base = _load("panels.json")[0]
        def _bad(**changes):
            p = json.loads(json.dumps(base)); p.update(changes); return p
        # (a) values_ref not in data
        p = _bad(series=[{"name": "x", "values_ref": "nope", "role": "emphasis"}])
        with pytest.raises(SpecError, match="nope"):
            validate(resolve([p], data))
        # (b) label count != value count
        p = _bad(labels_ref="growth_periods")
        with pytest.raises(SpecError, match="不一致"):
            validate(resolve([p], data))
        # (c) 9 series
        p = _bad(series=[{"name": f"s{i}", "values_ref": "zyn_us.shipments", "role": "context"} for i in range(9)])
        with pytest.raises(SpecError, match="1〜8"):
            validate(resolve([p], data))
        # (d) empty caption / judgment
        with pytest.raises(SpecError, match="caption"):
            validate(resolve([_bad(caption="")], data))
        with pytest.raises(SpecError, match="caption"):
            validate(resolve([_bad(judgment="  ")], data))
        # (e) form not allowed
        with pytest.raises(SpecError, match="form"):
            validate(resolve([_bad(form="pie")], data))
        # plus: two emphasis, stacked with one series, unknown highlight
        p = _bad(series=[{"name": "a", "values_ref": "zyn_us.shipments", "role": "emphasis"},
                         {"name": "b", "values_ref": "zyn_us.offtake_base", "role": "emphasis"}])
        with pytest.raises(SpecError, match="emphasis"):
            validate(resolve([p], data))
        with pytest.raises(SpecError, match="stacked"):
            validate(resolve([_bad(form="stacked_bar", series=base["series"][:1])], data))
        with pytest.raises(SpecError, match="highlight"):
            validate(resolve([_bad(highlight=["27Q9"])], data))

    def test_value_gate_and_mixed_units(self):
        data = {"l": ["a", "b", "c"],
                "flat": {"unit": "件", "values": [5, 5, 5], "basis": ["公表"] * 3},
                "one": {"unit": "件", "values": [5, None, None], "basis": ["公表", None, None]},
                "pct": {"unit": "%", "values": [1, 2, 3], "basis": ["公表"] * 3},
                "cnt": {"unit": "件", "values": [1, 2, 3], "basis": ["公表"] * 3}}
        mk = lambda pid, series: {"id": pid, "form": "bar", "title": "t（件）", "labels_ref": "l",
                                  "series": series, "caption": "c", "judgment": "j"}
        valid, rejected = validate_panels([
            mk("flat", [{"name": "f", "values_ref": "flat", "role": "emphasis"}]),
            mk("one", [{"name": "o", "values_ref": "one", "role": "emphasis"}]),
            mk("mixed", [{"name": "p", "values_ref": "pct", "role": "emphasis"},
                         {"name": "c", "values_ref": "cnt", "role": "context"}]),
            mk("ok", [{"name": "c", "values_ref": "cnt", "role": "emphasis"}]),
        ], data)
        assert [p["id"] for p in valid] == ["ok"]
        reasons = {pid.get("id"): r for pid, r in rejected}
        assert "同じ" in reasons["flat"] and "1 つ" in reasons["one"] and "単位" in reasons["mixed"]
        # the per-chapter limit stops after N valid panels
        valid, _ = validate_panels([mk("a", [{"name": "c", "values_ref": "cnt", "role": "emphasis"}]),
                                    mk("b", [{"name": "c", "values_ref": "cnt", "role": "emphasis"}])], data, max_panels=1)
        assert [p["id"] for p in valid] == ["a"]

    def test_parse_and_prompt(self):
        assert parse_panels('ここに説明。```json\n[{"id": "a"}]\n```') == [{"id": "a"}]
        assert parse_panels("[{\"id\": \"b\"}, 3]") == [{"id": "b"}]
        assert parse_panels("no json here") == [] and parse_panels("") == []
        cat = [{"values_ref": "ts1", "labels_ref": "years_ts1", "kind": "時系列", "name": "x 市場規模",
                "metric": "市場規模", "unit": "億円", "labels": ["2022", "2023"], "points": 2,
                "basis": ["公表"], "sources": 1}]
        msgs = build_messages("1. 市場規模", "国内市場の推移", cat, max_panels=6, language="ja",
                              research_topic="炭素繊維")
        assert "最大 6" in msgs["system"] and "ts1" in msgs["user"] and "炭素繊維" in msgs["user"]
        assert "values_ref" in msgs["user"] and '"values"' not in msgs["user"]


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------

class TestRenderers:

    def test_html_page_is_self_contained_and_escaped(self):
        data = _load("data.json")
        panels, _ = validate_panels(_load("panels.json"), data)
        panels[0]["title"] = 'x </script><b>y</b>'
        page = render_html(panels, kpis=_load("kpis.json"), title="ZYN",
                           sections=[{"title": "1. 米国", "panel_ids": [p["id"] for p in panels[:3]]},
                                     {"title": "2. 世界", "panel_ids": [p["id"] for p in panels[3:]]}])
        assert 'src="http' not in page and "cdnjs" not in page
        assert page.count("<canvas") == 8 and "cv_capacity" in page
        assert "&lt;/script&gt;" in page                       # title escaped in markup
        assert "</script><b>" not in page.split("const SPEC")[1].split("</script>")[0]
        assert "<\\/script>" in page                          # … and inside the spec JSON
        assert "7.94億缶" in page and "2. 世界" in page
        assert len(page.encode("utf-8")) < 1_000_000

    def test_png_for_every_form(self, tmp_path):
        pytest.importorskip("matplotlib")
        data = _load("data.json")
        panels, _ = validate_panels(_load("panels.json"), data)
        forms = {}
        for p in panels:
            out = render_png(p, tmp_path / f"{p['id']}.png")
            assert out.is_file() and out.stat().st_size > 5000
            forms[p["form"]] = True
        assert set(forms) == {"bar", "hbar", "stacked_bar", "stacked_hbar", "line_xy"}


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, content):
        self.content = content


class _SpecLLM:
    """Fake chart designer: answers with panels built from the catalog keys
    in the prompt (never a number)."""
    def __init__(self, n_panels=8, mode="ok"):
        self.n, self.mode, self.calls = n_panels, mode, 0

    def generate(self, prompt, system_prompt=None, **kw):
        self.calls += 1
        if self.mode == "garbage":
            return _Resp("すみません、チャートは作れません。")
        if self.mode == "raise":
            raise RuntimeError("LLM down")
        cat = json.loads(prompt.split("値は含めない）:\n", 1)[1])
        panels = []
        for i in range(self.n):
            it = cat[i % len(cat)]
            panels.append({"id": f"p{i + 1}", "form": "bar", "title": f"{it['name']}（{it['unit']}）",
                           "labels_ref": it["labels_ref"],
                           "series": [{"name": it["name"], "values_ref": it["values_ref"], "role": "emphasis"}],
                           "unit": it["unit"], "caption": f"出所: {it['sources']} 件（{'/'.join(it['basis'])}）",
                           "judgment": "推移を見る。"})
        return _Resp(json.dumps(panels, ensure_ascii=False))


def _pipeline_tool(tmp_path, llm, max_panels=6):
    from deep_research_tool.config import create_config
    from deep_research_tool.main import DeepResearchTool
    cfg = create_config(provider="openai", openai_api_key="t", output_dir=str(tmp_path),
                        auto_figures=True, chart_max_per_section=max_panels)
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.config = cfg
    tool.llm_client = llm
    tool.stage_llm_clients = {}
    return tool


def _store():
    store = NumericalDataStore(research_topic="炭素繊維")
    gov = "https://www.stat.go.jp/x"
    for y, v in ((2021, 40), (2022, 50), (2023, 55), (2024, 60)):
        store.add(_dp("炭素繊維市場", "市場規模", y, v, "億円", gov, section_id="1"))
    for e, v in (("Toray", 300), ("Teijin", 200), ("Mitsubishi", 150)):
        store.add(_dp(e, "売上高", 2024, v, "億円", "https://news.example.com/z", section_id="1"))
    for y, v in ((2023, 10), (2024, 12)):
        store.add(_dp("航空機向け", "需要", y, v, "千トン", gov, section_id="2"))
    return store


@pytest.fixture
def session_and_locker(tmp_path):
    from deep_research_tool.evidence.locker import EvidenceLocker
    locker = EvidenceLocker(research_id="r", output_dir=tmp_path / "ev")
    session = SimpleNamespace(
        query="炭素繊維の市場調査", research_plan=None,
        section_contents={"1": {"title": "市場規模", "content": "本文…", "sources": []},
                          "2": {"title": "需要", "content": "本文…", "sources": []},
                          "_executive_summary": {}})
    return session, locker


class TestPipeline:

    def test_spec_charts_render_pngs_and_the_html_page(self, tmp_path, session_and_locker):
        pytest.importorskip("matplotlib")
        session, locker = session_and_locker
        llm = _SpecLLM(n_panels=8)
        tool = _pipeline_tool(tmp_path, llm, max_panels=2)
        figs = tool._spec_charts(session, locker, _store(), tmp_path / "figures")
        assert llm.calls == 2                                  # one call per chapter
        by_sec = {}
        for f in figs:
            by_sec.setdefault(f.section_id, []).append(f)
        assert len(by_sec["1"]) == 2 and len(by_sec["2"]) == 1      # ≤ max per chapter; ch.2 has one series
        f = by_sec["1"][0]
        assert f.figure_id.startswith("s1_p") and Path(f.image_path).is_file()
        assert next(iter(f.chart_data["series"].values())) and f.chart_data["labels"]
        assert "見るべき点" in f.caption and f.title.endswith("）")
        html = tmp_path / "figures" / "charts.html"
        assert html.is_file() and tool.charts_html_path == html
        page = html.read_text(encoding="utf-8")
        assert "1. 市場規模" in page and "2. 需要" in page and 'src="http' not in page
        assert (tmp_path / "figures" / "chart_data_1.json").is_file()

    def test_garbage_or_failing_llm_yields_no_charts_and_no_crash(self, tmp_path, session_and_locker):
        from deep_research_tool.utils.helpers import ResearchWarnings
        session, locker = session_and_locker
        tool = _pipeline_tool(tmp_path, _SpecLLM(mode="garbage"))
        assert tool._spec_charts(session, locker, _store(), tmp_path / "figures") == []
        collector = ResearchWarnings.bind()
        try:
            tool = _pipeline_tool(tmp_path, _SpecLLM(mode="raise"))
            assert tool._spec_charts(session, locker, _store(), tmp_path / "figures2") == []
            msgs = [w["message"] for w in collector.to_dict_list()]
            assert any("チャート設計に失敗" in m for m in msgs)
        finally:
            ResearchWarnings.unbind()

    def test_defaults_prefer_the_spec_engine_and_no_fabrication(self):
        from deep_research_tool.config import create_config
        cfg = create_config(provider="openai", openai_api_key="t")
        rp = cfg.report
        assert rp.chart_engine == "spec" and rp.chart_max_per_section == 6
        assert rp.derived_fill_missing is False and rp.derived_market_shares is False
        assert rp.numerical_extract_from_sections is False
        from deep_research_tool.evidence.numerical_extractor import DerivedMetricsCalculator
        store = _store()
        calc = DerivedMetricsCalculator(store)
        with_shares = calc.calculate_all(include_shares=True)
        without = calc.calculate_all(include_shares=False)
        assert len(without) <= len(with_shares)
        assert not any("share" in (d.metric_name or "").lower() for d in without)


class TestReportAppendix:
    """The V1 renderer ships chapters that live outside the research plan
    (the verified figure appendix 付録D) in every format — otherwise the
    frozen appendix never reaches the artifact and the content check
    fails with "断片が欠落"."""

    APPENDIX = ("## 付録D. 図表一覧\n\n### 図: 炭素繊維市場 市場規模（億円）\n"
                "出所 1 件（二次）　見るべき点: 推移と差を見る。\n"
                "データ: 炭素繊維市場 市場規模:2023=50.0; 炭素繊維市場 市場規模:2024=60.0\n"
                "軸・単位: 億円\n\n### 表: 主要メーカー\n\n| 社名 | シェア |\n|---|---|\n"
                "| A社 | 40% |\n| B社 | 25% |\n")

    def _session(self, tmp_path):
        from deep_research_tool.tests.test_finalization_integration import (
            make_plan, make_session, make_locker)
        plan = make_plan([("1", "市場規模の推移", "d")])
        session = make_session(plan, {"1": {"title": "市場規模の推移",
                                            "content": "本文 [1]",
                                            "evidence": [("https://ex.com/a", "x " * 30)]}})
        session.section_contents["付録D"] = {"title": "付録D", "content": self.APPENDIX,
                                            "sources": [], "extracted_content": []}
        return session, make_locker(tmp_path, [("https://ex.com/a", "x " * 30)])

    @pytest.mark.parametrize("fmt", ["markdown", "docx", "html", "pdf"])
    def test_extra_chapter_reaches_every_format(self, tmp_path, fmt):
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.report.semantic_manifest import (
            build_semantic_manifest, check_artifact)
        from deep_research_tool.config import ReportFormat
        session, locker = self._session(tmp_path)
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat(fmt), verification_result=None)
        assert path.exists()
        if fmt == "html":
            # the HTML artifact's own CSS trips the "additions" direction of
            # the check (pre-existing); verify the appendix content directly
            text = path.read_text(encoding="utf-8")
            assert "<h2>付録D. 図表一覧</h2>" in text
            assert "データ: 炭素繊維市場 市場規模:2023=50.0" in text
            assert "<td>A社</td><td>40%</td>" in text
            return
        manifest = build_semantic_manifest(
            {"1": "本文 [1]", "付録D": self.APPENDIX}, figure_collection=None)
        res = check_artifact(manifest, path, locker)
        assert res["status"] == "pass", (fmt, res)
        assert res["checked"] >= 1

    def test_markdown_blocks_and_plan_filter(self):
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.tests.test_finalization_integration import make_plan
        blocks = ReportGenerator._markdown_blocks(self.APPENDIX)
        kinds = [k for k, _ in blocks]
        assert kinds == ["h2", "h3", "para", "h3", "table"]
        assert blocks[-1][1] == [["社名", "シェア"], ["A社", "40%"], ["B社", "25%"]]
        plan = make_plan([("1", "a", ""), ("1.1", "b", ""), ("2", "c", "")])
        extra = ReportGenerator._extra_sections(plan, {
            "1": {"content": "x"}, "1.1": {"content": "y"}, "2": {"content": "z"},
            "_executive_summary": {"content": "e"}, "付録D": {"content": "app"},
            "付録E": {"content": "   "}})
        assert extra == [("付録D", "app")]


class TestWordInsertion:
    """Spec charts reach the Word report: the PNG rendered by chartkit is
    embedded (with its caption) by the existing V1 DOCX insertion path."""

    def test_chart_png_embedded_in_docx(self, tmp_path):
        docx = pytest.importorskip("docx")
        from deep_research_tool.chartkit import render_png, validate_panels
        from deep_research_tool.report.figure_table_generator import (
            Figure, FigureTableCollection, FigureTableGenerator, FigureType)
        data = json.loads((SAMPLE / "data.json").read_text(encoding="utf-8"))
        panels = json.loads((SAMPLE / "panels.json").read_text(encoding="utf-8"))
        valid, _ = validate_panels(panels, data)
        png = tmp_path / "chart_s1_p1.png"
        render_png(valid[0], png)
        fig = Figure(figure_id="s1_p1", figure_type=FigureType.CHART,
                     title=valid[0]["title"], caption="出所 2 件　見るべき点: 回復基調。",
                     section_id="1", image_path=str(png))
        coll = FigureTableCollection(charts=[fig])
        doc = docx.Document()
        doc.add_heading("1. 市場規模の推移", level=1)
        doc.add_paragraph("本文。")
        doc.add_heading("2. 競合", level=1)
        doc.add_paragraph("本文2。")
        src = tmp_path / "r.docx"
        doc.save(src)
        gen = FigureTableGenerator(llm_client=None, output_dir=tmp_path, language="ja")
        out = gen.add_figures_to_docx(docx_path=src, collection=coll)
        assert out and Path(out).exists()
        d = docx.Document(str(out))
        assert len(d.inline_shapes) == 1
        texts = [p.text for p in d.paragraphs]
        i_cap = texts.index("出所 2 件　見るべき点: 回復基調。")
        assert texts.index("1. 市場規模の推移") < i_cap < texts.index("2. 競合")


class TestValueFormatting:
    """'%' values are signed only on change panels; shares stay unsigned.
    The PNG and the Chart.js prelude apply the same rule."""

    def test_percent_sign_rule(self):
        from deep_research_tool.chartkit.render_png import _fmt, _signed_pct
        share = {"title": "ZYNの米国シェア（%）", "series": [{"values": [61.5, 56]}]}
        growth = {"title": "前年比成長率（%）", "series": [{"values": [12.0, 8.5]}]}
        mixed = {"title": "増減", "series": [{"values": [3, -2]}]}
        assert not _signed_pct(share) and _signed_pct(growth) and _signed_pct(mixed)
        assert _fmt(56, "%", _signed_pct(share)) == "56%"
        assert _fmt(12.0, "%", _signed_pct(growth)) == "+12%"
        assert _fmt(-2, "%", True) == "-2%"
        assert _fmt(1234, "百万缶") == "1,234" and _fmt(2.345, "億円") == "2.3"
        prelude = (Path(__file__).resolve().parent.parent / "chartkit" / "prelude.js").read_text(encoding="utf-8")
        assert "signedPct" in prelude and "前年比|前期比|成長" in prelude
