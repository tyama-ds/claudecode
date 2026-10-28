"""
Chart styles (report.chart_style): "classic" keeps the v1.5 look and only
fixes defects; "enhanced" adds the recommended changes.

- setting: config / Web UI parameter validation, classic by default
- fixes for both styles: scale repair against the original text, year
  coercion, a point claimed through the locker's section map, legend
  names that name another series, y_max far above the data, the unit in
  the title, yield counters, unparsable LLM replies told apart from []
- rendering QA in both styles over awkward panels: negative values on the
  axis, no overlapping labels, nothing cut off at the image edge; the
  enhanced style draws at the printed width and turns crowded bars
  horizontal
- grounding: a value that is not in its quote / a quote not in the page
- facts: a judgment that contradicts the data is replaced; enhanced adds
  the computed fact and the data's source line
- enhanced data rules: research firms / PR wires are 二次, actual before
  forecast for one year
- PDF / HTML reports carry the figures under their chapters and still pass
  the frozen-body artifact check; the pipeline writes chart_yield.json
No paid API: fake clients only.
"""

import copy
import json
from pathlib import Path

import pytest

from deep_research_tool.chartkit import build_section_catalog, render_png, validate_panels
from deep_research_tool.chartkit.data_layer import (SourcePolicy, _choose, classify_source_basis,
                                                    coerce_year, repair_scale)
from deep_research_tool.chartkit.facts import (apply_facts, judgment_conflict, panel_facts,
                                               source_line)
from deep_research_tool.chartkit.grounding import (QUOTE_NOT_IN_SOURCE, VALUE_NOT_IN_QUOTE,
                                                   ground_panel, quote_in_source, value_in_quote)
from deep_research_tool.chartkit.spec import fix_title_unit, parse_panels
from deep_research_tool.evidence.numerical_extractor import NumericalDataPoint, NumericalDataStore

SAMPLE = Path(__file__).resolve().parent.parent / "chartkit" / "samples" / "zyn"
PUB = "公表"


def _dp(subject, metric, year, value, unit, url="https://example.com/a", **kw):
    kw.setdefault("raw_text", f"{value}{unit}")
    return NumericalDataPoint(value=value, unit=unit, metric_name=metric, subject=subject,
                              year=year, source_url=url, source_title=url.split("/")[2], **kw)


def _series(values, unit, basis=None, kind="comparison", name="", sources=None, labels_ref="L"):
    return {"unit": unit, "values": values, "basis": basis or [PUB] * len(values),
            "name": name, "kind": kind, "labels_ref": labels_ref, "source": "s",
            "sources": sources or [{"url": "https://www.meti.go.jp/x", "title": "経産省",
                                    "evidence_id": "e1", "raw_text": f"{v}{unit}"} for v in values]}


# --------------------------------------------------------------------------
# setting
# --------------------------------------------------------------------------

class TestSetting:

    def test_config_default_and_validation(self):
        from deep_research_tool.config import create_config
        assert create_config(provider="openai", openai_api_key="t").report.chart_style == "classic"
        cfg = create_config(provider="openai", openai_api_key="t", chart_style="Enhanced")
        assert cfg.report.chart_style == "enhanced"
        with pytest.raises(ValueError):
            create_config(provider="openai", openai_api_key="t", chart_style="fancy")

    def test_web_parameter(self):
        from deep_research_tool.webui.server import _CONFIG_PARAM_MAP, _chart_style_param
        assert _CONFIG_PARAM_MAP["chart_style"] == "chart_style"
        assert _chart_style_param(" enhanced ") == "enhanced"
        with pytest.raises(ValueError):
            _chart_style_param("zzz")

    def test_ui_selector_round_trip(self):
        page = (Path(__file__).resolve().parent.parent / "webui" / "static" / "index.html").read_text(
            encoding="utf-8")
        assert 'id="feat_chart_style"' in page and 'value="classic" selected' in page
        assert "s.chart_style = $('feat_chart_style').value" in page
        assert "p.chart_style = $('feat_chart_style').value" in page


# --------------------------------------------------------------------------
# fixes for both styles: data and spec
# --------------------------------------------------------------------------

class TestDataFixes:

    def test_repair_scale(self):
        # the number of '50億ドル' with a billion unit: 10x off → the text's scale
        assert repair_scale(50, "市場は50億ドル規模", "billion USD") == ("億ドル", True)
        # a correct conversion is left alone
        assert repair_scale(5, "市場は50億ドル規模", "billion USD") == ("billion USD", False)
        assert repair_scale(1.2, "売上高は1.2兆円", "億円") == ("兆円", True)
        # English scale word: the unit keeps its own currency spelling
        assert repair_scale(5, "about 5 billion dollars", "億ドル") == ("billion ドル", True)
        # compound scales (千万) and percentages are never touched
        assert repair_scale(3, "3千万円", "億円") == ("億円", False)
        assert repair_scale(40, "シェア40%", "%") == ("%", False)

    def test_coerce_year(self):
        assert coerce_year(2024) == 2024 and coerce_year("FY2023") == 2023
        assert coerce_year("2025年") == 2025 and coerce_year("n/a") is None
        assert coerce_year(True) is None and coerce_year(3000) is None

    def test_point_claimed_through_the_locker_section_map(self):
        pts = [_dp("市場", "規模", 2023, 10, "億円", section_id="2", evidence_id="e9"),
               _dp("市場", "規模", 2024, 12, "億円", section_id="2", evidence_id="e9")]
        assert build_section_catalog(pts, "1", []).is_empty()
        cat = build_section_catalog(pts, "1", ["e9"])
        assert not cat.is_empty()
        assert cat.stats["points"] == 2 and cat.stats["series"] == 1

    def test_yield_counters(self):
        pts = [_dp("A", "", 2024, 1, "億円", section_id="1"),           # no metric
               _dp("A", "売上", 2024, 5, "億円", section_id="1")]       # one year, one entity
        cat = build_section_catalog(pts, "1")
        assert cat.is_empty()
        assert cat.stats["no_metric"] == 1 and cat.stats["time_too_short"] == 1
        assert cat.stats["comparison_too_few"] == 1 and cat.stats["series"] == 0


class TestSpecFixes:

    def _data(self):
        return {"years": ["2023", "2024"],
                "ts1": dict(_series([10, 12], "億円", kind="time", name="A社 売上", labels_ref="years"),
                            entity="a社"),
                "ts2": dict(_series([30, 25], "億円", kind="time", name="B社 売上", labels_ref="years"),
                            entity="b社")}

    def _panel(self, **kw):
        p = {"id": "p1", "form": "bar", "title": "売上（億円）", "labels_ref": "years",
             "series": [{"name": "B社", "values_ref": "ts1", "role": "emphasis"},
                        {"name": "A社", "values_ref": "ts2", "role": "context"}],
             "caption": "c", "judgment": "j"}
        p.update(kw)
        return p

    def test_swapped_legend_names_take_the_data_names(self):
        valid, rej = validate_panels([self._panel()], self._data())
        assert not rej
        names = [s["name"] for s in valid[0]["series"]]
        assert names == ["A社 売上", "B社 売上"] and len(valid[0]["name_fixes"]) == 2

    def test_y_max_far_above_the_data_is_dropped(self):
        valid, _ = validate_panels([self._panel(y_max=5000)], self._data())
        assert "y_max" not in valid[0]
        valid, _ = validate_panels([self._panel(y_max=40)], self._data())
        assert valid[0]["y_max"] == 40

    def test_title_unit(self):
        assert fix_title_unit("市場規模（億円）", "百万円") == "市場規模（百万円）"
        assert fix_title_unit("推計：調査会社別（10億ドル）", "億ドル") == "推計：調査会社別（億ドル）"
        assert fix_title_unit("推計（調査会社別）", "億ドル") == "推計（調査会社別）"
        assert fix_title_unit("推計", "億ドル") == "推計"
        assert fix_title_unit("推計", "億ドル", append=True) == "推計（億ドル）"

    def test_unparsable_reply_told_apart_from_an_empty_list(self):
        st = {}
        assert parse_panels("作れません", st) == [] and st["ok"] is False
        assert parse_panels("[]", st) == [] and st["ok"] is True

    def test_over_limit_is_counted(self):
        p2 = self._panel(id="p2", form="line_xy")
        stats = {}
        valid, rej = validate_panels([self._panel(), p2], self._data(), max_panels=1, stats=stats)
        assert len(valid) == 1 and stats["over_limit"] == 1

    def test_line_xy_sources_stay_aligned_with_points(self):
        data = {"years": ["2022", "2023", "2024"],
                "ts1": _series([5, None, 7], "億円", basis=[PUB, None, PUB], kind="time",
                               labels_ref="years",
                               sources=[{"raw_text": "5億円"}, None, {"raw_text": "7億円"}])}
        p = {"id": "x", "form": "line_xy", "title": "t", "labels_ref": "years",
             "series": [{"name": "s", "values_ref": "ts1", "role": "emphasis"}],
             "caption": "c", "judgment": "j"}
        valid, _ = validate_panels([p], data)
        s = valid[0]["series"][0]
        assert [v["y"] for v in s["values"]] == [5.0, 7.0]
        assert [x["raw_text"] for x in s["sources"]] == ["5億円", "7億円"]


# --------------------------------------------------------------------------
# rendering QA, both styles
# --------------------------------------------------------------------------

def _qa_panels():
    long = [f"製品カテゴリ{i}（国内向け）" for i in range(12)]
    data = {
        "cats": ["A社", "B社", "C社", "D社", "E社"],
        "long": long,
        "years": ["2019", "2020", "2021", "2022", "2023"],
        "neg": _series([12, -8, 5, -15, 3], "%", labels_ref="cats"),
        "many": _series([1200 + 137 * i for i in range(12)], "億円", labels_ref="long"),
        "growth": _series([5, -12, 8, 3, -2], "%", kind="time", labels_ref="years"),
        "close1": _series([9.0, 9.4, 9.5, 9.6, 10], "億円", kind="time", labels_ref="years"),
        "close2": _series([9.1, 9.3, 9.6, 9.5, 10.2], "億円", kind="time", labels_ref="years"),
        "tiny": _series([0.04, 0.031, 0.05, 0.02, 0.06], "億円", labels_ref="cats"),
        "huge": _series([1_250_000, 980_000, 2_400_000, 1_700_000, 640_000], "トン", labels_ref="cats"),
        "segA": _series([56, 40, 30, 1, 45], "%", labels_ref="cats"),
        "segB": _series([3, 35, 50, 60, 10], "%", labels_ref="cats"),
        "segC": _series([41, 25, 20, 39, 45], "%", labels_ref="cats"),
    }

    def panel(pid, form, ref, keys, **kw):
        roles = ["emphasis"] + ["context"] * 7 if not form.startswith("stacked") else ["part"] * 4
        p = {"id": pid, "form": form, "title": f"{pid}（単位）", "labels_ref": ref,
             "series": [{"name": k, "values_ref": k, "role": r} for k, r in zip(keys, roles)],
             "caption": "出所: テスト（公表）。定義の違いに注意。" * 2,
             "judgment": "比較して見る。"}
        p.update(kw)
        return p
    panels = [
        panel("neg_hbar", "hbar", "cats", ["neg"]),
        panel("neg_bar", "bar", "cats", ["neg"]),
        panel("many_bar", "bar", "long", ["many"]),
        panel("growth_line", "line_xy", "years", ["growth"]),
        panel("close_lines", "line_xy", "years", ["close1", "close2"]),
        panel("tiny_bar", "bar", "cats", ["tiny"]),
        panel("huge_hbar", "hbar", "cats", ["huge"]),
        panel("segments", "stacked_hbar", "cats", ["segA", "segB", "segC"]),
        panel("share_ymax", "hbar", "cats", ["segB"], y_max=100),
    ]
    valid, rejected = validate_panels(panels, data)
    assert not rejected, rejected
    return valid


@pytest.mark.parametrize("style", ["classic", "enhanced"])
def test_render_qa(tmp_path, style):
    pytest.importorskip("matplotlib")
    for panel in _qa_panels():
        qa = {}
        render_png(copy.deepcopy(panel), tmp_path / f"{panel['id']}.png", style=style, qa=qa)
        pid = panel["id"]
        assert qa["label_overlaps"] == 0, (pid, qa)
        assert qa["tick_overlaps"] == 0, (pid, qa)
        assert qa["spill"] == [], (pid, qa["spill"])
        lo, hi = qa["value_lim"]
        vals = [v["y"] if isinstance(v, dict) else v for s in panel["series"] for v in s["values"]
                if v is not None]
        if not panel["form"].startswith("stacked"):
            assert lo <= min(vals) and max(vals) <= hi, (pid, qa["value_lim"])
        if style == "enhanced":
            assert qa["size_px"][0] == 1100           # 5.5 in at 200 dpi: printed 1:1
        else:
            assert qa["size_px"][0] == 1500           # the v1.5 size


def test_enhanced_turns_crowded_bars_horizontal_classic_keeps_the_form(tmp_path):
    pytest.importorskip("matplotlib")
    many = next(p for p in _qa_panels() if p["id"] == "many_bar")
    c, e = copy.deepcopy(many), copy.deepcopy(many)
    render_png(c, tmp_path / "c.png", style="classic")
    render_png(e, tmp_path / "e.png", style="enhanced")
    assert "form_rendered" not in c and e["form_rendered"] == "hbar"


def test_value_labels_keep_small_numbers_and_separators():
    from deep_research_tool.chartkit.render_png import _fmt, _wrap
    assert _fmt(0.04, "億円") == "0.04" and _fmt(2707, "億円") == "2,707"
    assert _fmt(-8, "%", signed=True) == "-8%"
    wrapped = _wrap("2024年は6億缶、1.5倍に増えた。" * 3, 12)
    assert "6億缶" in wrapped.replace("\n", "") and all(not ln.startswith("、") for ln in wrapped.split("\n"))
    assert "1.5" in wrapped and "1.\n5" not in wrapped


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------

class TestGrounding:

    def test_value_and_quote(self):
        assert value_in_quote(3.1, "10億ドル", "約31億ドル")
        assert value_in_quote(56, "%", "数量シェア56%")
        assert value_in_quote(-8, "%", "前年比8%減")
        assert not value_in_quote(5, "億円", "売上高は50億円")
        assert quote_in_source("売上高は1,234億円", "当社の売上高は 1234 億円でした")
        assert quote_in_source("2024年 売上 1,234億円", "…2024年度の売上は1,234億円…")
        assert not quote_in_source("売上高は999億円", "当社の売上高は1234億円でした")

    def test_ground_panel_flags_both_problems(self):
        srcs = [{"evidence_id": "e1", "url": "u1", "raw_text": "A社は120億円"},
                {"evidence_id": "e2", "url": "u2", "raw_text": "B社は80億円"},
                {"evidence_id": "e3", "url": "u3", "raw_text": "C社は30億円"}]
        panel = {"labels": ["A社", "B社", "C社"], "unit": "億円",
                 "series": [{"name": "売上", "unit": "億円", "values": [120, 8, 30],
                             "basis": [PUB, "二次", "導出"], "sources": srcs}]}
        pages = {"e1": "決算: A社は120億円", "e2": "B社は80億円"}
        g = ground_panel(panel, lambda eid, url: pages.get(eid))
        assert g["checked"] == 2                    # 導出 is computed, not quoted
        assert [(i["label"], i["problem"]) for i in g["issues"]] == [("B社", VALUE_NOT_IN_QUOTE)]
        pages["e1"] = "別の記事"
        g = ground_panel(panel, lambda eid, url: pages.get(eid))
        assert ("A社", QUOTE_NOT_IN_SOURCE) in [(i["label"], i["problem"]) for i in g["issues"]]


# --------------------------------------------------------------------------
# facts
# --------------------------------------------------------------------------

def _line(values, judgment, forecast_last=False):
    srcs = [{"title": "業界団体", "url": "https://x.or.jp/a", "forecast": False} for _ in values]
    if forecast_last:
        srcs[-1] = dict(srcs[-1], forecast=True)
    return {"form": "line_xy", "unit": "%", "labels": [], "title": "成長率", "caption": "業界団体",
            "judgment": judgment,
            "series": [{"name": "成長率", "unit": "%", "basis": [PUB] * len(values), "sources": srcs,
                        "values": [{"x": float(2019 + i), "y": v} for i, v in enumerate(values)]}]}


class TestFacts:

    def test_direction_and_maximum_claims(self):
        f = panel_facts(_line([5, -12, 8, 3, -2], ""))
        assert f["kind"] == "time" and f["direction"] == "down"
        assert judgment_conflict("一貫して増加している。", f)
        assert judgment_conflict("2020年に大きく落ち込んだ。", f) is None     # true of 2020
        assert judgment_conflict("2020年に増加した。", f)
        assert judgment_conflict("2020年に落ち込んだが2021年に回復。", f) is None   # two labels
        assert judgment_conflict("増加は見られない。", f) is None              # negated
        bar = {"form": "bar", "unit": "億円", "labels": ["A社", "B社", "C社"],
               "series": [{"values": [300, 120, 50]}]}
        fb = panel_facts(bar)
        assert judgment_conflict("B社が最大。", fb) and judgment_conflict("A社が最大。", fb) is None

    def test_classic_replaces_only_a_contradiction(self):
        ok = _line([5, -12, 8, 3, -2], "2020年に大きく落ち込んだ。")
        assert apply_facts(ok, "classic") == {} and ok["judgment"] == "2020年に大きく落ち込んだ。"
        assert ok["caption"] == "業界団体" and ok["title"] == "成長率"
        bad = _line([5, -12, 8, 3, -2], "一貫して増加している。")
        rec = apply_facts(bad, "classic")
        assert rec["judgment_replaced"]["from"] == "一貫して増加している。"
        assert bad["judgment"].startswith("2019→2023で5%→-2%（減少")

    def test_enhanced_adds_fact_source_line_and_unit(self):
        p = _line([5, -12, 8, 3, -2], "2020年に大きく落ち込んだ。", forecast_last=True)
        rec = apply_facts(p, "enhanced")
        assert p["judgment"] == "2020年に大きく落ち込んだ。データ: 2019→2023で5%→-2%（減少、-7pt）。"
        assert "出所: 業界団体（公表5）" in p["caption"] and "予測値: 2023" in p["caption"]
        assert p["title"] == "成長率（%）" and rec["source_line"]

    def test_source_line_counts_and_truncates(self):
        srcs = [{"title": f"資料{i}", "url": f"https://s{i}.example/x"} for i in range(5)]
        panel = {"labels": list("abcde"),
                 "series": [{"values": [1, 2, 3, 4, 5], "basis": [PUB, PUB, "二次", "換算", "二次"],
                             "sources": srcs}]}
        assert source_line(panel) == "出所: 資料0、資料1、資料2 ほか2件（公表2・換算1・二次2）。"


# --------------------------------------------------------------------------
# enhanced data rules
# --------------------------------------------------------------------------

class TestEnhancedSourceRules:

    def test_research_firms_and_pr_wires_are_secondary(self):
        classic = SourcePolicy.for_style("classic")
        enhanced = SourcePolicy.for_style("enhanced")
        pr = "https://prtimes.jp/main/html/rd/p/000.html"
        firm = "https://www.grandviewresearch.com/press-release/global-market"
        assert classify_source_basis(pr, classic.primary_domains, classic.secondary_domains) == "公表"
        assert classify_source_basis(pr, enhanced.primary_domains, enhanced.secondary_domains) == "二次"
        assert classify_source_basis(firm, classic.primary_domains) == "公表"   # the /press-release path
        assert classify_source_basis(firm, enhanced.primary_domains, enhanced.secondary_domains) == "二次"
        assert classify_source_basis("https://www.meti.go.jp/x", enhanced.primary_domains,
                                     enhanced.secondary_domains) == "公表"

    def test_actual_before_forecast_for_one_year(self):
        gov = "https://www.stat.go.jp/x"
        forecast = _dp("市場", "規模", 2024, 120, "億円", gov, is_forecast=True)
        actual = _dp("市場", "規模", 2024, 110, "億円", "https://news.example.com/a")
        assert _choose([forecast, actual], SourcePolicy.for_style("classic")) is forecast
        assert _choose([forecast, actual], SourcePolicy.for_style("enhanced")) is actual


# --------------------------------------------------------------------------
# PDF / HTML reports
# --------------------------------------------------------------------------

class TestReportFigures:

    def _setup(self, tmp_path):
        from deep_research_tool.report.figure_table_generator import (
            Figure, FigureTableCollection, FigureType, TableData)
        from deep_research_tool.report.semantic_manifest import figure_semantics_markdown
        from deep_research_tool.tests.test_finalization_integration import (
            make_locker, make_plan, make_session)
        data = json.loads((SAMPLE / "data.json").read_text(encoding="utf-8"))
        panels = json.loads((SAMPLE / "panels.json").read_text(encoding="utf-8"))
        valid, _ = validate_panels(panels, data)
        png = tmp_path / "chart.png"
        render_png(valid[0], png, style="enhanced")
        coll = FigureTableCollection(
            charts=[Figure(figure_id="c1", figure_type=FigureType.CHART, title=valid[0]["title"],
                           caption="出所 2 件　見るべき点: 回復基調。", section_id="1", image_path=png),
                    Figure(figure_id="c2", figure_type=FigureType.CHART, title="章外の図",
                           caption="どの章にも属さない図。", section_id="9", image_path=png)],
            tables=[TableData(table_id="t1", title="主要メーカー", caption="出所: 各社IR",
                              headers=["社名", "シェア"], rows=[["A社", "40%"], ["B社", "25%"]],
                              section_id="2")])
        plan = make_plan([("1", "市場規模の推移", "d"), ("2", "競合", "d")])
        ev = [("https://ex.com/a", "x " * 30)]
        session = make_session(plan, {"1": {"title": "市場規模の推移", "content": "本文 [1]", "evidence": ev},
                                      "2": {"title": "競合", "content": "本文2 [1]", "evidence": ev}})
        appendix = figure_semantics_markdown(coll, language="ja", section_id="付録D")
        session.section_contents["付録D"] = {"title": "付録D", "content": appendix,
                                            "sources": [], "extracted_content": []}
        return session, make_locker(tmp_path, ev), coll, appendix

    def test_pdf_places_figures_under_their_chapters(self, tmp_path):
        pypdf = pytest.importorskip("pypdf")
        pytest.importorskip("reportlab")
        from deep_research_tool.config import ReportFormat
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.report.semantic_manifest import build_semantic_manifest, check_artifact
        session, locker, coll, appendix = self._setup(tmp_path)
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        gen.figure_collection = coll
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.PDF, verification_result=None)
        assert gen.figures_inserted == 3
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(p.extract_text() for p in reader.pages)
        assert sum(len(p.images) for p in reader.pages) == 2
        assert text.index("1. 市場規模の推移") < text.index("見るべき点: 回復基調") < text.index("2. 競合")
        assert text.index("2. 競合") < text.index("主要メーカー")
        assert "どの章にも属さない図" in text                     # never silently dropped
        manifest = build_semantic_manifest({"1": "本文 [1]", "2": "本文2 [1]", "付録D": appendix},
                                           figure_collection=coll)
        assert check_artifact(manifest, path, locker)["status"] == "pass"

    def test_html_embeds_figures(self, tmp_path):
        from deep_research_tool.config import ReportFormat
        from deep_research_tool.report.generator import ReportGenerator
        session, locker, coll, _ = self._setup(tmp_path)
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        gen.figure_collection = coll
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.HTML, verification_result=None)
        page = path.read_text(encoding="utf-8")
        assert page.count('class="report-figure"><img src="data:image/png;base64,') == 2
        # (the table of contents lists the chapters first: anchor on the headings)
        assert page.index("<h2>1. 市場規模の推移</h2>") < page.index("見るべき点: 回復基調") \
            < page.index("<h2>2. 競合</h2>")
        assert "<caption>主要メーカー — 出所: 各社IR</caption>" in page

    def test_html_report_passes_the_artifact_check(self, tmp_path):
        # the page's own CSS used to count as unverified additions, so every
        # verified V1 HTML report failed the check
        from deep_research_tool.config import ReportFormat
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.report.semantic_manifest import build_semantic_manifest, check_artifact
        session, locker, coll, appendix = self._setup(tmp_path)
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        gen.figure_collection = coll
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.HTML, verification_result=None)
        manifest = build_semantic_manifest({"1": "本文 [1]", "2": "本文2 [1]", "付録D": appendix},
                                           figure_collection=coll)
        res = check_artifact(manifest, path, locker)
        assert res["status"] == "pass", res

    def test_pdf_bullets_of_a_long_limitation_are_not_additions(self, tmp_path):
        # the limitations land in the LAST chapter — with figures, the
        # appendix — whose "- " items the PDF draws as "• "
        pytest.importorskip("reportlab")
        from deep_research_tool.config import ReportFormat
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.report.semantic_manifest import (build_semantic_manifest, check_artifact,
                                                                 normalize_semantic_text)
        assert normalize_semantic_text("• 未解決: 範囲") == normalize_semantic_text("- 未解決: 範囲")
        session, locker, coll, appendix = self._setup(tmp_path)
        appendix += ("\n### 調査上の限界\n\n- 未解決: 未検証の本文範囲が残っています（抽出チャンク 1/4 失敗、"
                     "リトライ2回後）。この範囲の数値は出典で確認できていないため、判断には使わないでください。\n")
        session.section_contents["付録D"]["content"] = appendix
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        gen.figure_collection = coll
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.PDF, verification_result=None)
        manifest = build_semantic_manifest({"1": "本文 [1]", "2": "本文2 [1]", "付録D": appendix},
                                           figure_collection=coll)
        res = check_artifact(manifest, path, locker)
        assert res["status"] == "pass", res

    def test_without_a_collection_nothing_changes(self, tmp_path):
        from deep_research_tool.config import ReportFormat
        from deep_research_tool.report.generator import ReportGenerator
        session, locker, _coll, _ = self._setup(tmp_path)
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.HTML, verification_result=None)
        assert gen.figures_inserted == 0
        assert 'class="report-figure"' not in path.read_text(encoding="utf-8")


class TestInsertionPaths:

    def _tool(self, tmp_path):
        from deep_research_tool.config import create_config
        from deep_research_tool.main import DeepResearchTool
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config = create_config(provider="openai", openai_api_key="t", output_dir=str(tmp_path))
        return tool

    def test_docx_failure_warns_instead_of_crashing(self, tmp_path):
        from types import SimpleNamespace
        from deep_research_tool.report.figure_table_generator import FigureTableCollection
        from deep_research_tool.utils.helpers import ResearchWarnings
        gen = SimpleNamespace(output_dir=tmp_path, add_figures_to_docx=lambda **kw: None)
        report = tmp_path / "r.docx"
        report.write_bytes(b"x")
        collector = ResearchWarnings.bind()
        try:
            out = self._tool(tmp_path)._insert_figures_into_report(
                report_path=report, collection=FigureTableCollection(), generator=gen)
            assert out is None
            assert any("DOCXへの挿入に失敗" in w["message"] for w in collector.to_dict_list())
        finally:
            ResearchWarnings.unbind()

    def test_a_finished_pdf_is_not_replaced_by_a_figures_only_file(self, tmp_path):
        from types import SimpleNamespace
        from deep_research_tool.report.figure_table_generator import FigureTableCollection
        from deep_research_tool.utils.helpers import ResearchWarnings
        called = []
        gen = SimpleNamespace(output_dir=tmp_path,
                              add_figures_to_pdf=lambda **kw: called.append(kw) or tmp_path / "x.pdf")
        report = tmp_path / "r.pdf"
        report.write_bytes(b"%PDF")
        collector = ResearchWarnings.bind()
        try:
            out = self._tool(tmp_path)._insert_figures_into_report(
                report_path=report, collection=FigureTableCollection(), generator=gen)
            assert out is None and not called
            assert any("PDF には図表を挿入できません" in w["message"] for w in collector.to_dict_list())
        finally:
            ResearchWarnings.unbind()


# --------------------------------------------------------------------------
# pipeline: style, grounding, facts, yield
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, content):
        self.content = content


class _LLM:
    """Writes one panel per chapter whose judgment contradicts the data."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system_prompt=None, **kw):
        self.calls += 1
        cat = json.loads(prompt.split("値は含めない）:\n", 1)[1])
        it = next(c for c in cat if c["kind"] == "時系列")
        return _Resp(json.dumps([{
            "id": "p1", "form": "bar", "title": f"{it['name']}（{it['unit']}）",
            "labels_ref": it["labels_ref"],
            "series": [{"name": it["name"], "values_ref": it["values_ref"], "role": "emphasis"}],
            "caption": "出所: 統計", "judgment": "一貫して減少している。"}], ensure_ascii=False))


@pytest.mark.parametrize("style", ["classic", "enhanced"])
def test_pipeline_style_grounding_facts_and_yield(tmp_path, style):
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace
    from deep_research_tool.config import create_config
    from deep_research_tool.evidence.locker import EvidenceLocker
    from deep_research_tool.main import DeepResearchTool
    from deep_research_tool.utils.helpers import ResearchWarnings
    cfg = create_config(provider="openai", openai_api_key="t", output_dir=str(tmp_path),
                        auto_figures=True, chart_style=style)
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.config, tool.llm_client, tool.stage_llm_clients = cfg, _LLM(), {}
    store = NumericalDataStore(research_topic="炭素繊維")
    gov = "https://www.stat.go.jp/x"
    for y, v in ((2021, 40), (2022, 50), (2023, 55)):
        store.add(_dp("炭素繊維市場", "市場規模", y, v, "億円", gov, section_id="1"))
    # a value whose quote says something else (grounding)
    store.add(_dp("炭素繊維市場", "市場規模", 2024, 60, "億円", gov, section_id="1",
                  raw_text="2024年は6.5億円"))
    locker = EvidenceLocker(research_id="r", output_dir=tmp_path / "ev")
    session = SimpleNamespace(query="炭素繊維", research_plan=None,
                              section_contents={"1": {"title": "市場規模", "content": "本文", "sources": []},
                                                "2": {"title": "需要", "content": "本文", "sources": []}})
    collector = ResearchWarnings.bind()
    try:
        figs = tool._spec_charts(session, locker, store, tmp_path / "figures")
        msgs = [w["message"] for w in collector.to_dict_list()]
    finally:
        ResearchWarnings.unbind()
    assert len(figs) == 1
    sem = figs[0].chart_data
    assert sem["judgment"].startswith("2021→2024で40億円→60億円（増加")   # replaced (both styles)
    assert any("見るべき点" in m and "差し替え" in m for m in msgs)
    assert any("出典本文で確認できません" in m for m in msgs)
    y = json.loads((tmp_path / "figures" / "chart_yield.json").read_text(encoding="utf-8"))
    assert y["style"] == style and y["charts"] == 1
    ch = y["chapters"]["1"]
    assert ch["catalog"]["points"] == 4 and ch["valid"] == 1 and ch["rendered"] == 1
    assert ch["panels"][0]["grounding"]["issues"][0]["label"] == "2024"
    assert y["chapters"]["2"]["catalog"]["points"] == 0
    if style == "enhanced":
        assert "出所: www.stat.go.jp" in sem["caption"] or "出所: stat.go.jp" in sem["caption"]
        assert "要確認" in sem["caption"]
    else:
        assert sem["caption"] == "出所: 統計"
