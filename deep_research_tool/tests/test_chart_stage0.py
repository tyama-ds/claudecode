"""
Charts, stage 0 of the chart-variety plan (v1.15.2) — groundwork, nothing
new is drawn yet:
- the classic style is frozen: its prompt (hashed), its five forms
- forms / prompts are chosen per style (new forms go to enhanced only)
- a form no renderer can draw is an error, never silently a bar
- the HTML page draws the form the PNG was drawn in (enhanced turns a
  crowded bar chart horizontal)
- chart_yield.json: the form the LLM asked for, and the forms drawn
- enhanced: a comparison mixing periods (CAGR 2019-2024 next to
  2015-2024) is left out; classic keeps its output
- the census script counts candidate shapes in past runs (no LLM)
No paid API, no network.
"""

import hashlib
import json

import pytest

from deep_research_tool.chartkit import build_section_catalog, render_html, render_png, validate_panels
from deep_research_tool.chartkit import spec
from deep_research_tool.evidence.numerical_extractor import (DataType, DerivedMetricsCalculator,
                                                             MetricCategory, NumericalDataPoint,
                                                             NumericalDataStore)
from deep_research_tool.scripts import chart_census

CLASSIC_PROMPT_SHA = {
    "ja": "ec44ee6f917431dde4a1df47d57de4a7711ff2baddc67a11f7d17192b29a6aab",
    "en": "6ab97a37192f464b3d270b097848c7598c1db0a836093aebfb5accd06d3e98fc",
}
CLASSIC_FORMS = ("bar", "hbar", "stacked_bar", "stacked_hbar", "line_xy")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestClassicFrozen:
    def test_prompt_is_unchanged(self):
        for lang in ("ja", "en"):
            guide = spec.STYLE_GUIDES["classic"][lang]
            assert _sha(guide) == CLASSIC_PROMPT_SHA[lang], lang
            msgs = spec.build_messages("1. 章", "目的", [], max_panels=3, language=lang, style="classic")
            assert msgs["system"] == guide.replace("{max_panels}", "3")
        # no style (older callers) is classic
        assert spec.build_messages("1", "g", [], language="ja")["system"] == \
            spec.build_messages("1", "g", [], language="ja", style="classic")["system"]

    def test_forms_are_the_five(self):
        assert spec.FORMS_BY_STYLE["classic"] == CLASSIC_FORMS
        assert spec.forms_for(None) == CLASSIC_FORMS
        assert spec.forms_for("no-such-style") == CLASSIC_FORMS
        assert set(CLASSIC_FORMS) <= set(spec.forms_for("enhanced"))


def _data():
    return {"L": ["A社", "B社", "C社", "D社", "E社", "F社", "G社", "H社"],
            "s1": {"unit": "億円", "values": [80, 70, 60, 50, 40, 30, 20, 10],
                   "basis": ["公表"] * 8, "name": "売上高 2024", "kind": "comparison",
                   "labels_ref": "L", "source": "s",
                   "sources": [{"url": "https://www.meti.go.jp/x", "title": "経産省",
                                "evidence_id": "e1", "raw_text": "x"}] * 8}}


def _panel(form="bar", pid="p1"):
    return {"id": pid, "form": form, "title": "売上高（億円）", "labels_ref": "L",
            "series": [{"name": "売上高 2024", "values_ref": "s1", "role": "emphasis"}],
            "caption": "経産省（公表）", "judgment": "A社が最大"}


class TestFormsPerStyle:
    def test_an_unknown_form_is_rejected_for_either_style(self):
        for style in (None, "classic", "enhanced"):
            valid, rejected = validate_panels([_panel("pie")], _data(), style=style)
            assert not valid and "form が不正" in rejected[0][1]

    def test_the_requested_form_is_kept(self):
        valid, _ = validate_panels([_panel("bar")], _data(), style="enhanced")
        assert valid[0]["form_requested"] == "bar"

    def test_a_renderer_never_draws_an_unknown_form_as_bars(self, tmp_path):
        valid, _ = validate_panels([_panel("bar")], _data(), style="enhanced")
        p = dict(valid[0], form="pie")
        for style in ("classic", "enhanced"):
            with pytest.raises(ValueError, match="描画できません"):
                render_png(p, tmp_path / f"x_{style}.png", style=style)
        with pytest.raises(ValueError, match="描画できません"):
            render_html([p], title="t")

    def test_html_draws_the_form_the_png_was_drawn_in(self, tmp_path):
        valid, _ = validate_panels([_panel("bar")], _data(), style="enhanced")
        p = valid[0]
        render_png(p, tmp_path / "e.png", style="enhanced", embed_text=False)
        assert p.get("form_rendered") == "hbar"            # 8 labels: crowded
        page = render_html([p], title="t")
        assert '"form": "hbar"' in page or '"form":"hbar"' in page
        # classic draws the bar chart as asked, in both
        valid, _ = validate_panels([_panel("bar")], _data(), style="classic")
        q = valid[0]
        render_png(q, tmp_path / "c.png", style="classic", embed_text=False)
        assert "form_rendered" not in q
        page = render_html([q], title="t")
        assert '"form": "bar"' in page or '"form":"bar"' in page


def _cagr_points():
    pts = []
    for subj, y0, y1, v0, v1 in (("A社", 2019, 2024, 100, 200), ("B社", 2020, 2024, 50, 80),
                                 ("C社", 2015, 2024, 30, 90)):
        for y, v in ((y0, v0), (y1, v1)):
            pts.append(NumericalDataPoint(value=v, raw_text=f"{subj} 売上 {v}億円 {y}",
                                          normalized_value=v * 1e8, unit="億円", metric_name="売上高",
                                          subject=subj, data_type=DataType.CURRENCY,
                                          category=list(MetricCategory)[0], year=y,
                                          source_url="https://example.co.jp/ir", section_id="1"))
    store = NumericalDataStore()
    store.add_many(pts)
    return pts + DerivedMetricsCalculator(store).calculate_cagrs()


class TestMixedPeriods:
    def test_enhanced_leaves_out_cagrs_of_different_periods(self):
        pts = _cagr_points()
        names = lambda cat: [s.name for s in cat.series.values()]
        classic = build_section_catalog(pts, "1")
        assert any("CAGR" in n for n in names(classic))           # classic output unchanged
        enhanced = build_section_catalog(pts, "1", drop_mixed_periods=True)
        assert not any("CAGR" in n for n in names(enhanced))
        assert enhanced.stats["period_mixed"] == 1
        assert any("期間の違う値" in d for d in enhanced.dropped)
        # the same-year comparison and the time series stay
        assert any("2024" in n for n in names(enhanced))
        assert len([s for s in enhanced.series.values() if s.kind == "time"]) == 3

    def test_cagrs_of_one_period_stay(self):
        pts = [p for p in _cagr_points() if not getattr(p, "is_derived", False)]
        pts = [p for p in pts if p.subject != "B社" or p.year != 2020]
        pts.append(NumericalDataPoint(value=40, raw_text="B社 売上 40億円 2019", normalized_value=4e9,
                                      unit="億円", metric_name="売上高", subject="B社",
                                      data_type=DataType.CURRENCY, category=list(MetricCategory)[0],
                                      year=2019, source_url="https://example.co.jp/ir", section_id="1"))
        pts = [p for p in pts if p.subject != "C社"]
        store = NumericalDataStore()
        store.add_many(pts)
        all_pts = pts + DerivedMetricsCalculator(store).calculate_cagrs()   # both 2019-2024
        cat = build_section_catalog(all_pts, "1", drop_mixed_periods=True)
        assert any("CAGR" in s.name for s in cat.series.values())


class TestCensus:
    def _write(self, folder, data, yld=None):
        folder.mkdir(parents=True)
        (folder / "chart_data_1.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if yld:
            (folder / "chart_yield.json").write_text(json.dumps(yld), encoding="utf-8")

    def test_counts_candidate_shapes(self, tmp_path):
        ent = ["A社", "B社", "C社", "D社"]
        data = {"_meta": {}, "E": ent, "Y2": ["2020", "2024"], "Y5": ["2020", "2021", "2022", "2023"],
                "Q": ["2024Q1", "2024Q2", "2024Q3"],
                "share": {"kind": "comparison", "unit": "%", "values": [40, 30, 20, 10], "labels_ref": "E",
                          "name": "シェア 2024", "year": 2024},
                "part": {"kind": "comparison", "unit": "%", "values": [30, 20, 10, 5], "labels_ref": "E",
                         "name": "上位シェア 2023", "year": 2023},
                "rev": {"kind": "comparison", "unit": "億円", "values": [9, 8, 7, 6], "labels_ref": "E",
                        "name": "売上高 2024", "year": 2024},
                "two": {"kind": "time", "unit": "億円", "values": [1, 2], "labels_ref": "Y2",
                        "name": "A社 店舗数", "entity": "A社"},
                "qtr": {"kind": "time", "unit": "億円", "values": [1, 2, 3], "labels_ref": "Q",
                        "name": "A社 受注", "entity": "A社"}}
        for e in ent:
            data[f"t_{e}"] = {"kind": "time", "unit": "億円", "values": [1, 2, 3, 4], "labels_ref": "Y5",
                              "name": f"{e} 売上高", "entity": e}
        run = tmp_path / "out" / "job-1" / "reports" / "figures"
        self._write(run, data, {"chapters": {"1": {"panels": [
            {"form": "bar", "rendered": True}, {"form": "bar", "form_rendered": "hbar", "rendered": True},
            {"form": "line_xy", "rendered": False}]}}})
        r = chart_census.census_run(run)
        c = r["candidates"]
        assert c["donut"] == 1 and c["donut_part"] == 1
        assert c["slope"] == 1 and c["quarterly"] == 1
        assert c["ridge"] == 1 and c["heatmap"] == 1        # four companies' 売上高
        assert c["scatter"] == 1                             # シェア 2024 × 売上高 2024
        assert r["drawn"] == {"bar": 1, "hbar": 1}
        assert chart_census.find_runs(tmp_path / "out") == [run]

    def test_command_line(self, tmp_path, capsys):
        assert chart_census.main([str(tmp_path / "nothing")]) == 1
        assert "見つかりません" in capsys.readouterr().out
        run = tmp_path / "o" / "figures"
        self._write(run, {"_meta": {}, "E": ["a", "b", "c"],
                          "s": {"kind": "comparison", "unit": "%", "values": [50, 30, 20], "labels_ref": "E",
                                "name": "x", "year": 2024}})
        assert chart_census.main([str(tmp_path / "o"), "--json"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["total"]["donut"] == 1
