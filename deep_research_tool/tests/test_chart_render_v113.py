"""
v1.13.0 chart rendering (chartkit render_png / prelude.js / render_html):

- classic: plain value-axis ticks with thousands separators (no '1e6'
  offset), classic's sizes unchanged
- embed_text: report figures leave the title, the 見るべき点 line and the
  caption out of the PNG (main.py passes embed_text=False)
- wrapping: CJK breaks anywhere, Latin at spaces / hyphens, no line
  starts or ends with a space, none is longer than the width
- point / value labels never sit on another series' markers, lines or
  bars (they move or are dropped; the emphasised series keeps its labels)
- a line chart's value axis follows the data (0 only near 0); bars keep 0
- forecasts: dashed segment + hollow marker, dotted hollow bar with a
  dashed outline, a legend note; a second actual series is solid
- one series: no legend box (the notes stay)
- proportional CJK fonts first in both styles
- prelude.js: the same rules on the HTML page (run under node when it is
  installed), maxBarThickness only on bar datasets
No paid API, no network: fakes only.
"""

import copy
import importlib
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.chartkit import render_png, validate_panels

RP = importlib.import_module("deep_research_tool.chartkit.render_png")
PUB = "公表"


def _series(values, unit, basis=None, kind="comparison", forecast=None):
    fc = forecast or [False] * len(values)
    return {"unit": unit, "values": values, "basis": basis or [PUB] * len(values), "name": "", "kind": kind,
            "labels_ref": "L", "source": "s",
            "sources": [{"url": "https://www.meti.go.jp/x", "title": "経産省", "evidence_id": "e1",
                         "raw_text": f"{v}{unit}", "forecast": f} for v, f in zip(values, fc)]}


def _panel(form, labels, specs, unit="億円", title="タイトル（億円）", **kw):
    """A resolved panel. ``specs``: [(values, role, {basis=…, forecast=…})]."""
    data = {"L": list(labels)}
    ser = []
    for i, (vals, role, extra) in enumerate(specs):
        data[f"s{i}"] = _series(list(vals), unit, kind="time" if form == "line_xy" else "comparison", **extra)
        ser.append({"name": f"s{i}", "values_ref": f"s{i}", "role": role})
    p = {"id": "x", "form": form, "title": title, "labels_ref": "L", "series": ser,
         "caption": kw.pop("caption", "出所: 経産省（公表）。"), "judgment": kw.pop("judgment", "伸びが続く。")}
    p.update(kw)
    valid, rej = validate_panels([p], data)
    assert not rej, rej
    return valid[0]


def _draw(panel, style, tmp_path, name="c", **kw):
    """Render and return what the saved figure holds."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seen = {}
    orig = plt.Figure.savefig

    def ticks(axis, lim):
        lo, hi = sorted(lim)
        return [(loc, t.get_text()) for loc, t in zip(axis.get_ticklocs(), axis.get_ticklabels())
                if lo <= loc <= hi and t.get_text()]

    def spy(fig, *a, **k):
        ax = fig.axes[0]
        seen["title"] = RP._title_artist(ax).get_text()
        seen["fig_texts"] = [t.get_text() for t in fig.texts if t.get_text()]
        seen["legend"] = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
        seen["labels"] = [(t.get_text().strip(), t.get_va()) for t in ax.texts if t.get_visible() and t.get_text()]
        seen["yticks"] = ticks(ax.yaxis, ax.get_ylim())
        seen["xticks"] = ticks(ax.xaxis, ax.get_xlim())
        seen["ytick_size"] = {t.get_fontsize() for t in ax.get_yticklabels()}
        seen["offset"] = (ax.xaxis.get_offset_text().get_text(), ax.yaxis.get_offset_text().get_text())
        seen["lines"] = [{"ls": ln.get_linestyle(), "face": ln.get_markerfacecolor(), "marker": ln.get_marker(),
                          "x": [float(v) for v in ln.get_xdata()]} for ln in ax.lines]
        seen["bars"] = [{"hatch": p.get_hatch(), "face": tuple(p.get_facecolor()), "ls": p.get_linestyle()}
                        for p in ax.patches]
        return orig(fig, *a, **k)
    plt.Figure.savefig = spy
    qa = {}
    try:
        render_png(copy.deepcopy(panel), tmp_path / f"{name}_{style}.png", style=style, qa=qa, **kw)
    finally:
        plt.Figure.savefig = orig
    seen["qa"] = qa
    return seen


def _num(text):
    return float(text.replace(",", "").replace("%", ""))


STYLES = ["classic", "enhanced"]
YEARS = [str(y) for y in range(2019, 2024)]


# --------------------------------------------------------------------------
# (1) classic value-axis ticks
# --------------------------------------------------------------------------

class TestValueAxisTicks:

    @pytest.mark.parametrize("style", STYLES)
    def test_no_offset_and_thousands_separators(self, tmp_path, style):
        p = _panel("bar", ["A社", "B社", "C社"], [([1_250_000, 2_400_000, 3_100_000], "emphasis", {})], unit="トン")
        seen = _draw(p, style, tmp_path)
        assert seen["offset"] == ("", "")                          # never '1e6'
        assert any(t == "3,000,000" or t == "2,000,000" for _l, t in seen["yticks"]), seen["yticks"]
        for loc, text in seen["yticks"]:
            assert abs(_num(text) - loc) < 1e-6 * max(1.0, abs(loc)), (loc, text)
            assert abs(loc) < 1000 or "," in text, text

    def test_classic_hbar_and_small_values_get_separators(self, tmp_path):
        p = _panel("hbar", ["A", "B", "C"], [([12_345, 98_000, 5_100], "emphasis", {})], unit="人")
        seen = _draw(p, "classic", tmp_path)
        assert any("," in t for _l, t in seen["xticks"]), seen["xticks"]
        assert seen["offset"] == ("", "")

    def test_classic_keeps_its_tick_size_and_image_size(self, tmp_path):
        p = _panel("bar", ["A社", "B社", "C社"], [([1_250_000, 2_400_000, 3_100_000], "emphasis", {})], unit="トン")
        seen = _draw(p, "classic", tmp_path)
        assert seen["ytick_size"] == {10.0}                       # the v1.5 value tick size (font.size)
        assert seen["qa"]["size_px"] == (1500, 780)

    @pytest.mark.parametrize("style", STYLES)
    def test_years_print_as_years(self, tmp_path, style):
        p = _panel("line_xy", YEARS, [([10, 12, 15, 14, 18], "emphasis", {})])
        seen = _draw(p, style, tmp_path)
        assert {t for _l, t in seen["xticks"]} <= set(YEARS), seen["xticks"]


# --------------------------------------------------------------------------
# (7) embed_text
# --------------------------------------------------------------------------

class TestEmbedText:

    def _two(self):
        return _panel("line_xy", YEARS, [([10, 12, 15, 14, 18], "emphasis", {}), ([8, 9, 9.5, 11, 12], "context", {})],
                      caption="出所: 経産省（公表）", judgment="2023年に最高を更新。")

    @pytest.mark.parametrize("style", STYLES)
    def test_default_draws_title_judgment_and_caption(self, tmp_path, style):
        seen = _draw(self._two(), style, tmp_path)
        assert seen["title"] == "タイトル（億円）"
        joined = "".join(seen["fig_texts"]).replace("\n", "")
        assert "見るべき点" in joined and "2023年に最高を更新。" in joined and "出所: 経産省（公表）" in joined

    @pytest.mark.parametrize("style", STYLES)
    def test_false_leaves_the_text_out_and_keeps_the_legend(self, tmp_path, style):
        seen = _draw(self._two(), style, tmp_path, embed_text=False)
        assert seen["title"] == "" and seen["fig_texts"] == []
        assert seen["legend"] == ["s0", "s1"]                      # the legend is part of the chart
        qa = seen["qa"]
        assert qa["spill"] == [] and qa["label_overlaps"] == 0 and qa["label_on_title"] == 0

    def test_sizes(self, tmp_path):
        with_text = _draw(self._two(), "classic", tmp_path, "a")["qa"]["size_px"]
        without = _draw(self._two(), "classic", tmp_path, "b", embed_text=False)["qa"]["size_px"]
        assert with_text == without == (1500, 780)                  # classic keeps its size
        e_text = _draw(self._two(), "enhanced", tmp_path, "c")["qa"]["size_px"]
        e_bare = _draw(self._two(), "enhanced", tmp_path, "d", embed_text=False)["qa"]["size_px"]
        assert e_text[0] == e_bare[0] == 1100 and e_bare[1] < e_text[1]   # (the text is printed outside)

    def test_the_pipeline_renders_report_figures_without_text(self, tmp_path, monkeypatch):
        pytest.importorskip("matplotlib")
        from deep_research_tool.config import create_config
        from deep_research_tool.evidence.locker import EvidenceLocker
        from deep_research_tool.evidence.numerical_extractor import NumericalDataStore
        from deep_research_tool.main import DeepResearchTool
        from deep_research_tool.tests.test_chart_styles import _LLM, _dp
        import deep_research_tool.chartkit as ck
        calls = []
        real = ck.render_png

        def spy(*a, **k):
            calls.append(k)
            return real(*a, **k)
        monkeypatch.setattr(ck, "render_png", spy)
        cfg = create_config(provider="openai", openai_api_key="t", output_dir=str(tmp_path), auto_figures=True)
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config, tool.llm_client, tool.stage_llm_clients = cfg, _LLM(), {}
        store = NumericalDataStore(research_topic="x")
        for y, v in ((2021, 40), (2022, 50), (2023, 55)):
            store.add(_dp("市場", "市場規模", y, v, "億円", "https://www.stat.go.jp/x", section_id="1"))
        session = SimpleNamespace(query="x", research_plan=None,
                                  section_contents={"1": {"title": "市場規模", "content": "本文", "sources": []}})
        figs = tool._spec_charts(session, EvidenceLocker(research_id="r", output_dir=tmp_path / "ev"),
                                 store, tmp_path / "figures")
        assert len(figs) == 1 and calls and all(k.get("embed_text") is False for k in calls)
        # the report still carries the judgment once, in the figure caption
        assert "見るべき点: 2021年が過去最高。" in figs[0].caption


# --------------------------------------------------------------------------
# (10) wrapping
# --------------------------------------------------------------------------

_CORPUS = [
    "2024年度の売上高は前年比12.5%増の3,456億円、営業利益率は8.2%で、2020年度以降の最高水準を5年連続で更新した。",
    "「工業統計」によると国内出荷は（前年比）増加した。ショッピングモールの来客数は約1.6億人。",
    "Global market size of state-of-the-art EV batteries (2020-2025)",
    "データセンター（DC） 向けGPU出荷 2026年末までに年率換算で1,200万台",
    "出所: 経済産業省「生産動態統計」　各社決算資料　（2024年）",
    "abcdefghi claude-3-opus x  and  gpt-4o-mini or gemini-1.5-pro",
    "  先頭に空白がある説明文、そして末尾にも空白  ",
    "https://www.meti.go.jp/statistics/tyo/seidou/result/ichiran/08_seidou.html の表",
]


class TestWrap:

    def test_lines_never_exceed_the_width_or_start_or_end_with_a_space(self):
        for text in _CORPUS:
            for width in range(4, 31):
                out = RP._wrap(text, width)
                lines = out.split("\n")
                assert all(0 < len(ln) <= width for ln in lines), (text, width, lines)
                assert all(ln == ln.strip(RP._EDGE_SPACE) for ln in lines), (text, width, lines)
                # nothing is lost or reordered (only the spaces at the breaks)
                assert "".join(out.split()) == "".join(text.split()), (text, width)

    def test_japanese_line_breaking_rules(self):
        for text in _CORPUS[:5]:
            for width in range(6, 31):
                lines = RP._wrap(text, width).split("\n")
                for a, b in zip(lines, lines[1:]):
                    assert b[0] not in RP._NO_LINE_START, (text, width, lines)
                    assert a[-1] not in RP._NO_LINE_END, (text, width, lines)

    def test_cjk_breaks_anywhere_latin_at_spaces_and_hyphens(self):
        assert RP._wrap("あいうえおかきくけこさしすせそ", 5) == "あいうえお\nかきくけこ\nさしすせそ"
        assert RP._wrap("出典: " + "あ" * 12, 10) == "出典: " + "あ" * 6 + "\n" + "あ" * 6
        for text, width in [("The quick brown fox jumps over the lazy dog", 10),
                            ("Fuji-Xerox Holdings market share", 12), ("well-known brand-new design", 11)]:
            assert RP._wrap(text, width) == "\n".join(textwrap.wrap(text, width)), text

    def test_numbers_and_latin_words_stay_whole(self):
        text = "国内の出荷は1.6億缶に達し、Holdingsは12,345台を販売した"
        for width in range(9, 30):
            out = RP._wrap(text, width)
            for whole in ("1.6億", "Holdings", "12,345"):
                assert whole in out, (width, out)

    def test_a_word_longer_than_a_line_is_cut_within_the_width(self):
        out = RP._wrap("出所: https://www.example.com/" + "x" * 80, 30)
        assert all(len(ln) <= 30 for ln in out.split("\n"))


# --------------------------------------------------------------------------
# (11) labels off other series' marks
# --------------------------------------------------------------------------

class TestLabelsOffOtherSeries:

    def test_segment_box_geometry(self):
        from matplotlib.transforms import Bbox
        box = Bbox.from_extents(10, 10, 20, 20)
        assert RP._segment_hits((0, 15), (30, 15), box, 0.0)           # straight through
        assert RP._segment_hits((12, 12), (14, 14), box, 0.0)          # inside
        assert not RP._segment_hits((0, 0), (5, 30), box, 0.0)         # passes beside
        assert not RP._segment_hits((0, 25), (30, 25), box, 2.0)       # above, thicker line still clear
        assert RP._segment_hits((0, 21), (30, 21), box, 2.0)           # a thick line grazes it

    @pytest.mark.parametrize("style", STYLES)
    def test_close_lines_labels_move_below_instead_of_sitting_on_the_other_line(self, tmp_path, style):
        p = _panel("line_xy", YEARS, [([20, 21, 22, 23, 24], "emphasis", {}),
                                      ([19.8, 20.8, 21.8, 22.8, 23.8], "context", {})])
        seen = _draw(p, style, tmp_path)
        qa = seen["qa"]
        assert qa["label_on_marks"] == 0 and qa["label_overlaps"] == 0 and qa["spill"] == [], qa
        shown = dict(seen["labels"])
        assert {"20", "21", "22", "23", "24"} <= set(shown)              # the emphasised series keeps all
        moved = [v for v in ("20.8", "21.8", "22.8") if shown.get(v) == "top"]
        assert moved, seen["labels"]                                      # moved under its point, not dropped

    @pytest.mark.parametrize("style", STYLES)
    def test_emphasised_line_keeps_its_labels_and_none_sit_on_marks(self, tmp_path, style):
        p = _panel("line_xy", YEARS, [([10, 11, 12, 13, 14], "context", {}),
                                      ([10.2, 11.2, 12.2, 13.2, 14.2], "emphasis", {}),
                                      ([9.9, 11.1, 12.1, 12.9, 14.1], "context2", {})], unit="%")
        seen = _draw(p, style, tmp_path)
        assert {"10.2%", "11.2%", "12.2%", "13.2%", "14.2%"} <= {t for t, _va in seen["labels"]}
        assert seen["qa"]["label_overlaps"] == 0

    @pytest.mark.parametrize("style", STYLES)
    def test_bar_labels_never_print_over_another_series_bar(self, tmp_path, style):
        labels = [f"P{i}" for i in range(6)]
        p = _panel("bar", labels, [([123_456.7 + 1000 * i for i in range(6)], "context", {}),
                                   ([223_456.7 + 1000 * i for i in range(6)], "emphasis", {}),
                                   ([173_456.7 + 1000 * i for i in range(6)], "context2", {})])
        seen = _draw(p, style, tmp_path)
        qa = seen["qa"]
        assert qa["label_on_marks"] == 0 and qa["label_overlaps"] == 0, qa
        shown = {t for t, _va in seen["labels"]}
        assert {f"{223_456.7 + 1000 * i:,.1f}" for i in range(6)} <= shown


# --------------------------------------------------------------------------
# (12) value axis of a line chart
# --------------------------------------------------------------------------

class TestLineAxis:

    def test_floor_rule(self):
        assert RP._line_floor(1_000_100, 1_000_300) == pytest.approx(1_000_080)
        assert RP._line_floor(40, 60) == pytest.approx(38)
        assert RP._line_floor(5, 60) == 0.0 and RP._line_floor(10, 10) == 0.0 and RP._line_floor(0, 0) == 0.0

    @pytest.mark.parametrize("style", STYLES)
    def test_line_follows_the_data_bars_keep_zero(self, tmp_path, style):
        # a line that varies by ~15% around a large level (a series that
        # varies by less than 1% is rejected as flat by spec.validate)
        big = [1_000_000, 1_040_000, 1_090_000, 1_120_000, 1_150_000]
        lo, hi = _draw(_panel("line_xy", YEARS, [(big, "emphasis", {})], unit="トン"), style, tmp_path,
                       "l")["qa"]["value_lim"]
        assert 0 < lo <= min(big) and hi >= max(big)
        lo, _ = _draw(_panel("line_xy", YEARS, [([5, 20, 40, 50, 60], "emphasis", {})]), style, tmp_path,
                      "z")["qa"]["value_lim"]
        assert lo == 0
        lo, _ = _draw(_panel("bar", ["A", "B", "C"], [([1000, 1100, 1200], "emphasis", {})]), style, tmp_path,
                      "b")["qa"]["value_lim"]
        assert lo == 0
        lo, _ = _draw(_panel("line_xy", YEARS, [([5, -12, 8, 3, -2], "emphasis", {})], unit="%"), style, tmp_path,
                      "n")["qa"]["value_lim"]
        assert lo < -12


# --------------------------------------------------------------------------
# (13) forecasts
# --------------------------------------------------------------------------

class TestForecast:

    def test_flags_from_sources_and_note(self):
        s = {"values": [1, 2, None, 4], "sources": [{}, {"forecast": True}, None, {}]}
        assert RP.forecast_flags(s, ["a", "b", "c", "d"]) == [False, True, False, False]
        s = {"values": [1, 2, 3], "sources": [], "note": "2023: 1.2億ドル を 億円に換算；2024: 予測値"}
        assert RP.forecast_flags(s, ["2022", "2023", "2024"]) == [False, False, True]
        xy = {"values": [{"x": 2023.0, "y": 1}, {"x": 2024.0, "y": 2}], "note": "2024: 予測値"}
        assert RP.forecast_flags(xy) == [False, True]
        # free text never marks values (review: a quoted source sentence
        # mentioning a forecast turned actual values into forecasts)
        assert RP.forecast_flags({"values": [1, 2], "note": "予測値"}, ["a", "b"]) == [False, False]
        s = {"values": [1, 2], "sources": [{"forecast": False}, {"forecast": False}],
             "note": "2023: 2023年の実績は110,000百万円; 2030年の予測値は300,000百万円 を 億に換算"}
        assert RP.forecast_flags(s, ["2023", "2024"]) == [False, False]
        # the source's own flag wins over a note
        assert RP.forecast_flags({"values": [1], "sources": [{"forecast": False}], "note": "a: 予測値"},
                                 ["a"]) == [False]
        assert RP.forecast_flags({"values": [1, 2], "note": "2023: 換算"}, ["a", "b"]) == [False, False]

    @pytest.mark.parametrize("style", STYLES)
    def test_line_forecast_is_dashed_and_hollow_actual_series_are_solid(self, tmp_path, style):
        p = _panel("line_xy", YEARS, [([10, 12, 15, 17, 19], "emphasis", {"forecast": [False] * 3 + [True] * 2}),
                                      ([8, 9, 9.5, 11, 12], "context", {})])
        seen = _draw(p, style, tmp_path)
        segs = [ln for ln in seen["lines"] if ln["ls"] in ("-", "--")]
        dashed = [ln for ln in segs if ln["ls"] == "--"]
        assert len(dashed) == 1 and dashed[0]["x"] == [2021.0, 2022.0, 2023.0]   # into and through the forecast
        assert sum(1 for ln in segs if ln["ls"] == "-") == 2                       # actuals of both series: solid
        hollow = [ln for ln in seen["lines"] if ln["face"] == "white"]
        assert [x for ln in hollow for x in ln["x"]] == [2022.0, 2023.0]
        assert RP.FORECAST_NOTE in seen["legend"] and RP.STRIPE_NOTE not in seen["legend"]

    @pytest.mark.parametrize("style", STYLES)
    def test_line_note_names_hollow_points_not_stripes(self, tmp_path, style):
        p = _panel("line_xy", YEARS, [([10, 12, 15, 17, 19], "emphasis", {"basis": [PUB, "二次", PUB, PUB, PUB]})])
        seen = _draw(p, style, tmp_path)
        assert seen["legend"] == [RP.HOLLOW_NOTE]                  # one series: only the note
        assert [x for ln in seen["lines"] if ln["face"] == "white" for x in ln["x"]] == [2020.0]

    @pytest.mark.parametrize("style", STYLES)
    def test_bar_forecast_has_its_own_fill(self, tmp_path, style):
        p = _panel("bar", ["2021", "2022", "2023", "2024"],
                   [([10, 12, 15, 17], "emphasis", {"basis": [PUB, "推定", PUB, PUB],
                                                    "forecast": [False, False, False, True]})])
        seen = _draw(p, style, tmp_path)
        hatches = [b["hatch"] for b in seen["bars"]]
        assert hatches == [None, "////", None, RP.FORECAST_HATCH]
        fc = seen["bars"][3]
        assert fc["face"] == (1.0, 1.0, 1.0, 1.0) and fc["ls"] != "solid"
        assert seen["legend"] == [RP.STRIPE_NOTE, RP.FORECAST_NOTE]

    def test_note_marks_a_forecast_when_the_source_does_not(self, tmp_path):
        p = _panel("line_xy", YEARS, [([10, 12, 15, 17, 19], "emphasis", {}), ([8, 9, 9.5, 11, 12], "context", {})])
        p["series"][0]["note"] = "2023: 予測値"
        p["series"][0]["sources"] = []          # no per-point source to say otherwise
        seen = _draw(p, "classic", tmp_path)
        assert [ln["x"] for ln in seen["lines"] if ln["ls"] == "--"] == [[2022.0, 2023.0]]


# --------------------------------------------------------------------------
# (16) legend
# --------------------------------------------------------------------------

class TestLegend:

    @pytest.mark.parametrize("style", STYLES)
    def test_one_series_has_no_legend_box(self, tmp_path, style):
        for form in ("bar", "hbar", "line_xy"):
            p = _panel(form, YEARS, [([10, 12, 15, 14, 18], "emphasis", {})])
            assert _draw(p, style, tmp_path, form)["legend"] == [], form

    @pytest.mark.parametrize("style", STYLES)
    def test_several_series_keep_their_legend(self, tmp_path, style):
        p = _panel("bar", ["A", "B", "C"], [([10, 12, 15], "emphasis", {}), ([8, 9, 11], "context", {})])
        assert _draw(p, style, tmp_path)["legend"] == ["s0", "s1"]

    def test_classic_size_without_the_legend(self, tmp_path):
        hbar = _panel("hbar", list("ABCDE"), [([5, 4, 3, 2, 1], "emphasis", {})])
        assert _draw(hbar, "classic", tmp_path)["qa"]["size_px"] == (1500, round(max(3.2, 0.45 * 5 + 2.0) * 150))


# --------------------------------------------------------------------------
# (17) fonts
# --------------------------------------------------------------------------

class TestFonts:

    def _pick(self, monkeypatch, names, style="classic"):
        monkeypatch.setattr(RP, "_font_cache", {})
        monkeypatch.setattr(RP, "FONT_STATE", {"japanese_missing": False})
        plt = SimpleNamespace(rcParams={})
        fm = SimpleNamespace(fontManager=SimpleNamespace(ttflist=[SimpleNamespace(name=n) for n in names]))
        return RP.setup_japanese_font(plt, fm, style), plt.rcParams

    def test_proportional_before_fixed_pitch_in_both_styles(self, monkeypatch):
        for style in STYLES:
            name, rc = self._pick(monkeypatch, ["IPAGothic", "IPAPGothic", "DejaVu Sans"], style)
            assert name == "IPAPGothic" and rc["font.family"] == "IPAPGothic"
            assert self._pick(monkeypatch, ["IPAGothic", "BIZ UDPGothic", "IPAexGothic"], style)[0] == "BIZ UDPGothic"
            assert self._pick(monkeypatch, ["IPAGothic", "Noto Sans CJK JP"], style)[0] == "Noto Sans CJK JP"
            assert self._pick(monkeypatch, ["IPAGothic", "DejaVu Sans"], style)[0] == "IPAGothic"   # last resort

    def test_a_mono_variant_only_when_alone(self, monkeypatch):
        assert self._pick(monkeypatch, ["WenQuanYi Zen Hei Mono", "WenQuanYi Zen Hei"])[0] == "WenQuanYi Zen Hei"
        assert self._pick(monkeypatch, ["DejaVu Sans"])[0] is None


# --------------------------------------------------------------------------
# HTML page (prelude.js / render_html)
# --------------------------------------------------------------------------

PRELUDE = Path(RP.__file__).resolve().parent / "prelude.js"


class TestHtml:

    def test_spec_json_carries_forecast_flags(self):
        from deep_research_tool.chartkit.render_html import _spec_json
        # a single-series bar needs 3 categories (spec.validate)
        p = _panel("bar", ["2022", "2023", "2024"],
                   [([9, 10, 12], "emphasis", {"forecast": [False, False, True]})])
        spec = json.loads(_spec_json([p]))
        assert spec[0]["series"][0]["forecast"] == [False, False, True]

    def test_prelude_prints_the_same_notes_as_the_png(self):
        js = PRELUDE.read_text(encoding="utf-8")
        for note in (RP.STRIPE_NOTE, RP.HOLLOW_NOTE, RP.FORECAST_NOTE):
            assert note in js
        assert "beginAtZero: lineFromZero(p)" in js
        assert re.search(r"if \(line\) \{.*?\} else \{[^}]*maxBarThickness", js, re.S)

    def _run(self, tmp_path, panels):
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed")
        from deep_research_tool.chartkit.render_html import _spec_json
        harness = """
const captured = [], elements = {};
globalThis.matchMedia = () => ({ matches: false });
globalThis.document = {
  getElementById: (id) => (elements[id] = elements[id] || { innerHTML: '' }),
  createElement: () => ({ getContext: () => ({ fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {},
    stroke() {}, arc() {}, fill() {}, createPattern: () => 'pattern' }) }),
};
globalThis.Chart = function (el, cfg) { captured.push(cfg); };
Chart.defaults = { font: {} };
const SPEC = %s;
%s
console.log(JSON.stringify(captured.map((cfg, n) => ({
  num: [223456.7, 0.04, 2707, 12.34].map(num),
  type: cfg.type, y: cfg.options.scales.y, plugins: cfg.plugins.map(p => p.id),
  legend: elements['lg_' + SPEC[n].id].innerHTML,
  datasets: cfg.data.datasets.map(d => {
    const o = {};
    for (const [k, v] of Object.entries(d)) if (typeof v !== 'function' && k !== 'segment') o[k] = v;
    if (d.segment) o.dash = d.data.slice(1).map((_, j) => d.segment.borderDash({ p0DataIndex: j, p1DataIndex: j + 1 }) || null);
    return o;
  }) }))));
""" % (_spec_json(panels), PRELUDE.read_text(encoding="utf-8"))
        script = tmp_path / "harness.js"
        script.write_text(harness, encoding="utf-8")
        res = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout)

    def test_prelude_datasets_axis_and_legend(self, tmp_path):
        line = _panel("line_xy", YEARS, [([1_000_100, 1_000_150, 1_000_220, 1_000_260, 1_000_300], "emphasis",
                                          {"forecast": [False, False, False, True, True]}),
                                         ([1_000_000, 1_000_050, 1_000_120, 1_000_160, 1_000_200], "context",
                                          {"basis": [PUB, "二次", PUB, PUB, PUB]})], unit="トン")
        line["id"] = "ln"
        near0 = _panel("line_xy", YEARS, [([5, 20, 40, 50, 60], "emphasis", {})])
        near0["id"] = "z"
        bar = _panel("bar", ["A", "B", "C"], [([10, 12, 15], "emphasis", {"forecast": [False, False, True]})])
        bar["id"] = "b"
        out = self._run(tmp_path, [line, near0, bar])
        ln, z, b = out
        # value labels read as in the PNG (separators, 0.04 not 0.0)
        assert ln["num"] == [RP._fmt(v, "億円") for v in (223456.7, 0.04, 2707, 12.34)]
        # line datasets: no bar sizing; actual series solid, forecast segments dashed, hollow points
        for d in ln["datasets"]:
            assert "maxBarThickness" not in d and "barPercentage" not in d
            assert "borderDash" not in d
        assert ln["datasets"][0]["dash"] == [None, None, [6, 4], [6, 4]]
        assert ln["datasets"][1]["dash"] == [None, None, None, None]
        assert ln["datasets"][0]["pointBackgroundColor"][3] != ln["datasets"][0]["borderColor"]
        assert ln["datasets"][1]["pointBackgroundColor"][1] != ln["datasets"][1]["borderColor"]
        assert ln["y"]["beginAtZero"] is False and z["y"]["beginAtZero"] is True
        assert ln["legend"].count('class="sw"') == 2
        assert "白抜き＝換算・導出・推定・二次" in ln["legend"] and "破線・白抜き＝予測値" in ln["legend"]
        assert "斜線" not in ln["legend"]
        # one series: no legend box
        assert z["legend"] == ""
        # bar datasets: bar sizing; a forecast bar gets its own fill and the dashed outline
        d = b["datasets"][0]
        assert d["maxBarThickness"] == 24 and 0 < d["barPercentage"] <= 1 and 0 < d["categoryPercentage"] <= 1
        assert "fco" in b["plugins"] and "vl" in b["plugins"]
        assert 'class="sw"' not in b["legend"] and "破線・白抜き＝予測値" in b["legend"]
