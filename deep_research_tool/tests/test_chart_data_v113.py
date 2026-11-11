"""
v1.13.0 chart data fixes (the data group):

(2) display names — canon() only groups; every name a reader or the
    chart-design LLM sees keeps the sources' most frequent spelling
    ('Toray', 'GPa'), deterministically;
(3) entity aliases — legal forms, 「」, katakana / Latin spellings of one
    company, and prefix aliases whose values agree fold into one entity;
    A社 / B社 / C社 stay apart;
(4) metric keys — unit / note parentheses are dropped from the key, scope
    parentheses (国内 / 世界 / 連結 …) are not, common English metric names
    map to Japanese (display keeps the original);
(5) period granularity — annual and quarterly / half-year / monthly values
    are never one series; the extractor keeps an LLM's '2024Q1';
(6) informativeness — a line needs 3 points, a single-series bar 3
    categories, and line_xy gets the flat / single-value checks too.

No paid API, no network: fake LLMs only.
"""

import json
from types import SimpleNamespace

import pytest

from deep_research_tool.chartkit import build_section_catalog, validate_panels
from deep_research_tool.chartkit.data_layer import (entity_base_key, metric_key, parse_period,
                                                    parse_unit, quarter_of,
                                                    resolve_entity_aliases, romanise,
                                                    unit_spelling)
from deep_research_tool.chartkit.facts import judgment_conflict, panel_facts
from deep_research_tool.chartkit.spec import STYLE_GUIDE_EN, STYLE_GUIDE_JA, SpecError, resolve, validate
from deep_research_tool.evidence.numerical_extractor import (NumericalDataExtractor,
                                                             NumericalDataPoint)


def _dp(subject, metric, year, value, unit, url="https://example.com/a", section_id="1", **kw):
    kw.setdefault("raw_text", f"{value}{unit}")
    return NumericalDataPoint(value=value, unit=unit, metric_name=metric, subject=subject, year=year,
                              source_url=url, source_title=url.split("/")[2] if url else "",
                              section_id=section_id, **kw)


def _times(cat):
    return [s for s in cat.series.values() if s.kind == "time"]


def _cmps(cat):
    return [s for s in cat.series.values() if s.kind == "comparison"]


# --------------------------------------------------------------------------
# (2) display names
# --------------------------------------------------------------------------

class TestDisplayNames:

    def test_labels_and_names_keep_the_most_frequent_spelling(self):
        pts = [_dp("TORAY", "Revenue", 2022, 100, "億円"),
               _dp("Toray", "revenue", 2023, 110, "億円"),
               _dp("Toray", "Revenue", 2024, 120, "億円"),
               _dp("teijin", "Revenue", 2024, 80, "億円"),
               _dp("Teijin", "Revenue", 2023, 70, "億円"),
               _dp("Teijin", "Revenue", 2022, 60, "億円")]
        cat = build_section_catalog(pts, "1")
        names = sorted(s.name for s in _times(cat))
        # 'Toray' twice beats 'TORAY' once; 'Teijin' twice beats 'teijin'
        assert names == ["Teijin Revenue", "Toray Revenue"]
        assert {s.entity for s in _times(cat)} == {"Toray", "Teijin"}
        for c in _cmps(cat):
            assert cat.labels[c.labels_ref] == ["Toray", "Teijin"]
            assert c.metric == "Revenue" and c.name.startswith("Revenue ")
        # the LLM's catalog shows the same spellings, never canon()'s
        items = cat.catalog_for_prompt()
        shown = json.dumps(items, ensure_ascii=False)
        assert "toray" not in shown and "Toray" in shown and "Teijin" in shown

    def test_a_tie_goes_to_the_first_spelling_and_label_arrays_are_shared(self):
        pts = [_dp("nxp", "売上高", 2023, 50, "億ドル"), _dp("NXP", "売上高", 2024, 55, "億ドル"),
               _dp("Infineon", "売上高", 2023, 40, "億ドル"), _dp("Infineon", "売上高", 2024, 45, "億ドル")]
        cat = build_section_catalog(pts, "1")
        cmps = _cmps(cat)
        assert len(cmps) == 2
        # one spelling per key for the whole chapter: both years' arrays are
        # identical, so both comparisons share one labels_ref
        assert cat.labels[cmps[0].labels_ref] == ["nxp", "Infineon"]
        assert cmps[0].labels_ref == cmps[1].labels_ref
        # the same input in another order: same spelling choice rule
        cat2 = build_section_catalog(list(reversed(pts)), "1")
        assert cat2.labels[_cmps(cat2)[0].labels_ref] == ["NXP", "Infineon"]

    def test_units_keep_their_spelling_and_one_spelling_per_unit(self):
        assert parse_unit("GPa") == ("gpa", 1.0, "")          # the key stays lower case
        assert unit_spelling("GPa") == "GPa" and unit_spelling("MW") == "MW"
        assert unit_spelling("百万kWh") == "kWh" and unit_spelling("TWh") == "TWh"
        assert unit_spelling("billion USD") == "ドル" and unit_spelling("％") == "%"
        pts = [_dp("SUS304", "引張強度", y, v, "GPa") for y, v in ((2021, 0.5), (2022, 0.52), (2023, 0.55))]
        pts += [_dp("SUS316", "引張強度", y, v, u) for y, v, u in
                ((2021, 0.48, "GPa"), (2022, 0.5, "gpa"), (2023, 0.53, "GPa"))]
        pts += [_dp("風力", "発電量", y, v, "TWh") for y, v in ((2021, 10), (2022, 12), (2023, 15))]
        pts += [_dp("太陽光", "設備容量", y, v, "MW") for y, v in ((2021, 100), (2022, 150), (2023, 210))]
        cat = build_section_catalog(pts, "1")
        units = {s.name: s.unit for s in cat.series.values()}
        assert units["SUS304 引張強度"] == "GPa" and units["SUS316 引張強度"] == "GPa"
        assert units["風力 発電量"] == "TWh" and units["太陽光 設備容量"] == "MW"
        # the two GPa series are one unit for the panel check
        ts = {s.name: s for s in _times(cat)}
        data = cat.to_data()
        p = {"id": "p", "form": "bar", "title": "引張強度（GPa）", "labels_ref": ts["SUS304 引張強度"].labels_ref,
             "series": [{"name": "SUS304", "values_ref": ts["SUS304 引張強度"].key, "role": "emphasis"},
                        {"name": "SUS316", "values_ref": ts["SUS316 引張強度"].key, "role": "context"}],
             "caption": "c", "judgment": "j"}
        valid, rej = validate_panels([p], data)
        assert valid and not rej and valid[0]["unit"] == "GPa"

    def test_facts_print_the_label_as_shown(self):
        panel = {"form": "bar", "unit": "億円", "labels": ["Toray", "Teijin", "Mitsubishi"],
                 "series": [{"values": [300, 200, 150], "sources": []}]}
        facts = panel_facts(panel)
        why = judgment_conflict("teijinが最大。", facts)       # the LLM wrote another case
        assert why == "「Teijin」が最大とされていますが、最大は「Toray」です"
        # a label with its legal form is named without it
        panel["labels"] = ["トヨタ自動車株式会社", "本田技研工業", "日産自動車"]
        why = judgment_conflict("日産自動車が最大。", panel_facts(panel))
        assert why and "「トヨタ自動車株式会社」" in why
        why = judgment_conflict("本田技研工業が最大。", panel_facts(
            dict(panel, series=[{"values": [100, 300, 150], "sources": []}])))
        assert why is None
        assert judgment_conflict("トヨタ自動車が最小。", panel_facts(panel)).startswith("「トヨタ自動車株式会社」")


# --------------------------------------------------------------------------
# (3) entity aliases
# --------------------------------------------------------------------------

class TestEntityAliases:

    def test_legal_forms_and_brackets_fold(self):
        same = ["トヨタ自動車株式会社", "株式会社トヨタ自動車", "トヨタ自動車(株)", "㈱トヨタ自動車",
                "（株）トヨタ自動車", "「トヨタ自動車」", "『トヨタ自動車』", "トヨタ自動車"]
        assert {entity_base_key(s) for s in same} == {"トヨタ自動車"}
        latin = ["Toyota Motor Corp.", "Toyota Motor Corporation", "TOYOTA MOTOR", "Toyota Motor Co., Ltd.",
                 "Toyota Motor Inc", "Toyota Motor LLC"]
        assert {entity_base_key(s) for s in latin} == {"toyota motor"}
        # a group / holding company is NOT its namesake company (review:
        # ソフトバンクグループ ≠ ソフトバンク, Volkswagen Group ≠ Volkswagen)
        assert entity_base_key("Volkswagen Group") != entity_base_key("Volkswagen")
        assert entity_base_key("ソフトバンクグループ") != entity_base_key("ソフトバンク株式会社")
        assert entity_base_key("Panasonic Holdings") != entity_base_key("Panasonic Corporation")
        assert entity_base_key("Siemens AG") == "siemens" and entity_base_key("Airbus SA") == "airbus"
        assert entity_base_key("Shell PLC") == "shell" and entity_base_key("Acme Ltd.") == "acme"
        assert entity_base_key("ソニーグループ株式会社") == "ソニーグループ"
        assert entity_base_key("有限会社ヤマダ") == "ヤマダ"
        assert entity_base_key("Mitsui & Co.") == "mitsui"
        # never a bare 社, never a product's 'HD', never down to nothing
        assert [entity_base_key(s) for s in ("A社", "B社", "C社")] == ["a社", "b社", "c社"]
        assert entity_base_key("Ultra HD") == "ultra hd"
        assert entity_base_key("グループ") == "グループ" and entity_base_key("Group") == "group"

    def test_katakana_and_latin_names_are_one_entity(self):
        assert romanise("トヨタ") == "toyota" and romanise("ソニー") == "sonii"
        assert romanise("パナソニック") == "panasonikku" and romanise("トヨタ自動車") is None
        pairs = [("トヨタ", "Toyota"), ("ソニー", "Sony"), ("パナソニック", "Panasonic"), ("テスラ", "Tesla")]
        pts = []
        for i, (kana, latin) in enumerate(pairs):
            base = 100 * (i + 1)
            pts += [_dp(kana, "売上高", 2022, base, "億円"), _dp(latin, "売上高", 2023, base * 1.1, "億円"),
                    _dp(kana, "売上高", 2024, base * 1.2, "億円")]
        cat = build_section_catalog(pts, "1")
        ts = _times(cat)
        # each company is ONE three-year series under its most frequent spelling
        assert sorted(s.entity for s in ts) == ["ソニー", "テスラ", "トヨタ", "パナソニック"]
        assert all(cat.labels[s.labels_ref] == ["2022", "2023", "2024"] for s in ts)
        # ... and a comparison holds four companies, not eight
        assert all(len(cat.labels[c.labels_ref]) == 4 for c in _cmps(cat))

    def test_katakana_and_latin_must_really_match(self):
        cells = {"ソニー": {}, "suni": {}, "トヨタ": {}, "tayota": {}, "ホンダ": {}, "honda": {}}
        alias = resolve_entity_aliases(cells)
        assert alias["ソニー"] != alias["suni"] and alias["トヨタ"] != alias["tayota"]
        assert alias["ホンダ"] == alias["honda"]

    def test_prefix_aliases_merge_only_when_the_values_agree(self):
        pts = [_dp("トヨタ", "売上高", 2023, 450000, "億円"), _dp("トヨタ", "売上高", 2024, 480000, "億円"),
               _dp("トヨタ自動車", "売上高", 2024, 460000, "億円"),           # within 15 %
               _dp("トヨタ自動車", "売上高", 2023, 455000, "億円"),           # two shared amounts
               _dp("トヨタ自動車", "売上高", 2022, 370000, "億円"),
               _dp("Toyota", "営業利益", 2023, 50000, "億円"), _dp("Toyota", "営業利益", 2024, 53000, "億円"),
               _dp("Toyota Tsusho", "営業利益", 2023, 3500, "億円"),         # a different company
               _dp("Toyota Tsusho", "営業利益", 2024, 3600, "億円")]
        cat = build_section_catalog(pts, "1")
        sales = [s for s in _times(cat) if s.metric == "売上高"]
        assert len(sales) == 1 and cat.labels[sales[0].labels_ref] == ["2022", "2023", "2024"]
        profit = sorted(s.entity for s in _times(cat) if s.metric == "営業利益")
        # トヨタ = Toyota (katakana / Latin), so the operating-profit series
        # is named after the merged entity's most frequent spelling
        assert len(profit) == 2 and "Toyota Tsusho" in profit

    def test_prefix_alias_without_a_common_period_is_not_merged(self):
        cells = {"トヨタ": {("売上高", "円", "annual", "2023"): 100.0},
                 "トヨタ自動車": {("売上高", "円", "annual", "2024"): 101.0}}
        alias = resolve_entity_aliases(cells)
        assert alias["トヨタ"] != alias["トヨタ自動車"]
        cells["トヨタ自動車"][("売上高", "円", "annual", "2023")] = 300.0     # far apart
        alias = resolve_entity_aliases(cells)
        assert alias["トヨタ"] != alias["トヨタ自動車"]
        cells["トヨタ自動車"][("売上高", "円", "annual", "2023")] = 110.0     # close, but one amount
        alias = resolve_entity_aliases(cells)
        assert alias["トヨタ"] != alias["トヨタ自動車"]              # one shared amount needs 3 %
        cells["トヨタ"][("売上高", "円", "annual", "2024")] = 105.0       # a second one, close
        alias = resolve_entity_aliases(cells)
        assert alias["トヨタ"] == alias["トヨタ自動車"]
        # rates are no evidence, and a complement is never an alias
        cells = {"米国": {("シェア", "%", "annual", "2023"): 49.0, ("シェア", "%", "annual", "2024"): 52.0},
                 "米国以外": {("シェア", "%", "annual", "2023"): 51.0, ("シェア", "%", "annual", "2024"): 48.0}}
        assert len(set(resolve_entity_aliases(cells).values())) == 2
        # a short name with two candidates joins neither (no chaining)
        cells = {"トヨタ": {("売上高", "円", "annual", "2023"): 45.0, ("売上高", "円", "annual", "2024"): 48.0},
                 "トヨタ自動車": {("売上高", "円", "annual", "2023"): 45.1, ("売上高", "円", "annual", "2024"): 48.1},
                 "トヨタ紡織": {("売上高", "円", "annual", "2023"): 45.2, ("売上高", "円", "annual", "2024"): 48.2}}
        assert len(set(resolve_entity_aliases(cells).values())) == 3
        # 'Toyota' / 'Toyota Motor' (word boundary) and katakana → Latin prefix
        cells = {"toyota": {("x", "", "annual", "2024"): 10.0, ("x", "", "annual", "2023"): 9.0},
                 "toyota motor": {("x", "", "annual", "2024"): 10.5, ("x", "", "annual", "2023"): 9.2},
                 "sony": {("x", "", "annual", "2024"): 1.0},
                 "sonya": {("x", "", "annual", "2024"): 1.0}}
        alias = resolve_entity_aliases(cells)
        assert alias["toyota"] == alias["toyota motor"] and alias["sony"] != alias["sonya"]
        cells = {"トヨタ": {("x", "", "annual", "2024"): 10.0, ("x", "", "annual", "2023"): 9.0},
                 "toyota motor": {("x", "", "annual", "2024"): 10.5, ("x", "", "annual", "2023"): 9.2}}
        alias = resolve_entity_aliases(cells)
        assert alias["トヨタ"] == alias["toyota motor"]

    def test_a_b_c_companies_stay_apart(self):
        pts = [_dp(e, "シェア", 2024, v, "%") for e, v in (("A社", 40), ("B社", 35), ("C社", 25))]
        pts += [_dp(e, "シェア", 2023, v, "%") for e, v in (("A社", 38), ("B社", 36), ("C社", 26))]
        cat = build_section_catalog(pts, "1")
        for c in _cmps(cat):
            assert cat.labels[c.labels_ref] == ["A社", "B社", "C社"]

    def test_merged_names_are_deterministic(self):
        pts = [_dp("トヨタ自動車株式会社", "売上高", 2023, 100, "億円"),
               _dp("トヨタ自動車", "売上高", 2024, 105, "億円"),
               _dp("トヨタ自動車", "売上高", 2022, 95, "億円"),
               _dp("ホンダ", "売上高", 2024, 80, "億円"), _dp("Honda", "売上高", 2023, 78, "億円"),
               _dp("Honda", "売上高", 2022, 75, "億円")]
        a = build_section_catalog(pts, "1")
        b = build_section_catalog(list(reversed(pts)), "1")
        names_a = sorted(s.name for s in _times(a))
        assert names_a == sorted(s.name for s in _times(b)) == ["Honda 売上高", "トヨタ自動車 売上高"]


# --------------------------------------------------------------------------
# (4) metric keys
# --------------------------------------------------------------------------

class TestMetricKeys:

    def test_metric_key(self):
        assert metric_key("売上高（億円）") == metric_key("売上高") == metric_key("Revenue") == "売上高"
        assert metric_key("Sales (USD bn)") == metric_key("売上") == "売上高"
        assert metric_key("市場規模（推計）") == metric_key("Market Size (2024)") == "市場規模"
        assert metric_key("市場規模（国内）") == metric_key("国内市場規模") == metric_key("Market size (domestic)")
        assert metric_key("市場規模（世界）") == metric_key("Global market size") == metric_key("市場規模（グローバル）")
        assert metric_key("市場規模（国内）") != metric_key("市場規模（世界）")
        assert metric_key("売上高（連結）") == metric_key("Revenue (consolidated)") == metric_key("連結売上高")
        assert metric_key("売上高（連結）") != metric_key("売上高（単体）") != metric_key("売上高")
        assert metric_key("CAGR（2024-2030）") != metric_key("CAGR（2023-2028）")
        assert metric_key("Market share (%)") == metric_key("シェア")
        for en, ja in (("operating profit", "営業利益"), ("net income", "純利益"), ("shipments", "出荷量"),
                       ("production", "生産量"), ("growth rate", "成長率"), ("price", "価格"),
                       ("capacity", "生産能力")):
            assert metric_key(en) == metric_key(ja), en
        # unrelated words are left alone
        assert metric_key("User count") == "user count" and metric_key("日本酒消費量") == "日本酒消費量"
        assert metric_key("") == ""
        # a name that is nothing but a bracket is not lost ('【売上高】')
        assert metric_key("【売上高】") == "売上高" and metric_key("(国内)") == "国内"

    def test_domestic_and_global_never_share_a_line(self):
        pts = [_dp("EV", "市場規模（国内）", y, v, "億円") for y, v in ((2022, 10), (2023, 12), (2024, 15))]
        pts += [_dp("EV", "市場規模（世界）", y, v, "億円") for y, v in ((2022, 900), (2023, 1100), (2024, 1300))]
        cat = build_section_catalog(pts, "1")
        ts = sorted(_times(cat), key=lambda s: s.values[0])
        assert len(ts) == 2
        assert ts[0].values == [10.0, 12.0, 15.0] and ts[1].values == [900.0, 1100.0, 1300.0]
        assert ts[0].name == "EV 市場規模（国内）" and ts[1].name == "EV 市場規模（世界）"

    def test_english_and_japanese_names_join_and_the_display_keeps_the_original(self):
        pts = [_dp("EV", "Market size", 2022, 10, "億円"), _dp("EV", "市場規模", 2023, 12, "億円"),
               _dp("EV", "市場規模", 2024, 15, "億円"), _dp("EV", "市場規模（億円）", 2021, 9, "億円")]
        cat = build_section_catalog(pts, "1")
        ts = _times(cat)
        assert len(ts) == 1 and cat.labels[ts[0].labels_ref] == ["2021", "2022", "2023", "2024"]
        assert ts[0].metric == "市場規模" and ts[0].name == "EV 市場規模"


# --------------------------------------------------------------------------
# (5) period granularity
# --------------------------------------------------------------------------

class TestGranularity:

    def test_parse_period(self):
        assert parse_period("2024Q1") == ("quarter", 1) and parse_period("FY2024 Q2") == ("quarter", 2)
        assert parse_period("2024年第3四半期") == ("quarter", 3) and parse_period("第四四半期") == ("quarter", 4)
        assert parse_period("1Q24") == ("quarter", 1) and parse_period("2024年4-6月期") == ("quarter", 2)
        assert parse_period("first quarter of 2024") == ("quarter", 1)
        assert parse_period("上期") == ("half", 1) and parse_period("下半期") == ("half", 2)
        assert parse_period("2024H2") == ("half", 2) and parse_period("1H FY2024") == ("half", 1)
        assert parse_period("2024/03") == ("month", 3) and parse_period("2024-11") == ("month", 11)
        assert parse_period("2024年3月") == ("month", 3) and parse_period("Mar 2024") == ("month", 3)
        for annual in ("2024", "FY2024", "2024年度", "2024年3月期", "2024年3月決算", "2020-2024",
                       "2024/25", "FY ending March 2024", "", None):
            assert parse_period(annual) == ("annual", None), annual
        # a sub-annual period that cannot be placed
        assert parse_period("四半期") == ("quarter", None)
        assert parse_period("第3四半期累計") == ("quarter", None)
        assert parse_period("4-9月期") == ("half", None)
        assert quarter_of("2024年第1四半期") == "Q1" and quarter_of("2024") is None
        assert quarter_of("2024H1") is None

    def test_annual_and_quarterly_values_are_different_series(self):
        pts = [_dp("Toray", "売上高", y, v, "億円") for y, v in ((2021, 400), (2022, 420), (2023, 450))]
        pts += [_dp("Toray", "売上高", 2024, 110, "億円", quarter="Q1"),
                _dp("Toray", "売上高", 2024, 115, "億円", date_context="2024年第2四半期"),
                _dp("Toray", "売上高", 2024, 120, "億円", date_context="FY2024 Q3")]
        cat = build_section_catalog(pts, "1")
        ts = {tuple(cat.labels[s.labels_ref]): s for s in _times(cat)}
        assert set(ts) == {("2021", "2022", "2023"), ("2024Q1", "2024Q2", "2024Q3")}
        assert ts[("2021", "2022", "2023")].values == [400.0, 420.0, 450.0]
        assert ts[("2024Q1", "2024Q2", "2024Q3")].values == [110.0, 115.0, 120.0]
        assert ts[("2024Q1", "2024Q2", "2024Q3")].name == "Toray 売上高（四半期）"
        assert ts[("2021", "2022", "2023")].name == "Toray 売上高"
        assert cat.stats["granularity_split"] == 1
        # the quarterly value of 2024 never sits on the annual axis as '2024'
        assert all("2024" not in cat.labels[s.labels_ref] for s in cat.series.values())

    def test_half_year_and_monthly_series(self):
        pts = [_dp("X", "出荷量", 2023, 10, "万台", date_context="2023年上期"),
               _dp("X", "出荷量", 2023, 12, "万台", date_context="2023年下期"),
               _dp("X", "出荷量", 2024, 11, "万台", date_context="2024H1"),
               _dp("Y", "出荷量", 2024, 3, "万台", date_context="2024/01"),
               _dp("Y", "出荷量", 2024, 4, "万台", date_context="2024年2月"),
               _dp("Y", "出荷量", 2024, 5, "万台", date_context="Mar 2024")]
        cat = build_section_catalog(pts, "1")
        labels = sorted(tuple(cat.labels[s.labels_ref]) for s in _times(cat))
        assert labels == [("2023H1", "2023H2", "2024H1"), ("2024-01", "2024-02", "2024-03")]
        names = sorted(s.name for s in _times(cat))
        assert names == ["X 出荷量（半期）", "Y 出荷量（月次）"]

    def test_a_period_that_cannot_be_placed_is_dropped_with_a_reason(self):
        pts = [_dp("X", "売上高", 2024, 10, "億円", date_context="2024年 四半期"),
               _dp("X", "売上高", 2024, 30, "億円", date_context="2024年第3四半期累計"),
               _dp("X", "売上高", 2022, 40, "億円"), _dp("X", "売上高", 2023, 45, "億円")]
        cat = build_section_catalog(pts, "1")
        assert cat.stats["period_unknown"] == 2
        assert any("時期が特定できない" in d for d in cat.dropped)
        assert [cat.labels[s.labels_ref] for s in _times(cat)] == [["2022", "2023"]]

    def test_the_extractor_keeps_the_quarter(self):
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                items = [
                    {"value": 1, "raw_text": "1億円", "unit": "億円", "metric_name": "売上高", "subject": "A社",
                     "year": "2024Q1", "confidence": 0.9},
                    {"value": 2, "raw_text": "2億円", "unit": "億円", "metric_name": "売上高", "subject": "A社",
                     "year": "2024年第1四半期", "confidence": 0.9},
                    {"value": 3, "raw_text": "3億円", "unit": "億円", "metric_name": "売上高", "subject": "A社",
                     "year": "FY2024 Q2", "confidence": 0.9},
                    {"value": 4, "raw_text": "4億円", "unit": "億円", "metric_name": "売上高", "subject": "A社",
                     "year": 2023, "confidence": 0.9},
                    {"value": 5, "raw_text": "5億円", "unit": "億円", "metric_name": "売上高", "subject": "A社",
                     "year": 2024, "period": "第3四半期", "confidence": 0.9},
                    {"value": 6, "raw_text": "6億円", "unit": "億円", "metric_name": "売上高", "subject": "A社",
                     "year": "2024H1", "confidence": 0.9},
                ]
                return SimpleNamespace(content=json.dumps(items, ensure_ascii=False))
        llm = _LLM()
        ex = NumericalDataExtractor(llm_client=llm, language="ja", enable_unit_conversion=False)
        pts = ex._extract_with_llm(content="本文" * 10, source_url="https://x.example/a", source_title="x",
                                   evidence_id="e1", section_id="1", source_reliability=0.8,
                                   research_topic="t")
        got = [(p.value, p.year, p.quarter) for p in pts]
        assert got == [(1, 2024, "Q1"), (2, 2024, "Q1"), (3, 2024, "Q2"), (4, 2023, None),
                       (5, 2024, "Q3"), (6, 2024, None)]
        assert pts[0].date_context == "2024Q1" and pts[4].date_context == "2024 第3四半期"
        assert pts[3].date_context == "2023" and pts[5].date_context == "2024H1"
        assert "2024Q1" in llm.prompts[0]                    # the prompt asks for the period
        # ... and the data layer keeps the half-year apart from the quarters
        cat = build_section_catalog(pts, "1", min_points_time=1)
        labels = sorted(tuple(cat.labels[s.labels_ref]) for s in _times(cat))
        assert ("2024Q1", "2024Q2", "2024Q3") in labels and ("2024H1",) in labels and ("2023",) in labels


# --------------------------------------------------------------------------
# (6) informativeness
# --------------------------------------------------------------------------

def _xy_data(**series):
    data = {"years": ["2021", "2022", "2023", "2024"]}
    for key, vals in series.items():
        data[key] = {"unit": "億円", "kind": "time", "labels_ref": "years", "values": list(vals),
                     "basis": ["公表" if v is not None else None for v in vals]}
    return data


def _panel(pid, form, keys, ref="years"):
    return {"id": pid, "form": form, "title": "t（億円）", "labels_ref": ref,
            "series": [{"name": k, "values_ref": k, "role": "emphasis" if i == 0 else "context"}
                       for i, k in enumerate(keys)],
            "caption": "c", "judgment": "j"}


class TestInformativeness:

    def test_line_needs_three_points(self):
        data = _xy_data(two=[5, None, 7, None], three=[5, 6, None, 8], a=[1, 2, None, None],
                        b=[None, 2.5, 3, None], flat=[5, 5, 5, 5], one=[5, None, None, None])
        valid, rejected = validate_panels([
            _panel("two", "line_xy", ["two"]), _panel("three", "line_xy", ["three"]),
            _panel("ab", "line_xy", ["a", "b"]),          # 2 + 2 points over 3 years
            _panel("flat", "line_xy", ["flat"]), _panel("one", "line_xy", ["one"])], data)
        assert [p["id"] for p in valid] == ["three", "ab"]
        reasons = {p.get("id"): r for p, r in rejected}
        assert "3 時点以上" in reasons["two"] and "（2 時点）" in reasons["two"]
        assert "同じ" in reasons["flat"] and "1 つ" in reasons["one"]

    def test_line_xy_over_points_gets_the_same_checks(self):
        def xy(pts):
            return {"unit": "億円", "values": [{"x": x, "y": y} for x, y in pts], "basis": ["公表"] * len(pts)}
        data = {"short": xy([(2024, 1), (2025, 2)]), "flat": xy([(2024, 3), (2025, 3), (2026, 3)]),
                "ok": xy([(2024, 1), (2025, 2), (2030, 5)])}
        mk = lambda pid, key: {"id": pid, "form": "line_xy", "title": "t", "caption": "c", "judgment": "j",
                               "series": [{"name": key, "values_ref": key, "role": "emphasis"}]}
        valid, rejected = validate_panels([mk("short", "short"), mk("flat", "flat"), mk("ok", "ok")], data)
        assert [p["id"] for p in valid] == ["ok"]
        reasons = {p.get("id"): r for p, r in rejected}
        assert "3 時点以上" in reasons["short"] and "同じ" in reasons["flat"]

    def test_single_series_bar_needs_three_categories(self):
        data = {"two": ["A社", "B社"], "three": ["A社", "B社", "C社"],
                "s2": {"unit": "%", "values": [60, 40], "basis": ["公表"] * 2},
                "t2": {"unit": "%", "values": [55, 45], "basis": ["公表"] * 2},
                "gap": {"unit": "%", "values": [60, None, 40], "basis": ["公表", None, "公表"]},
                "s3": {"unit": "%", "values": [50, 30, 20], "basis": ["公表"] * 3}}
        valid, rejected = validate_panels([
            _panel("bar2", "bar", ["s2"], "two"), _panel("hbar2", "hbar", ["s2"], "two"),
            _panel("gap", "bar", ["gap"], "three"), _panel("bar3", "bar", ["s3"], "three"),
            _panel("multi2", "hbar", ["s2", "t2"], "two"),                      # two series: fine
            dict(_panel("stack2", "stacked_hbar", ["s2", "t2"], "two"),
                 series=[{"name": "s2", "values_ref": "s2", "role": "part"},
                         {"name": "t2", "values_ref": "t2", "role": "part"}])], data)
        assert [p["id"] for p in valid] == ["bar3", "multi2", "stack2"]
        reasons = {p.get("id"): r for p, r in rejected}
        for pid in ("bar2", "hbar2", "gap"):
            assert "3 項目以上" in reasons[pid] and "（2 項目）" in reasons[pid], reasons[pid]

    def test_validate_raises_for_a_two_point_line(self):
        data = _xy_data(two=[5, 7, None, None])
        with pytest.raises(SpecError, match="折れ線"):
            validate(resolve([_panel("x", "line_xy", ["two"])], data))

    def test_the_prompt_states_the_rule(self):
        assert "3 時点未満の折れ線" in STYLE_GUIDE_JA and "3 項目未満の単一系列の棒" in STYLE_GUIDE_JA
        assert "fewer than 3 points" in STYLE_GUIDE_EN and "fewer than 3 categories" in STYLE_GUIDE_EN

    def test_a_chapter_with_only_two_points_gets_no_chart(self, tmp_path):
        pytest.importorskip("matplotlib")
        from deep_research_tool.evidence.locker import EvidenceLocker
        from deep_research_tool.evidence.numerical_extractor import NumericalDataStore
        from deep_research_tool.tests.test_chartkit import _pipeline_tool, _SpecLLM
        store = NumericalDataStore(research_topic="t")
        for y, v in ((2021, 40), (2022, 50), (2023, 55)):
            store.add(_dp("炭素繊維市場", "市場規模", y, v, "億円", "https://www.stat.go.jp/x", section_id="1"))
        for y, v in ((2023, 10), (2024, 12)):
            store.add(_dp("航空機向け", "需要", y, v, "千トン", "https://www.stat.go.jp/x", section_id="2"))
        locker = EvidenceLocker(research_id="r", output_dir=tmp_path / "ev")
        session = SimpleNamespace(query="炭素繊維", research_plan=None,
                                  section_contents={"1": {"title": "市場規模", "content": "本文", "sources": []},
                                                    "2": {"title": "需要", "content": "本文", "sources": []}})
        tool = _pipeline_tool(tmp_path, _SpecLLM(n_panels=1), max_panels=2)
        figs = tool._spec_charts(session, locker, store, tmp_path / "figures")
        assert [f.section_id for f in figs] == ["1"]
        y = json.loads((tmp_path / "figures" / "chart_yield.json").read_text(encoding="utf-8"))
        rejected = y["chapters"]["2"]["rejected"]
        assert rejected and "3 項目以上" in rejected[0]["reason"]


class TestReviewFixes:
    """Fixes from the independent review of v1.13.0."""

    def test_segments_products_and_bases_stay_apart(self):
        assert metric_key("Revenue (Data Center)") != metric_key("Revenue (Gaming)") != metric_key("Revenue")
        assert metric_key("販売台数（BEV）") != metric_key("販売台数（PHEV）")
        assert metric_key("シェア（台数ベース）") != metric_key("シェア（金額ベース）")
        # a period in the name keeps a quarter off the annual series
        assert metric_key("売上高（第1四半期）") != metric_key("売上高")
        # units, years and actual / forecast notes do not split a metric
        for n in ("市場規模（億円）", "市場規模（単位：億円）", "Market size (USD billion)",
                  "市場規模（2023年度）", "市場規模（予測）", "市場規模（実績）"):
            assert metric_key(n) == metric_key("市場規模"), n
        assert metric_key("市場規模（グローバル）") == metric_key("Market size (Global)")
        assert metric_key("市場規模（国内）") != metric_key("市場規模（世界）")

    def test_period_texts(self):
        from deep_research_tool.chartkit.data_layer import parse_period
        cases = {"2024年1月～3月": ("quarter", 1), "January-March 2024": ("quarter", 1),
                 "2024年4月～2025年3月": ("annual", None), "2024/3期": ("annual", None),
                 "Q1-Q3 2024": ("quarter", None), "2024 YTD": ("quarter", None),
                 "2025年3月期第1四半期": ("quarter", None), "2024年4-6月期": ("quarter", 2),
                 "2024Q1": ("quarter", 1), "2024/03": ("month", 3), "2024年3月期": ("annual", None)}
        for text, want in cases.items():
            assert parse_period(text) == want, text
