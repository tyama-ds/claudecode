"""
Chart settings in v1.13.0 (fakes only; no network, no paid API):
- the improved style ('enhanced', 改良版) is the default everywhere:
  ReportConfig, create_config, the Web UI parameter, the UI selector and
  its saved / restored setting, the CLI
- charts per chapter default 6 -> 3
- a report-wide cap chart_max_total (default 12, 0 = unlimited, 0..60),
  validated and plumbed through create_config, the server, the UI and the
  CLI
"""

import json
import os
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from deep_research_tool.config import (
    CHART_MAX_TOTAL_LIMIT, DEFAULT_CHART_MAX_PER_SECTION, DEFAULT_CHART_MAX_TOTAL,
    DEFAULT_CHART_STYLE, ReportConfig, create_config,
)

INDEX = Path(__file__).resolve().parent.parent / "webui" / "static" / "index.html"


def _cfg(tmp_path, **kw):
    return create_config(provider="openai", openai_api_key="t", output_dir=str(tmp_path), **kw)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

class TestConfigDefaults:

    def test_report_config_defaults(self):
        rp = ReportConfig()
        assert rp.chart_style == "enhanced" == DEFAULT_CHART_STYLE
        assert rp.chart_max_per_section == 3 == DEFAULT_CHART_MAX_PER_SECTION
        assert rp.chart_max_total == 12 == DEFAULT_CHART_MAX_TOTAL
        assert CHART_MAX_TOTAL_LIMIT == 60

    def test_create_config_defaults(self, tmp_path):
        rp = _cfg(tmp_path).report
        assert (rp.chart_style, rp.chart_max_per_section, rp.chart_max_total) == ("enhanced", 3, 12)

    def test_an_empty_style_is_the_default_and_classic_stays_selectable(self, tmp_path):
        assert _cfg(tmp_path, chart_style="").report.chart_style == "enhanced"
        assert _cfg(tmp_path, chart_style=None).report.chart_style == "enhanced"
        assert _cfg(tmp_path, chart_style=" CLASSIC ").report.chart_style == "classic"
        with pytest.raises(ValueError):
            _cfg(tmp_path, chart_style="v15")

    @pytest.mark.parametrize("value", [0, 1, 12, 60])
    def test_chart_max_total_in_range(self, tmp_path, value):
        assert _cfg(tmp_path, chart_max_total=value).report.chart_max_total == value

    def test_chart_max_total_none_is_the_default(self, tmp_path):
        assert _cfg(tmp_path, chart_max_total=None).report.chart_max_total == 12

    @pytest.mark.parametrize("value", [-1, 61, 1000, True, False, 2.5, 12.0, "12", "abc"])
    def test_chart_max_total_rejected(self, tmp_path, value):
        with pytest.raises(ValueError, match="chart_max_total"):
            _cfg(tmp_path, chart_max_total=value)

    def test_validate_reports_a_bad_total_set_directly(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert not [e for e in cfg.validate() if "chart_max_total" in e]
        for bad in (-1, 61, None, 3.5):
            cfg.report.chart_max_total = bad
            assert [e for e in cfg.validate() if "chart_max_total" in e], bad
        cfg.report.chart_max_total = 0
        assert not [e for e in cfg.validate() if "chart_max_total" in e]

    def test_run_research_style_kwargs_reach_the_config(self, tmp_path):
        # deep_research()/run_research() forward **kwargs to create_config
        rp = _cfg(tmp_path, chart_max_per_section=5, chart_max_total=0, chart_style="classic").report
        assert (rp.chart_max_per_section, rp.chart_max_total, rp.chart_style) == (5, 0, "classic")


# --------------------------------------------------------------------------
# Web UI server parameters
# --------------------------------------------------------------------------

class TestServerParams:

    def test_map_and_style_default(self):
        from deep_research_tool.webui.server import _CONFIG_PARAM_MAP, _chart_style_param
        assert _CONFIG_PARAM_MAP["chart_max_total"] == "chart_max_total"
        assert _CONFIG_PARAM_MAP["chart_style"] == "chart_style"
        assert _chart_style_param(None) == "enhanced"
        assert _chart_style_param("Classic") == "classic"

    @pytest.mark.parametrize("raw, want", [("0", 0), ("12", 12), ("60", 60), (7, 7)])
    def test_total_accepted(self, raw, want):
        from deep_research_tool.webui.server import build_config_kwargs
        assert build_config_kwargs({"chart_max_total": raw})["chart_max_total"] == want

    @pytest.mark.parametrize("raw", ["-1", "61", "2.5", 2.5, True, "abc"])
    def test_total_rejected_with_the_field_name(self, raw):
        from deep_research_tool.webui.server import FieldError, build_config_kwargs
        with pytest.raises(FieldError) as ei:
            build_config_kwargs({"chart_max_total": raw})
        assert ei.value.field == "chart_max_total"

    def test_empty_values_fall_back_to_config_defaults(self, tmp_path):
        from deep_research_tool.webui.server import build_config_kwargs
        kw = build_config_kwargs({"provider": "openai", "openai_api_key": "t", "output_dir": str(tmp_path),
                                  "chart_max_total": "", "chart_style": "", "chart_max_per_section": ""})
        assert "chart_max_total" not in kw and "chart_style" not in kw
        rp = create_config(**kw).report
        assert (rp.chart_style, rp.chart_max_per_section, rp.chart_max_total) == ("enhanced", 3, 12)

    def test_ui_values_reach_the_config(self, tmp_path):
        from deep_research_tool.webui.server import build_config_kwargs
        kw = build_config_kwargs({"provider": "openai", "openai_api_key": "t", "output_dir": str(tmp_path),
                                  "chart_max_total": "0", "chart_style": "classic",
                                  "chart_max_per_section": "6"})
        rp = create_config(**kw).report
        assert (rp.chart_style, rp.chart_max_per_section, rp.chart_max_total) == ("classic", 6, 0)

    def test_per_chapter_range_unchanged(self):
        from deep_research_tool.webui.server import FieldError, build_config_kwargs
        assert build_config_kwargs({"chart_max_per_section": "12"})["chart_max_per_section"] == 12
        for bad in ("0", "13"):
            with pytest.raises(FieldError):
                build_config_kwargs({"chart_max_per_section": bad})


# --------------------------------------------------------------------------
# Web UI page (static checks)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def page():
    return INDEX.read_text(encoding="utf-8")


class TestPageMarkup:

    def test_style_selector_labels_and_default(self, page):
        sel = re.search(r'<select id="feat_chart_style".*?</select>', page, re.S).group(0)
        assert '<option value="enhanced" selected>改良版（既定）</option>' in sel
        assert '<option value="classic">コア維持版</option>' in sel
        assert "コア維持版（既定）" not in page

    def test_no_classic_fallbacks_left(self, page):
        assert "|| 'classic'" not in page and '|| "classic"' not in page

    def test_count_inputs(self, page):
        assert re.search(r'<input type="number" id="feat_chart_max" value="3" min="1" max="12"', page)
        assert re.search(r'<input type="number" id="feat_chart_max_total" value="12" min="0" max="60"', page)
        assert "図表の上限（全体）" in page
        assert "既定 3" in page and "既定 6" not in page

    def test_total_is_saved_restored_and_sent(self, page):
        assert "s.chart_max_total = chartMaxTotal()" in page
        assert "p.chart_max_total = chartMaxTotal()" in page
        assert "restoreChartSettings(saved)" in page
        assert "s.chart_settings_rev = CHART_SETTINGS_REV" in page


# --------------------------------------------------------------------------
# Web UI page in a real browser (skipped where Playwright / Chromium are missing)
# --------------------------------------------------------------------------

@pytest.fixture
def ui_url(tmp_path):
    from deep_research_tool.webui.server import JobManager, WebUIHandler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = "/opt/pw-browsers/chromium"
    pw = sync_api.sync_playwright().start()
    try:
        b = pw.chromium.launch(**({"executable_path": exe} if os.path.exists(exe) else {}),
                               args=["--no-sandbox"])
    except Exception as e:                       # no browser installed here
        pw.stop()
        pytest.skip(f"chromium not available: {e}")
    yield b
    b.close()
    pw.stop()


def _open(browser, url, saved=None):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    if saved is not None:
        page.add_init_script(
            f"if (!sessionStorage.getItem('t_seeded')) {{"
            f" localStorage.setItem('drt_settings', {json.dumps(json.dumps(saved))});"
            f" sessionStorage.setItem('t_seeded', '1'); }}")
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(300)
    return page, errors


def _sent(page):
    p = page.evaluate("settingsParams()")
    return p.get("chart_style"), p.get("chart_max_per_section"), p.get("chart_max_total")


@pytest.mark.parametrize("saved, want", [
    # never saved: the new defaults
    (None, ("enhanced", 3, 12)),
    # saved before v1.13.0 (no chart_settings_rev): the old defaults were
    # stored without being chosen, so the new defaults apply
    ({"chart_style": "classic", "chart_max_per_section": 6}, ("enhanced", 3, 12)),
    # ... but values that differ from the old defaults were choices
    ({"chart_style": "enhanced", "chart_max_per_section": 4}, ("enhanced", 4, 12)),
    ({"chart_style": "classic", "chart_max_per_section": 2}, ("enhanced", 2, 12)),
    # saved from v1.13.0 on: restored as saved, classic and 6 included
    ({"chart_style": "classic", "chart_max_per_section": 6, "chart_max_total": 0,
      "chart_settings_rev": 2}, ("classic", 6, 0)),
    ({"chart_style": "enhanced", "chart_max_per_section": 3, "chart_max_total": 25,
      "chart_settings_rev": 2}, ("enhanced", 3, 25)),
    # broken / out-of-range values never stop the page; the field is clamped
    ({"chart_style": "fancy", "chart_max_per_section": "x", "chart_max_total": 999,
      "chart_settings_rev": 2}, ("enhanced", 3, 60)),
])
def test_restored_settings(browser, ui_url, saved, want):
    page, errors = _open(browser, ui_url, saved)
    assert errors == [], errors
    assert _sent(page) == want


def test_saving_keeps_an_explicit_classic_and_a_zero_total(browser, ui_url):
    page, errors = _open(browser, ui_url, {})
    assert errors == []
    page.evaluate("""() => {
        document.getElementById('feat_chart_style').value = 'classic';
        document.getElementById('feat_chart_max').value = '6';
        document.getElementById('feat_chart_max_total').value = '0';
        saveSettings();
    }""")
    stored = json.loads(page.evaluate("localStorage.getItem('drt_settings')"))
    assert stored["chart_style"] == "classic" and stored["chart_max_per_section"] == 6
    assert stored["chart_max_total"] == 0 and stored["chart_settings_rev"] == 2
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(300)
    assert _sent(page) == ("classic", 6, 0)


def test_sent_values_are_clamped(browser, ui_url):
    page, _ = _open(browser, ui_url)
    page.evaluate("""() => {
        document.getElementById('feat_chart_max').value = '40';
        document.getElementById('feat_chart_max_total').value = '-5';
    }""")
    assert _sent(page)[1:] == (12, 0)
    page.evaluate("document.getElementById('feat_chart_max_total').value = ''")
    assert _sent(page)[2] == 12


def test_history_summary_shows_the_total(browser, ui_url):
    page, _ = _open(browser, ui_url)
    html = page.evaluate("historyParams({auto_figures: true, chart_max_per_section: 3, chart_max_total: 0,"
                         " chart_style: 'classic'}, {})")
    assert "グラフ最大 3/章" in html and "全体の上限なし" in html and "コア維持版" in html
    html = page.evaluate("historyParams({auto_figures: true, chart_max_total: 12, chart_style: 'enhanced'}, {})")
    assert "全体で最大 12" in html and "改良版" in html
    # a run from before v1.13.0 has no total: no total chip
    html = page.evaluate("historyParams({auto_figures: true, chart_max_per_section: 6}, {})")
    assert "全体" not in html


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _invoke(monkeypatch, tmp_path, *args):
    from click.testing import CliRunner
    from deep_research_tool import cli as cli_mod
    seen = {}
    real = cli_mod.create_config

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(cli_mod, "create_config", spy)
    res = CliRunner().invoke(cli_mod.cli, ["research", "テーマ", "--openai-key", "sk-t",
                                           "--output-dir", str(tmp_path), *args], input="n\n")
    return res, seen


class TestCli:

    def test_defaults(self, monkeypatch, tmp_path):
        res, seen = _invoke(monkeypatch, tmp_path)
        assert res.exit_code == 0, res.output
        assert seen["chart_style"] == "enhanced" and seen["chart_max_total"] == 12

    def test_options_reach_the_config(self, monkeypatch, tmp_path):
        res, seen = _invoke(monkeypatch, tmp_path, "--auto-figures",
                            "--chart-style", "classic", "--chart-max-total", "0")
        assert res.exit_code == 0, res.output
        assert seen["chart_style"] == "classic" and seen["chart_max_total"] == 0
        assert "Charts" in res.output and "classic" in res.output and "Cancelled" in res.output

    @pytest.mark.parametrize("args", [("--chart-max-total", "61"), ("--chart-max-total", "-1"),
                                      ("--chart-style", "fancy")])
    def test_bad_values_are_usage_errors(self, monkeypatch, tmp_path, args):
        res, seen = _invoke(monkeypatch, tmp_path, *args)
        assert res.exit_code == 2 and not seen

    def test_help_lists_the_options(self):
        from click.testing import CliRunner
        from deep_research_tool.cli import cli
        out = CliRunner().invoke(cli, ["research", "--help"]).output
        assert "--chart-style" in out and "--chart-max-total" in out
