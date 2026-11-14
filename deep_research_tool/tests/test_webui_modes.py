"""
Web UI シンプル / プロ (v1.15.0):
- the recommended settings are defined once on the server (presets.py) and
  read by the page from GET /api/presets (仕上がり / 時間のかけ方 too)
- a request with use_recommended runs with them, whatever else it carried;
  a request without it is unchanged
- a missing connection setting stops a simple start with what is missing
- the history records the mode; older records without it still load
- the page: the toggle, what each mode shows, the choice kept over a
  reload, what a simple start sends, シンプル → プロ fills the form

No job ever runs here: the job manager's launch is a no-op, and nothing
talks to an LLM server or the Web.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from deep_research_tool.config import create_config
from deep_research_tool.presets import (DELIVERABLE_PRESETS, RESEARCH_PRESETS,
                                        apply_recommended, recommended_params)
from deep_research_tool.tests.test_webui_state import _browser, _load, _url
from deep_research_tool.webui import server as srv
from deep_research_tool.webui.server import JobManager, WebUIHandler, build_config_kwargs

INDEX = Path(srv.STATIC_DIR) / "index.html"


class NoRunManager(JobManager):
    """Records what a job would run with; never starts it (the recorded
    jobs stay 'running', so the cap is lifted: nothing here is queued)."""

    MAX_CONCURRENT = 100

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.launched = []

    def _launch(self, job, params):
        self.launched.append(dict(params))


@pytest.fixture
def no_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def ui_server(tmp_path, no_keys):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = NoRunManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _get(httpd, path):
    with urllib.request.urlopen(_url(httpd, path), timeout=10) as res:
        return json.loads(res.read())


def _post(httpd, path, body):
    req = urllib.request.Request(_url(httpd, path), method="POST",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---------------------------------------------------------------------------
# GET /api/presets
# ---------------------------------------------------------------------------

# the table of the request (クラウドAPI / ローカルLLM)
TABLE = {
    "openai": {"deliverable": "standard", "research_preset": "standard",
               "extraction_max_chunks": None, "extraction_max_tokens": None, "plan_review": False},
    "anthropic": {"deliverable": "standard", "research_preset": "standard",
                  "extraction_max_chunks": None, "extraction_max_tokens": None, "plan_review": False},
    "local": {"deliverable": "standard", "research_preset": "quick",
              "extraction_max_chunks": 3, "extraction_max_tokens": 3000, "plan_review": False},
}


class TestPresetsApi:
    def test_recommended_settings_follow_the_table(self, ui_server):
        data = _get(ui_server, "/api/presets")
        assert data["recommended"]["cloud"] == TABLE["openai"]
        assert data["recommended"]["local"] == TABLE["local"]
        for provider, want in TABLE.items():
            got = data["recommended_params"][provider]
            for key, value in want.items():
                assert got.get(key) == value, (provider, key, got)
            # the presets are expanded into the concrete settings
            for key, value in DELIVERABLE_PRESETS[want["deliverable"]]["values"].items():
                if value is not None:
                    assert got[key] == value, (provider, key)
            for key, value in RESEARCH_PRESETS[want["research_preset"]]["values"].items():
                assert got[key] == value, (provider, key)
            # model-dependent: never part of the recommended settings
            assert "local_think" not in got and "local_num_ctx" not in got
        assert data["recommended_params"]["openai"]["verification_profile"] == "balanced"
        assert data["recommended_params"]["local"]["verification_profile"] == "fast"

    def test_the_form_presets_come_from_the_server_only(self, ui_server):
        data = _get(ui_server, "/api/presets")
        assert set(data["deliverable_presets"]) == {"summary", "standard", "detailed"}
        assert set(data["research_presets"]) == {"quick", "standard", "thorough"}
        assert data["research_presets"]["thorough"]["values"]["verification_profile"] == "strict"
        # the page keeps no copy of the values (one place: presets.py)
        html = INDEX.read_text(encoding="utf-8")
        for p in list(DELIVERABLE_PRESETS.values()) + list(RESEARCH_PRESETS.values()):
            assert p["desc"] not in html
        assert "max_pages_per_query: 2" not in html and "preferred_body_chars: 4000" not in html


# ---------------------------------------------------------------------------
# POST /api/research
# ---------------------------------------------------------------------------

class TestUseRecommended:
    def test_the_recommended_settings_win_over_anything_else_sent(self, ui_server):
        status, data = _post(ui_server, "/api/research", {
            "query": "炭素繊維の市場調査", "use_recommended": True,
            "provider": "openai", "openai_api_key": "sk-test",
            "output_format": "pdf", "language": "en",
            # pro-form values that must not survive
            "verification_profile": "strict", "deliverable": "detailed", "research_preset": "thorough",
            "iterations": 9, "plan_review": True, "deep_think": True, "requirements_extra": "x",
            "research_engine": "gpt-researcher", "stage_llm": {"writing": {"provider": "local"}},
            "crawl_mode": "aicrawl", "extraction_max_chunks": 8, "fermi_estimation": True,
        })
        assert status == 202, data
        params = ui_server.job_manager.launched[-1]
        assert params["verification_profile"] == "balanced"
        assert params["deliverable"] == "standard" and params["research_preset"] == "standard"
        assert params["iterations"] == 2 and params["max_iterations"] == 4
        assert params["plan_review"] is False and params["deep_think"] is False
        assert params["research_engine"] == "builtin" and params["crawl_mode"] == "standard"
        for gone in ("stage_llm", "requirements_extra", "extraction_max_chunks", "fermi_estimation"):
            assert gone not in params
        # the simple inputs and the connection survive
        assert params["output_format"] == "pdf" and params["language"] == "en"
        assert params["openai_api_key"] == "sk-test"
        # and that is what the run is configured with
        cfg = create_config(**build_config_kwargs(params))
        assert cfg.research.verification_profile == "balanced"
        assert cfg.research.min_iterations == 2 and cfg.research.max_iterations == 4
        assert cfg.research.extraction_max_chunks == 8 and cfg.research.extraction_max_tokens is None
        assert cfg.research.plan_review is False

    def test_local_gets_the_shorter_run_and_the_extraction_caps(self, ui_server):
        status, data = _post(ui_server, "/api/research", {
            "query": "q", "use_recommended": True, "provider": "local",
            "local_base_url": "http://127.0.0.1:9", "local_model": "qwen3:8b",
            "local_think": False, "local_num_ctx": 8192,          # model settings pass through
            "verification_profile": "strict", "extraction_max_chunks": 8,
        })
        assert status == 202, data
        params = ui_server.job_manager.launched[-1]
        assert params["research_preset"] == "quick" and params["verification_profile"] == "fast"
        assert params["extraction_max_chunks"] == 3 and params["extraction_max_tokens"] == 3000
        assert params["crawl_mode"] == "fast_batch"
        assert params["local_think"] is False and params["local_num_ctx"] == 8192
        cfg = create_config(**build_config_kwargs(params))
        assert cfg.research.extraction_max_chunks == 3
        assert cfg.research.extraction_max_tokens == 3000
        assert cfg.api.local_model == "qwen3:8b"

    def test_only_local_documents_keep_the_crawl_card_unchanged(self):
        params = apply_recommended({"use_recommended": True, "provider": "local",
                                    "source_mode": "local", "local_documents": ["/x/a.pdf"]})
        assert "crawl_mode" not in params
        assert params["local_documents"] == ["/x/a.pdf"]

    def test_a_request_without_the_flag_is_unchanged(self, ui_server):
        request = {"query": "q", "provider": "openai", "openai_api_key": "sk-test",
                   "verification_profile": "strict", "iterations": 3, "max_iterations": 6,
                   "deliverable": "detailed", "research_preset": "thorough", "plan_review": True,
                   "extraction_max_chunks": 5, "deep_think": True}
        status, data = _post(ui_server, "/api/research", request)
        assert status == 202, data
        params = ui_server.job_manager.launched[-1]
        assert params["ui_mode"] == "pro"
        # the job manager adds the job's own folders, as it always did
        own = ("ui_mode", "output_dir", "cache_dir")
        assert {k: v for k, v in params.items() if k not in own} == request

        def config(p):
            return {k: v for k, v in build_config_kwargs(p).items() if k not in own}
        assert config(params) == config(request)
        # use_recommended false is the same as leaving it out
        status, _ = _post(ui_server, "/api/research", dict(request, use_recommended=False))
        assert status == 202
        assert config(ui_server.job_manager.launched[-1]) == config(request)

    def test_a_missing_connection_setting_stops_the_start(self, ui_server, monkeypatch):
        status, data = _post(ui_server, "/api/research",
                             {"query": "q", "use_recommended": True, "provider": "openai"})
        assert status == 400
        assert data["field"] == "connection"
        assert [m["param"] for m in data["missing"]] == ["openai_api_key"]
        assert "OPENAI_API_KEY" in data["missing"][0]["message"]
        status, data = _post(ui_server, "/api/research",
                             {"query": "q", "use_recommended": True, "provider": "anthropic",
                              "local_llm_role": "verify"})
        assert [m["param"] for m in data["missing"]] == ["anthropic_api_key", "local_base_url"]
        status, data = _post(ui_server, "/api/research",
                             {"query": "q", "use_recommended": True, "provider": "nope"})
        assert [m["param"] for m in data["missing"]] == ["provider"]
        assert ui_server.job_manager.launched == []
        assert ui_server.job_manager.history() == []
        # the server's environment counts (the page cannot see it)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        status, _ = _post(ui_server, "/api/research", {"query": "q", "use_recommended": True})
        assert status == 202
        # a local LLM needs nothing more (the backend's default URL applies)
        status, _ = _post(ui_server, "/api/research", {"query": "q", "use_recommended": "true",
                                                         "provider": "local"})
        assert status == 202


# ---------------------------------------------------------------------------
# the mode in the history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_the_mode_is_recorded(self, ui_server):
        _post(ui_server, "/api/research", {"query": "simple run", "use_recommended": True,
                                           "openai_api_key": "sk-test"})
        _post(ui_server, "/api/research", {"query": "pro run", "openai_api_key": "sk-test"})
        modes = {r["query"]: r["params_summary"].get("ui_mode") for r in _get(ui_server, "/api/history")["history"]}
        assert modes == {"simple run": "simple", "pro run": "pro"}
        ledger = json.loads((Path(ui_server.output_dir) / "jobs_ledger.json").read_text(encoding="utf-8"))
        assert sorted(r["params_summary"]["ui_mode"] for r in ledger["jobs"].values()) == ["pro", "simple"]
        # secrets never reach the history
        assert "sk-test" not in json.dumps(ledger)

    def test_an_older_ledger_without_the_mode_still_loads(self, tmp_path, no_keys):
        (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {"job-old": {
            "job_id": "job-old", "query": "以前の調査", "state": "completed", "started_at": 1.0,
            "params_summary": {"research_preset": "standard", "deliverable": "standard"}}}}), encoding="utf-8")
        manager = NoRunManager(output_dir=str(tmp_path))
        recs = manager.history()
        assert recs[0]["query"] == "以前の調査" and "ui_mode" not in recs[0]["params_summary"]

    def test_feedback_and_verify_again_follow_the_mode(self, ui_server, tmp_path):
        session = tmp_path / "job-a" / "session_a.json"
        session.parent.mkdir(parents=True)
        session.write_text("{}", encoding="utf-8")
        (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {"job-a": {
            "job_id": "job-a", "query": "元の調査", "state": "completed", "started_at": 1.0,
            "output_dir": str(session.parent),
            "params_summary": {"verification_profile": "strict", "deep_think": True, "requirements": "経営層向け",
                               "provider": "anthropic", "model": "claude-old-model", "local_num_ctx": 4096,
                               "use_recommended": True, "ui_mode": "simple"},
            "result": {"saved_artifacts": {"session": str(session)}}}}}), encoding="utf-8")
        manager = NoRunManager(output_dir=str(tmp_path))
        ui_server.job_manager = manager
        # simple: the recommended settings; the panel's own plan review choice stays
        status, data = _post(ui_server, "/api/feedback", {
            "job_id": "job-a", "feedback": "市場規模を補足", "use_recommended": True,
            "openai_api_key": "sk-test", "plan_review": True, "enable_verification": True,
            "verification_profile": "custom"})
        assert status == 202, data
        p = manager.launched[-1]
        assert p["ui_mode"] == "simple" and p["verification_profile"] == "balanced"
        assert p["deep_think"] is False and p["plan_review"] is True
        assert p["feedback"]["text"] == "市場規模を補足"
        # nothing of the original run's settings: its provider / model / pro choices
        assert p["provider"] == "openai"
        for gone in ("model", "requirements", "local_num_ctx"):
            assert gone not in p, gone
        # pro: the original's settings + this request's; the original's mode is not inherited
        status, data = _post(ui_server, "/api/feedback", {
            "job_id": "job-a", "feedback": "市場規模を補足", "openai_api_key": "sk-test"})
        assert status == 202, data
        p = manager.launched[-1]
        assert p["ui_mode"] == "pro" and p["verification_profile"] == "strict" and p["deep_think"] is True
        assert p["model"] == "claude-old-model" and p["requirements"] == "経営層向け"      # as before
        assert "use_recommended" not in p
        # 検証のみ再実行
        status, data = _post(ui_server, "/api/verify", {
            "job_id": "job-a", "use_recommended": True, "provider": "local", "verification_profile": "strict"})
        assert status == 202, data
        p = manager.launched[-1]
        assert p["ui_mode"] == "simple" and p["verification_profile"] == "fast" and p["verify"]
        status, data = _post(ui_server, "/api/verify", {"job_id": "job-a", "use_recommended": True})
        assert status == 400 and data["missing"][0]["param"] == "openai_api_key"
        # 中断箇所から再開 continues the run as it was; changed settings are not "recommended"
        status, data = _post(ui_server, "/api/resume", {"job_id": "job-a", "openai_api_key": "sk-test"})
        assert status == 202, data
        assert manager.launched[-1]["ui_mode"] == "simple"
        status, data = _post(ui_server, "/api/resume", {"job_id": "job-a", "overrides": {"deep_think": True}})
        assert status == 202, data
        p = manager.launched[-1]
        assert p["ui_mode"] == "pro" and "use_recommended" not in p

    def test_fermi_runs_are_recorded_as_pro(self, ui_server):
        status, data = _post(ui_server, "/api/fermi", {"question": "国内の EV 充電器の数は？"})
        assert status == 202, data
        assert ui_server.job_manager.launched[-1]["ui_mode"] == "pro"


def test_recommended_params_are_complete_requests():
    # every expanded value passes the same validation as the form's values
    for provider in ("openai", "anthropic", "local"):
        kwargs = build_config_kwargs(dict(recommended_params(provider), provider=provider,
                                          openai_api_key="k", anthropic_api_key="k"))
        create_config(**kwargs)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

SIMPLE_VISIBLE = ["#query", "#sourceModes", "input[name=simple_output_format] >> xpath=../..",
                  "input[name=simple_language] >> xpath=../..", "#summaryStrip", "#startBtn"]
PRO_ONLY = ["#presetCard", "#runModes", "#engineBox", "#requirements", "#deliverableModes",
            "#presetModes", "#presetEffects", "#advancedCard", "#fermiCard", "#output_format", "#language"]


def _visible(page, selectors):
    return {s: page.is_visible(s) for s in selectors}


def test_the_page_opens_in_simple_and_the_choice_is_kept(ui_server):
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(ui_server, "/"), None)
        assert errors == [], errors
        assert page.evaluate("uiMode") == "simple"
        assert page.get_attribute("#uiModeToggle [data-set-mode=simple]", "aria-checked") == "true"
        assert all(_visible(page, SIMPLE_VISIBLE).values()), _visible(page, SIMPLE_VISIBLE)
        assert not any(_visible(page, PRO_ONLY).values()), _visible(page, PRO_ONLY)
        assert not page.is_visible("#plan_review")
        # the recommended settings, from the server, in one or two lines
        strip = page.inner_text("#summaryStrip")
        assert "推奨設定（クラウドAPI）" in strip and "標準レポート" in strip
        assert "検証: Balanced" in page.get_attribute("#summaryStrip .simple-rec", "title")   # details on hover
        assert page.inner_text("#startBtn").strip() == "🚀 調査を開始"
        # ⚙設定: the connection tab only, with the simple mode's provider
        page.click("#settingsBtn")
        assert page.is_visible("#set_provider") and not page.is_visible(".modal-tab[data-tab=tabFeatures]")
        page.select_option("#set_provider", "local")
        page.click("#settingsSaveBtn")
        strip = page.inner_text("#summaryStrip")
        assert "推奨設定（ローカルLLM）" in strip and "短時間" in strip and "3チャンク/3000" in strip

        page.click("#uiModeToggle [data-set-mode=pro]")
        assert page.evaluate("document.body.dataset.uiMode") == "pro"
        assert all(_visible(page, ["#presetCard", "#runModes", "#deliverableModes", "#advancedCard",
                                   "#output_format", "#plan_review"]).values())
        assert not page.is_visible("input[name=simple_output_format] >> xpath=../..")
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(400)
        assert page.evaluate("uiMode") == "pro" and page.is_visible("#advancedCard")
        assert page.get_attribute("#uiModeToggle [data-set-mode=pro]", "aria-checked") == "true"
        page.click("#uiModeToggle [data-set-mode=simple]")
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(400)
        assert page.evaluate("uiMode") == "simple" and not page.is_visible("#advancedCard")
        assert page.evaluate("localStorage.getItem('drt_ui_mode')") == "simple"
        assert errors == [], errors
    finally:
        browser.close()
        pw.stop()


def test_a_simple_start_sends_only_the_simple_inputs(ui_server):
    pw, browser = _browser()
    try:
        page = browser.new_page()
        errors, sent = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))

        def research(route):
            sent.append(json.loads(route.request.post_data))
            route.fulfill(status=400, content_type="application/json",
                          body=json.dumps({"error": "stop here"}))
        page.route("**/api/research", research)
        page.goto(_url(ui_server, "/"), wait_until="networkidle")
        page.wait_for_timeout(500)
        # pro values set first, then back to simple: none of them may go out
        page.click("#uiModeToggle [data-set-mode=pro]")
        page.evaluate("""() => {
            setRadio('verification_profile', 'strict'); setRadio('research_preset', 'thorough');
            document.getElementById('model').value = 'gpt-x';
            document.getElementById('requirements').value = '経営層向け';
            document.getElementById('local_documents').value = '/srv/secret.pdf';
            document.querySelector('.stage-provider[data-stage=writing]').value = 'local';
            document.getElementById('feat_deep_think').checked = true;
            document.getElementById('feat_fermi').checked = true;
        }""")
        page.click("#uiModeToggle [data-set-mode=simple]")
        page.fill("#query", "炭素繊維の市場調査")
        page.check("input[name=simple_output_format][value=pdf]", force=True)
        page.check("input[name=simple_language][value=en]", force=True)
        page.check("input[name=source_mode][value=urls_web]", force=True)
        page.fill("#source_urls", "https://www.example.go.jp/a")
        page.click("#startBtn")
        page.wait_for_timeout(500)
        assert errors == [], errors
        body = sent[-1]
        assert body["use_recommended"] is True
        assert body["query"] == "炭素繊維の市場調査"
        assert body["output_format"] == "pdf" and body["language"] == "en"
        assert body["source_mode"] == "urls_web" and body["source_urls"] == ["https://www.example.go.jp/a"]
        assert body["provider"] == "openai"
        allowed = set(page.evaluate("PRESETS.simple_keys")) | {"use_recommended"}
        assert set(body) <= allowed, set(body) - allowed
        for gone in ("verification_profile", "research_preset", "model", "requirements", "local_documents",
                     "stage_llm", "deep_think", "fermi_estimation", "plan_review"):
            assert gone not in body
    finally:
        browser.close()
        pw.stop()


def test_a_missing_key_opens_the_settings_on_it(ui_server):
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(ui_server, "/"), None)
        page.fill("#query", "炭素繊維の市場調査")
        page.click("#startBtn")
        page.wait_for_timeout(600)
        assert page.is_visible("#settingsModal")
        assert "OpenAI の APIキー" in page.inner_text("#settingsMissing")
        assert page.evaluate("[...document.querySelectorAll('.field-missing')].map(e => e.id)") == ["set_openai_key"]
        assert "接続設定が足りません" in page.inner_text("#formError")
        assert ui_server.job_manager.launched == []
        page.fill("#set_openai_key", "sk-test")
        assert page.evaluate("document.querySelectorAll('.field-missing').length") == 0
        page.click("#settingsSaveBtn")
        page.click("#startBtn")
        page.wait_for_timeout(800)
        assert len(ui_server.job_manager.launched) == 1
        assert ui_server.job_manager.launched[0]["ui_mode"] == "simple"
        assert errors == [], errors
    finally:
        browser.close()
        pw.stop()


def test_switching_to_pro_shows_what_simple_applied(ui_server):
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(ui_server, "/"),
                                "localStorage.setItem('drt_settings', JSON.stringify({simple_provider: 'local'}))")
        assert page.evaluate("simpleProvider()") == "local"
        page.check("input[name=simple_output_format][value=pdf]", force=True)
        page.click("#uiModeToggle [data-set-mode=pro]")
        form = page.evaluate("""() => ({provider: provider.value, model: model.value,
            deliverable: radioValue('deliverable'), preset: radioValue('research_preset'),
            vp: radioValue('verification_profile'), crawl: radioValue('crawl_mode'),
            iterations: iterations.value, plan: plan_review.checked, fmt: output_format.value,
            lang: language.value, engine: radioValue('research_engine'), run: radioValue('run_mode')})""")
        assert form == {"provider": "local", "model": "", "deliverable": "standard", "preset": "quick",
                        "vp": "fast", "crawl": "fast_batch", "iterations": "1", "plan": False,
                        "fmt": "pdf", "lang": "ja", "engine": "builtin", "run": "research_verify"}
        note = page.inner_text("#proSwitchNote")
        assert "推奨設定をフォームに反映しました" in note and "チャンク 3・出力上限 3000" in note
        # the pro values it replaced come back once
        page.click("#uiModeToggle [data-set-mode=simple]")
        page.click("#uiModeToggle [data-set-mode=pro]")
        page.check("input[name=research_preset][value=thorough]", force=True)     # turns DeepThink on
        page.evaluate("setRadio('verification_profile', 'strict'); document.getElementById('model').value = 'qwen3:32b'")
        assert page.evaluate("document.getElementById('feat_deep_think').checked") is True
        page.click("#uiModeToggle [data-set-mode=simple]")
        page.click("#uiModeToggle [data-set-mode=pro]")
        assert page.evaluate("radioValue('verification_profile')") == "fast"
        assert page.evaluate("document.getElementById('feat_deep_think').checked") is False
        page.click("#proSwitchNote button[data-act=undo]")
        assert page.evaluate("radioValue('verification_profile')") == "strict"
        assert page.evaluate("document.getElementById('model').value") == "qwen3:32b"
        assert page.evaluate("radioValue('research_preset')") == "thorough"
        assert page.evaluate("collectParams().deep_think") is True                  # DeepThink is back
        # a page opened in pro is the pro form as before (nothing applied)
        page2, errors2, _ = _load(browser, _url(ui_server, "/"), "localStorage.setItem('drt_ui_mode', 'pro')")
        assert page2.evaluate("radioValue('research_preset')") == "standard"
        assert page2.evaluate("plan_review.checked") is True
        assert page2.evaluate("output_format.value") == "markdown"
        assert not page2.is_visible("#proSwitchNote")
        assert errors == [] and errors2 == []
    finally:
        browser.close()
        pw.stop()


def test_the_history_tells_the_modes_apart(tmp_path, no_keys):
    (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {
        "job-s": {"job_id": "job-s", "query": "シンプルの調査", "state": "completed", "started_at": 3.0,
                  "params_summary": {"ui_mode": "simple", "research_preset": "standard", "deliverable": "standard"}},
        "job-p": {"job_id": "job-p", "query": "プロの調査", "state": "completed", "started_at": 2.0,
                  "params_summary": {"ui_mode": "pro", "research_preset": "thorough", "deliverable": "detailed"}},
        "job-o": {"job_id": "job-o", "query": "以前の調査", "state": "completed", "started_at": 1.0,
                  "params_summary": {"research_preset": "quick", "deliverable": "summary"}}}}), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = NoRunManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        assert errors == [], errors
        rows = page.query_selector_all("#historyList .hist-row")
        meta = {r.query_selector(".q").inner_text(): r.query_selector(".meta").inner_text() for r in rows}
        assert "シンプル" in meta["シンプルの調査"]
        assert "プロ" in meta["プロの調査"]
        assert "シンプル" not in meta["以前の調査"] and "プロ" not in meta["以前の調査"]
        # the preset labels come from the server
        text = page.inner_text("#historyList")
        assert "詳細調査" in text and "要約" in text and "短時間" in text
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def test_the_page_carries_the_presets_and_copes_without_them(ui_server):
    # served by the server: the presets are in the page itself
    with urllib.request.urlopen(_url(ui_server, "/"), timeout=10) as res:
        html = res.read().decode("utf-8")
    start = html.index('<script id="presetsData" type="application/json">') + len(
        '<script id="presetsData" type="application/json">')
    data = json.loads(html[start:html.index("</script>", start)])
    assert data == _get(ui_server, "/api/presets")
    pw, browser = _browser()
    try:
        # a page served WITHOUT them asks /api/presets: slow -> a start waits for it
        page = browser.new_page()
        errors, sent = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        raw = INDEX.read_text(encoding="utf-8")
        page.route("**/", lambda r: r.fulfill(status=200, content_type="text/html; charset=utf-8", body=raw))

        def slow_presets(route):
            import time
            time.sleep(1.5)
            route.continue_()
        page.route("**/api/presets", slow_presets)

        def research(route):
            sent.append(json.loads(route.request.post_data))
            route.fulfill(status=400, content_type="application/json", body=json.dumps({"error": "stop"}))
        page.route("**/api/research", research)
        page.goto(_url(ui_server, "/"), wait_until="domcontentloaded")
        page.fill("#query", "炭素繊維")
        page.click("#startBtn")
        page.wait_for_timeout(2500)
        assert sent and sent[-1]["use_recommended"] is True and sent[-1]["query"] == "炭素繊維"
        assert "推奨設定（クラウドAPI）" in page.inner_text("#summaryStrip")
        # failing: said in the simple summary, in pro under the cards, and by a feedback start
        page2 = browser.new_page()
        page2.on("pageerror", lambda e: errors.append(str(e)))
        page2.route("**/", lambda r: r.fulfill(status=200, content_type="text/html; charset=utf-8", body=raw))
        page2.route("**/api/presets", lambda r: r.fulfill(status=500, content_type="application/json",
                                                         body=json.dumps({"error": "down"})))
        page2.goto(_url(ui_server, "/"), wait_until="networkidle")
        page2.wait_for_timeout(3800)                      # three attempts
        assert "推奨設定を取得できませんでした" in page2.inner_text("#summaryStrip")
        page2.evaluate("document.getElementById('feedback_text').value = 'x'; resultJobId = 'job-x'")
        page2.evaluate("startFeedback()")
        page2.wait_for_timeout(300)
        assert "推奨設定をサーバーから取得できていません" in page2.inner_text("#feedbackError")
        page2.click("#uiModeToggle [data-set-mode=pro]")
        assert "取得できませんでした" in page2.inner_text("#presetEffects")
        assert "フォームはそのまま" in page2.inner_text("#proSwitchNote")
        assert errors == [], errors
    finally:
        browser.close()
        pw.stop()


def test_the_fermi_link_opens_the_panel_in_either_mode(ui_server):
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(ui_server, "/#fermi"), None)
        assert errors == [], errors
        assert page.evaluate("uiMode") == "pro"
        assert page.is_visible("#fermi_question")
        # the saved choice is left alone
        assert page.evaluate("localStorage.getItem('drt_ui_mode')") is None
    finally:
        browser.close()
        pw.stop()


# ---------------------------------------------------------------------------
# second review
# ---------------------------------------------------------------------------

class TestConnectionValues:
    def test_a_blank_or_odd_key_is_not_a_key(self, ui_server, monkeypatch):
        for bad in ("   ", False, 0, ["sk"], {"k": 1}):
            status, data = _post(ui_server, "/api/research",
                                 {"query": "q", "use_recommended": True, "openai_api_key": bad})
            assert status == 400 and data["missing"][0]["param"] == "openai_api_key", bad
        monkeypatch.setenv("OPENAI_API_KEY", "   ")             # a blank environment variable
        status, data = _post(ui_server, "/api/research", {"query": "q", "use_recommended": True})
        assert status == 400
        # with a real environment key, a blank request value gives way to it
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        status, _ = _post(ui_server, "/api/research", {"query": "q", "use_recommended": True,
                                                         "openai_api_key": "   ", "local_base_url": " "})
        assert status == 202
        params = ui_server.job_manager.launched[-1]
        assert "openai_api_key" not in params and "local_base_url" not in params
        assert create_config(**build_config_kwargs(params)).api.openai_api_key == "sk-env"
        # text values are trimmed
        _post(ui_server, "/api/research", {"query": "q", "use_recommended": True,
                                           "openai_api_key": " sk-x ", "model": " gpt-x "})
        params = ui_server.job_manager.launched[-1]
        assert params["openai_api_key"] == "sk-x" and params["model"] == "gpt-x"

    def test_a_re_output_keeps_the_mode_of_its_run(self, ui_server, tmp_path):
        session = tmp_path / "job-s" / "session_s.json"
        session.parent.mkdir(parents=True)
        session.write_text("{}", encoding="utf-8")
        (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {
            "job-s": {"job_id": "job-s", "query": "元", "state": "completed", "started_at": 1.0,
                      "output_dir": str(session.parent), "params_summary": {"ui_mode": "simple"},
                      "result": {"saved_artifacts": {"session": str(session)}}},
            "job-o": {"job_id": "job-o", "query": "古い", "state": "completed", "started_at": 1.0,
                      "output_dir": str(session.parent), "params_summary": {},
                      "result": {"saved_artifacts": {"session": str(session)}}}}}), encoding="utf-8")
        manager = NoRunManager(output_dir=str(tmp_path))
        ui_server.job_manager = manager
        assert _post(ui_server, "/api/regenerate", {"job_id": "job-s", "output_format": "docx"})[0] == 202
        assert manager.launched[-1]["ui_mode"] == "simple"
        assert _post(ui_server, "/api/regenerate", {"job_id": "job-o", "output_format": "docx"})[0] == 202
        assert "ui_mode" not in manager.launched[-1]


def test_the_form_starts_from_the_server_presets(ui_server, monkeypatch):
    from deep_research_tool import presets
    monkeypatch.setitem(presets.RESEARCH_PRESETS["standard"], "values",
                        dict(presets.RESEARCH_PRESETS["standard"]["values"], iterations=3, max_pages_per_query=4))
    monkeypatch.setitem(presets.DELIVERABLE_PRESETS["standard"], "values",
                        dict(presets.DELIVERABLE_PRESETS["standard"]["values"], gap_fill_rounds=2))
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(ui_server, "/"), "localStorage.setItem('drt_ui_mode', 'pro')")
        assert errors == [], errors
        form = page.evaluate("({it: iterations.value, mp: max_pages_per_query.value, gap: gap_fill_rounds.value})")
        assert form == {"it": "3", "mp": "4", "gap": "2"}
        sent = page.evaluate("collectParams()")
        assert sent["iterations"] == 3 and sent["max_pages_per_query"] == 4 and sent["gap_fill_rounds"] == 2
        # a saved DeepThink (⚙設定 → 機能) is not reset by it
        page2, errors2, _ = _load(browser, _url(ui_server, "/"),
                                  "localStorage.setItem('drt_ui_mode', 'pro');"
                                  "localStorage.setItem('drt_settings', JSON.stringify({deep_think: true}))")
        assert page2.evaluate("document.getElementById('feat_deep_think').checked") is True
        assert errors2 == []
    finally:
        browser.close()
        pw.stop()


def test_the_review_fixes_on_the_page(ui_server, tmp_path):
    pw, browser = _browser()
    try:
        # #fermi shows pro; once the user picks シンプル, a reload keeps it
        page, errors, _ = _load(browser, _url(ui_server, "/#fermi"), None)
        page.click("#uiModeToggle [data-set-mode=simple]")
        assert page.evaluate("location.hash") != "#fermi"
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(300)
        assert page.evaluate("uiMode") == "simple"

        # the summary stays within two lines at the form's width, cloud and local
        for provider in ("openai", "local"):
            page.evaluate(f"document.getElementById('set_provider').value = '{provider}'; updateSummaryStrip()")
            lines = page.evaluate("""() => [...document.querySelectorAll('#summaryStrip .simple-rec > span')]
                .map(s => Math.round(s.getBoundingClientRect().height / parseFloat(getComputedStyle(s).lineHeight)))""")
            assert sum(lines) <= 2, (provider, lines)
        assert "3チャンク/3000" in page.inner_text("#summaryStrip")
        page.evaluate("document.getElementById('set_provider').value = 'openai'; updateSummaryStrip()")

        # the missing key: the message is next to the field, in view
        page.fill("#query", "q")
        page.click("#startBtn")
        page.wait_for_timeout(500)
        msg = page.query_selector("#set_openai_key + .missing-msg")
        assert msg and msg.is_visible() and "APIキー" in msg.inner_text()
        page.fill("#set_openai_key", "s")
        assert not page.query_selector("#set_openai_key + .missing-msg")
        page.click("#settingsCloseBtn")

        # a saved 検証 OFF: the switch says it turned 検証 on, and undo puts it back
        page2, errors2, _ = _load(browser, _url(ui_server, "/"),
                                  "localStorage.setItem('drt_settings', JSON.stringify({enable_verification: false}))")
        page2.click("#uiModeToggle [data-set-mode=pro]")
        note = page2.inner_text("#proSwitchNote")
        assert "検証 OFF→ON" in note and "実行モード: 検証入り調査" in note
        page2.click("#proSwitchNote button[data-act=undo]")
        assert page2.evaluate("radioValue('run_mode')") == "research_only"
        assert page2.evaluate("document.getElementById('feat_verification').checked") is False
        assert errors == [] and errors2 == []
    finally:
        browser.close()
        pw.stop()


def test_simple_copy_of_a_run_with_files_says_to_add_them_again(tmp_path, no_keys):
    (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {"job-f": {
        "job_id": "job-f", "query": "資料つきの調査", "state": "completed", "started_at": 1.0,
        "params_summary": {"ui_mode": "simple", "source_mode": "hybrid",
                           "local_documents": [str(tmp_path / "uploads" / "a.pdf")], "output_format": "pdf"}}}}),
        encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = NoRunManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        page.click("#historyList .hist-row button[data-act=dup]")
        page.wait_for_timeout(300)
        assert page.evaluate("radioValue('source_mode')") == "hybrid"
        assert page.evaluate("radioValue('simple_output_format')") == "pdf"
        page.evaluate("setView('history')")
        assert "もう一度追加してください" in page.inner_text("#historyList .hist-msg")
        assert errors == [], errors
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def test_a_page_without_the_presets_never_starts_twice(ui_server):
    pw, browser = _browser()
    try:
        page = browser.new_page()
        sent = []
        raw = INDEX.read_text(encoding="utf-8")
        page.route("**/", lambda r: r.fulfill(status=200, content_type="text/html; charset=utf-8", body=raw))

        def slow(route):
            import time
            time.sleep(1.5)
            route.continue_()
        page.route("**/api/presets", slow)

        def research(route):
            sent.append(route.request.post_data)
            route.fulfill(status=400, content_type="application/json", body=json.dumps({"error": "stop"}))
        page.route("**/api/research", research)
        page.goto(_url(ui_server, "/"), wait_until="domcontentloaded")
        page.fill("#query", "q")
        page.click("#startBtn")
        assert page.evaluate("document.getElementById('startBtn').disabled") is True
        page.click("#startBtn", force=True)
        page.wait_for_timeout(2500)
        assert len(sent) == 1
    finally:
        browser.close()
        pw.stop()
