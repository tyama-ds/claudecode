"""
Local-LLM delegation (⚙設定 → ロールルーティング / 工程別=ローカル) inside a run
whose MAIN provider is something else:

- role routing fills ONLY the finalization clients (verification /
  revision); the research-stage clients stay on the main provider
- DelegatedLLMClient: errors name the role and the setting, the first
  failure switches the role to the main client for the rest of the run
  (recorded), cancellations pass through, preflight
- a run refuses to start against a dead delegated server and announces
  the delegation in the progress log when it is alive; a server that
  dies mid-run no longer aborts the run
- the Web UI job status / result carry the routing snapshot

No paid API is ever called: fake clients and a loopback fake server.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deep_research_tool.api.delegation import (
    DelegatedLLMClient, DelegatedLLMUnavailable)
from deep_research_tool.config import create_config
from deep_research_tool.utils.helpers import ResearchWarnings


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _Response:
    def __init__(self, content, model="m"):
        self.content = content
        self.model = model
        self.usage = {}


class _Main:
    """Stand-in for the main (cloud) client."""
    model = "gpt-main"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system_prompt=None, **kw):
        self.calls += 1
        return _Response(f"main:{prompt[:10]}")

    def chat(self, messages, system_prompt=None, **kw):
        self.calls += 1
        return _Response("main:chat")

    def analyze_research_query(self, query, context=""):
        self.calls += 1
        return _Response("main:analyze")


class _Local:
    """Stand-in local client: fails after ``ok_calls`` successes."""
    base_url = "http://127.0.0.1:1234/v1"
    backend = "openai_compatible"
    model = "gpt-oss-20b"
    max_tokens = 8192
    timeout = 600.0

    def __init__(self, ok_calls=0, error=None):
        self.calls = 0
        self.ok_calls = ok_calls
        self.error = error or RuntimeError(
            "Local LLM read inactivity exceeded 600s. The server may still "
            "be generating; request was not retried.")

    def _hit(self):
        self.calls += 1
        if self.calls > self.ok_calls:
            raise self.error

    def generate(self, prompt, system_prompt=None, **kw):
        self._hit()
        return _Response("local:ok")

    def chat(self, messages, system_prompt=None, **kw):
        self._hit()
        return _Response("local:chat")

    def analyze_research_query(self, query, context=""):
        self._hit()
        return _Response("local:analyze")


class _Handler(BaseHTTPRequestHandler):
    """Loopback OpenAI-compatible server; records request bodies."""
    bodies = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            _Handler.bodies.append(json.loads(raw))
        except Exception:
            _Handler.bodies.append({})
        body = json.dumps({"id": "x", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OK"},
             "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1,
                      "total_tokens": 6}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def local_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _Handler.bodies = []
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def _tool(config, main_client=None):
    """A tool with only the client wiring built (no search / network)."""
    from deep_research_tool.main import DeepResearchTool
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.config = config
    tool.llm_client = main_client or _Main()
    tool.stage_llm_clients = tool._create_stage_llm_clients()
    return tool


def _fake_factory(**kw):
    return SimpleNamespace(provider=kw["provider"], model=kw.get("model"),
                           base_url=kw.get("base_url"), max_tokens=8192)


# --------------------------------------------------------------------------
# routing scope
# --------------------------------------------------------------------------

class TestRoutingScope:

    def _config(self, **kw):
        base = dict(provider="openai", openai_api_key="t",
                    local_base_url="http://llm.invalid/v1",
                    local_backend="openai_compatible")
        base.update(kw)
        with patch.dict("os.environ", {}, clear=True):
            return create_config(**base)

    def test_verify_role_touches_only_the_finalization_eval_client(self):
        tool = _tool(self._config(local_llm_role="verify"))
        # research-stage clients (extraction / synthesis) are NOT replaced
        assert tool.stage_llm_clients == {}
        assert set(tool.role_llm_clients) == {"verification"}
        eval_llm, writing_llm = tool._finalization_llms()
        assert isinstance(eval_llm, DelegatedLLMClient)
        assert writing_llm is None
        assert eval_llm.delegate.base_url == "http://llm.invalid/v1"
        assert eval_llm.role_label.startswith("最終本文の検証")
        assert "ロールルーティング = verify" in eval_llm.setting_label
        assert eval_llm.fallback is tool.llm_client

    def test_draft_and_all_roles(self):
        tool = _tool(self._config(local_llm_role="draft"))
        assert set(tool.role_llm_clients) == {"revision"}
        eval_llm, writing_llm = tool._finalization_llms()
        assert eval_llm is None and isinstance(writing_llm, DelegatedLLMClient)

        tool = _tool(self._config(local_llm_role="all"))
        assert set(tool.role_llm_clients) == {"verification", "revision"}
        # one local client shared by both roles (one server, one semaphore)
        assert (tool.role_llm_clients["verification"].delegate
                is tool.role_llm_clients["revision"].delegate)
        assert tool.stage_llm_clients == {}

    def test_off_role_creates_nothing(self):
        tool = _tool(self._config(local_llm_role="off"))
        assert tool.role_llm_clients == {} and tool.stage_llm_clients == {}
        assert tool._finalization_llms() == (None, None)

    def test_explicit_stage_override_wins_over_the_role(self):
        cfg = self._config(local_llm_role="all",
                           stage_llm={"evaluation": {"provider": "openai",
                                                     "model": "eval-x"}})
        with patch("deep_research_tool.main.get_client", side_effect=_fake_factory):
            tool = _tool(cfg)
        eval_llm, writing_llm = tool._finalization_llms()
        assert eval_llm is tool.stage_llm_clients["evaluation"]
        assert eval_llm.provider == "openai" and eval_llm.model == "eval-x"
        assert writing_llm is tool.role_llm_clients["revision"]

    def test_role_client_uses_the_configured_local_model(self):
        cfg = self._config(local_llm_role="verify", local_model="my-lmstudio-model")
        with patch("deep_research_tool.main.get_client", side_effect=_fake_factory) as f:
            _tool(cfg)
        assert f.call_args.kwargs["model"] == "my-lmstudio-model"
        assert f.call_args.kwargs["base_url"] == "http://llm.invalid/v1"
        # dataclass default + OpenAI-compatible backend: backend default
        with patch("deep_research_tool.main.get_client", side_effect=_fake_factory) as f:
            _tool(self._config(local_llm_role="verify"))
        assert f.call_args.kwargs["model"] is None
        # dataclass default + Ollama: the default IS an Ollama tag
        with patch("deep_research_tool.main.get_client", side_effect=_fake_factory) as f:
            _tool(self._config(local_llm_role="verify", local_backend="ollama"))
        assert f.call_args.kwargs["model"] == "llama3.1:8b"

    def test_local_stage_override_in_a_cloud_run_is_wrapped(self):
        cfg = self._config(stage_llm={"planning": {"provider": "local"}},
                           local_llm_fallback=False)
        with patch("deep_research_tool.main.get_client", side_effect=_fake_factory):
            tool = _tool(cfg)
        planning = tool.stage_llm_clients["planning"]
        assert isinstance(planning, DelegatedLLMClient)
        assert planning.fallback_enabled is False
        assert planning.role_label == "工程別: 計画"
        assert "工程別のAI切り替え" in planning.setting_label
        # a main-local run uses the same server everywhere: no wrapper
        with patch.dict("os.environ", {}, clear=True):
            cfg2 = create_config(provider="local", local_base_url="http://llm.invalid/v1",
                                 stage_llm={"planning": {"provider": "local"}})
        with patch("deep_research_tool.main.get_client", side_effect=_fake_factory):
            tool2 = _tool(cfg2)
        assert not isinstance(tool2.stage_llm_clients["planning"], DelegatedLLMClient)

    def test_routing_snapshot_names_the_url_source(self):
        tool = _tool(self._config(local_llm_role="verify"))
        snap = tool.llm_routing_snapshot()
        assert snap["default"] == {"provider": "openai", "model": "gpt-5-mini"}
        assert snap["delegated"] is True and snap["tripped"] is False
        assert snap["local_url_source"] == "settings"
        assert snap["roles"]["verification"]["provider"] == "local"
        assert snap["roles"]["verification"]["fallback_provider"] == "_Main"
        with patch.dict("os.environ", {"LOCAL_LLM_BASE_URL": "http://env.invalid/v1"}, clear=True):
            cfg = create_config(provider="openai", openai_api_key="t",
                                local_llm_role="verify")
            tool = _tool(cfg)
            assert tool.llm_routing_snapshot()["local_url_source"] == "env:LOCAL_LLM_BASE_URL"


# --------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------

class TestDelegatedClient:

    def _wrap(self, local, main, **kw):
        return DelegatedLLMClient(
            local, fallback=main, role_label="最終本文の検証（クレーム抽出・判定）",
            setting_label="⚙設定 → ロールルーティング = verify", **kw)

    def test_first_failure_switches_to_the_main_client_and_is_recorded(self):
        collector = ResearchWarnings.bind()
        try:
            local, main = _Local(ok_calls=0), _Main()
            d = self._wrap(local, main)
            r1 = d.generate("hello")
            r2 = d.chat([{"role": "user", "content": "x"}])
            assert r1.content.startswith("main:") and r2.content == "main:chat"
            assert local.calls == 1            # never asked again
            assert main.calls == 2 and d.tripped and d.fallback_calls == 2
            msgs = [w["message"] for w in collector.to_dict_list()]
            assert len(msgs) == 1
            assert "ローカルLLMへ委譲した最終本文の検証" in msgs[0]
            assert "verify" in msgs[0] and "inactivity" in msgs[0]
            desc = d.describe()
            assert desc["tripped"] and desc["failures"] == 1
            assert desc["events"][0]["action"] == "fallback"
            assert desc["fallback_provider"] == "_Main"
        finally:
            ResearchWarnings.unbind()

    def test_without_fallback_the_error_names_the_setting(self):
        local, main = _Local(ok_calls=0), _Main()
        d = self._wrap(local, main, fallback_enabled=False)
        with pytest.raises(RuntimeError) as ei:
            d.generate("hello")
        text = str(ei.value)
        assert text.startswith("[ローカルLLM委譲: 最終本文の検証")
        assert "ロールルーティング = verify" in text and "inactivity" in text
        assert main.calls == 0 and not d.tripped
        assert d.describe()["events"][0]["action"] == "raise"

    def test_cancellation_passes_through_untouched(self):
        from deep_research_tool.utils.cancellation import RunCancelled
        local, main = _Local(ok_calls=0, error=RunCancelled("cancelled")), _Main()
        d = self._wrap(local, main)
        with pytest.raises(RunCancelled):
            d.generate("hello")
        assert main.calls == 0 and not d.tripped and d.failures == 0

    def test_attributes_and_helpers_go_through_the_wrapper(self):
        local, main = _Local(ok_calls=1), _Main()
        d = self._wrap(local, main)
        assert d.model == "gpt-oss-20b" and d.base_url == local.base_url
        d.concurrency_limiter = "LIMITER"
        d.token_stats = "STATS"
        d.cancel_check = "CHECK"
        assert local.concurrency_limiter == "LIMITER"
        assert local.token_stats == "STATS" and local.cancel_check == "CHECK"
        # a public LLM helper is routed with the same fallback logic
        assert d.analyze_research_query("q").content == "local:analyze"
        assert d.analyze_research_query("q").content == "main:analyze"
        assert d.tripped and main.calls == 1

    def test_preflight_against_a_dead_port_names_the_setting(self):
        from deep_research_tool.api import get_client
        client = get_client(provider="local", api_key="x", model="m",
                            base_url="http://127.0.0.1:9/v1",
                            backend="openai_compatible", local_timeout=5)
        d = self._wrap(client, _Main())
        with pytest.raises(DelegatedLLMUnavailable) as ei:
            d.preflight()
        text = str(ei.value)
        assert "疎通確認に失敗" in text and "ロールルーティング = verify" in text
        assert "http://127.0.0.1:9/v1" in text and "調査は開始していません" in text
        assert ei.value.base_url == "http://127.0.0.1:9/v1"
        assert d.preflight_result["ok"] is False
        assert client.max_tokens == 8192 and client.timeout == 5.0   # restored

    def test_preflight_against_a_live_server_sends_one_tiny_request(self, local_server):
        from deep_research_tool.api import get_client
        client = get_client(provider="local", api_key="x", model="m",
                            base_url=local_server, backend="openai_compatible")
        d = self._wrap(client, _Main())
        result = d.preflight()
        assert result["ok"] is True and d.describe()["preflight"]["ok"] is True
        assert len(_Handler.bodies) == 1
        assert _Handler.bodies[0].get("max_tokens") == 8
        assert client.max_tokens == 8192


# --------------------------------------------------------------------------
# inside a run
# --------------------------------------------------------------------------

def _build_run_tool(tmp_path, verification):
    from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
    from deep_research_tool.tests.test_finalization_integration import build_tool
    tool = build_tool(tmp_path, FakeLLM(), FakeSearchClient(),
                      enable_verification=verification, auto_figures=False,
                      verification_profile="balanced", plan_review=False)
    rc = tool.config.research
    rc.use_enhanced_synthesis = False
    rc.max_gap_fill_rounds = 0
    rc.max_queries_per_iteration = 2
    rc.max_pages_per_query = 2
    rc.content_filter_mode = type(rc.content_filter_mode)("none")
    rc.min_iterations = 1
    rc.max_iterations = 1
    return tool


class TestRun:

    def test_run_refuses_to_start_against_a_dead_delegated_server(self, tmp_path):
        from deep_research_tool.api import get_client
        tool = _build_run_tool(tmp_path, verification=False)
        dead = get_client(provider="local", api_key="x", model="m",
                          base_url="http://127.0.0.1:9/v1",
                          backend="openai_compatible", local_timeout=5)
        tool.role_llm_clients = {"verification": DelegatedLLMClient(
            dead, fallback=tool.llm_client,
            role_label="最終本文の検証（クレーム抽出・判定）",
            setting_label="⚙設定 → ロールルーティング = verify")}
        msgs = []
        with pytest.raises(DelegatedLLMUnavailable) as ei:
            tool.run(query="炭素繊維の市場調査", requirements="目次は2章構成でよい",
                     progress_callback=lambda m, p: msgs.append(m))
        assert "ロールルーティング = verify" in str(ei.value)
        assert any(m.startswith("ローカルLLMへ委譲:") for m in msgs)
        assert not any("plan" in m.lower() for m in msgs)     # research never began
        assert not list(tmp_path.glob("**/session_*.json"))

    def test_live_delegation_is_announced_and_a_mid_run_failure_falls_back(self, tmp_path):
        tool = _build_run_tool(tmp_path, verification=True)
        # healthy for the preflight, dead afterwards (as a server that
        # unloads its model or times out hours into a run)
        local = _Local(ok_calls=1)
        role = DelegatedLLMClient(
            local, fallback=tool.llm_client,
            role_label="最終本文の検証（クレーム抽出・判定）",
            setting_label="⚙設定 → ロールルーティング = verify")
        tool.role_llm_clients = {"verification": role}
        msgs = []
        result = tool.run(query="炭素繊維の市場調査", requirements="目次は2章構成でよい",
                          progress_callback=lambda m, p: msgs.append(m))
        assert any(m.startswith("ローカルLLMへ委譲: 最終本文の検証") for m in msgs)
        assert any(m.startswith("ローカルLLMの疎通確認 OK") for m in msgs)
        # the run COMPLETED on the main client instead of aborting
        assert result.get("report_path")
        routing = result["llm_routing"]
        assert routing["tripped"] is True
        role_snap = routing["roles"]["verification"]
        assert role_snap["preflight"]["ok"] is True
        assert role_snap["tripped"] and role_snap["fallback_calls"] >= 1
        assert local.calls == 2        # preflight + the one failed call
        warnings = [w["message"] for w in result.get("warnings") or []]
        assert any("以後はメインのプロバイダ" in w for w in warnings)
        perf = json.loads((tmp_path / f"performance_{result['session_id']}.json")
                          .read_text(encoding="utf-8"))
        assert perf["llm_routing"]["roles"]["verification"]["tripped"] is True


# --------------------------------------------------------------------------
# Web UI plumbing
# --------------------------------------------------------------------------

class TestWebUI:

    def test_job_status_carries_the_routing_snapshot(self):
        from deep_research_tool.webui.server import ResearchJob
        job = ResearchJob("job-r", "q", params={"query": "q"})
        assert job.to_dict()["llm_routing"] is None
        job.routing_source = lambda: {"default": {"provider": "openai", "model": "m"},
                                      "roles": {}, "stages": {}, "delegated": False}
        assert job.to_dict()["llm_routing"]["default"]["provider"] == "openai"

    def test_settings_params_reach_the_config(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kwargs = build_config_kwargs({
            "provider": "openai", "local_base_url": "http://127.0.0.1:1234/v1",
            "local_llm_role": "draft", "local_model": "lmstudio-x",
            "local_llm_fallback": False})
        assert kwargs["local_model"] == "lmstudio-x"
        assert kwargs["local_llm_fallback"] is False
        assert kwargs["local_llm_role"] == "draft"
        with patch.dict("os.environ", {}, clear=True):
            cfg = create_config(openai_api_key="t", **kwargs)
        assert cfg.api.local_model == "lmstudio-x"
        assert cfg.research.local_llm_fallback is False
        # a main-local run: the form's model still wins over local_model
        with patch.dict("os.environ", {}, clear=True):
            cfg = create_config(provider="local", local_base_url="http://x/v1",
                                local_model="a", model="b")
        assert cfg.api.local_model == "b"
