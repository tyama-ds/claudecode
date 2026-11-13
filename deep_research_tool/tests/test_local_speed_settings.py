"""Settings that cut local-LLM research time (v1.14.0).

local_think / local_num_ctx shape the local client's requests;
extraction_max_tokens / extraction_max_chunks shape page extraction.
None of them changes anything unless it is set: the payloads and the
number of extraction calls stay exactly as before. No network is used —
the client's transport is replaced by a recorder, and extraction runs on
a fake LLM.
"""

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_research_tool.api import get_client
from deep_research_tool.api.local_client import LocalLLMClient
from deep_research_tool.config import create_config
from deep_research_tool.research.content_extractor import ContentExtractor


# --------------------------------------------------------------------------
# local client payloads
# --------------------------------------------------------------------------

OLLAMA_DONE = {"model": "m", "message": {"role": "assistant", "content": "ok"},
               "done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2}
OPENAI_DONE = {"model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": "ok"}}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}


def _client(backend, reply=None, **kw):
    """A local client whose transport records the payload instead of
    posting it."""
    base = "http://127.0.0.1:9" if backend == "ollama" else "http://127.0.0.1:9/v1"
    c = LocalLLMClient(model="qwen3:8b", backend=backend, base_url=base, **kw)
    sent = []

    def post(url, payload):
        sent.append((url, json.loads(json.dumps(payload))))
        data = json.loads(json.dumps(reply or (OLLAMA_DONE if backend == "ollama" else OPENAI_DONE)))
        return data
    c._post_with_retry = post
    return c, sent


class TestPayloadsUnchangedByDefault:
    def test_ollama_payload_has_only_the_old_keys(self):
        c, sent = _client("ollama")
        c.generate("hi")
        url, payload = sent[0]
        assert url.endswith("/api/chat")
        assert set(payload) == {"model", "messages", "stream", "options"}
        assert set(payload["options"]) == {"temperature", "num_predict"}

    def test_openai_compatible_payload_has_only_the_old_keys(self):
        c, sent = _client("openai_compatible")
        c.generate("hi")
        url, payload = sent[0]
        assert url.endswith("/v1/chat/completions")
        assert set(payload) == {"model", "messages", "temperature", "max_tokens",
                                "stream", "stream_options"}

    def test_get_client_passes_nothing_when_unset(self):
        c = get_client(provider="local", backend="ollama", base_url="http://127.0.0.1:9")
        assert c.think is None and c.num_ctx is None


class TestThink:
    @pytest.mark.parametrize("value", [False, True])
    def test_ollama_gets_a_top_level_think(self, value):
        c, sent = _client("ollama", think=value)
        c.generate("hi")
        payload = sent[0][1]
        assert payload["think"] is value
        assert "think" not in payload["options"]
        assert "chat_template_kwargs" not in payload

    @pytest.mark.parametrize("value", [False, True])
    def test_openai_compatible_gets_chat_template_kwargs(self, value):
        c, sent = _client("openai_compatible", think=value)
        c.generate("hi")
        payload = sent[0][1]
        assert payload["chat_template_kwargs"] == {"enable_thinking": value}
        assert "think" not in payload

    def test_get_client_forwards_it(self):
        c = get_client(provider="local", backend="ollama", base_url="http://127.0.0.1:9",
                       local_think=False, local_num_ctx=8192)
        assert c.think is False and c.num_ctx == 8192

    @pytest.mark.parametrize("bad", ["false", 0, 1, "off"])
    def test_client_rejects_a_non_bool(self, bad):
        with pytest.raises(ValueError):
            LocalLLMClient(model="m", backend="ollama", base_url="http://127.0.0.1:9", think=bad)


class TestThinkBlockIsRemoved:
    @pytest.mark.parametrize("backend", ["ollama", "openai_compatible"])
    def test_a_think_block_never_reaches_the_caller(self, backend):
        text = '<think>\nThe user wants JSON. {"draft": 1}\n</think>\n\n{"answer": 42}'
        if backend == "ollama":
            reply = {**OLLAMA_DONE, "message": {"role": "assistant", "content": text}}
        else:
            reply = json.loads(json.dumps(OPENAI_DONE))
            reply["choices"][0]["message"]["content"] = text
        c, _ = _client(backend, reply=reply)
        out = c.generate("hi").content
        assert out == '{"answer": 42}'
        assert json.loads(out) == {"answer": 42}

    def test_other_shapes_of_the_block(self):
        strip = LocalLLMClient._strip_think
        # the template opened <think>, so only the closing tag is in the text
        assert strip('reasoning...</think>\n{"a": 1}') == '{"a": 1}'
        # generation stopped while thinking: no answer was written
        assert strip("<think>still thinking") == ""
        assert strip('<THINK>x</THINK>{"a": 1}') == '{"a": 1}'
        # a reply without a block is returned exactly as it came
        assert strip('  {"a": 1}\n') == '  {"a": 1}\n'


class TestNumCtx:
    def test_ollama_gets_options_num_ctx(self):
        c, sent = _client("ollama", num_ctx=16384)
        c.generate("hi")
        assert sent[0][1]["options"]["num_ctx"] == 16384

    def test_openai_compatible_never_gets_it(self):
        c, sent = _client("openai_compatible", num_ctx=16384)
        c.generate("hi")
        payload = sent[0][1]
        assert "num_ctx" not in json.dumps(payload) and "options" not in payload

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "4096"])
    def test_client_rejects_a_bad_value(self, bad):
        with pytest.raises(ValueError):
            LocalLLMClient(model="m", backend="ollama", base_url="http://127.0.0.1:9", num_ctx=bad)


# --------------------------------------------------------------------------
# config / CLI / Web UI
# --------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        c = create_config()
        assert c.api.local_think is None and c.api.local_num_ctx is None
        assert c.research.extraction_max_tokens is None
        assert c.research.extraction_max_chunks == 8

    def test_values_reach_the_config(self):
        c = create_config(local_think=False, local_num_ctx=8192,
                          extraction_max_tokens=1500, extraction_max_chunks=3)
        assert c.api.local_think is False and c.api.local_num_ctx == 8192
        assert c.research.extraction_max_tokens == 1500
        assert c.research.extraction_max_chunks == 3

    @pytest.mark.parametrize("kw,msg", [
        ({"local_think": "false"}, "local_think must be True, False or None"),
        ({"local_think": 0}, "local_think must be True, False or None"),
        ({"local_num_ctx": 0}, "local_num_ctx must be a positive integer"),
        ({"local_num_ctx": 2.5}, "local_num_ctx must be a positive integer"),
        ({"extraction_max_tokens": True}, "extraction_max_tokens must be a positive integer"),
        ({"extraction_max_tokens": -5}, "extraction_max_tokens must be a positive integer"),
        ({"extraction_max_chunks": 9}, "extraction_max_chunks must be between 1 and 8"),
        ({"extraction_max_chunks": 0}, "extraction_max_chunks must be between 1 and 8"),
        ({"extraction_max_chunks": 2.0}, "extraction_max_chunks must be an integer between 1 and 8"),
    ])
    def test_bad_values_are_errors(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            create_config(**kw)

    def test_validate_catches_a_config_changed_directly(self):
        c = create_config()
        c.api.local_think = "yes"
        c.api.local_num_ctx = 0
        c.research.extraction_max_tokens = 0
        c.research.extraction_max_chunks = 12
        errors = " | ".join(c.validate())
        for name in ("local_think", "local_num_ctx", "extraction_max_tokens",
                     "extraction_max_chunks"):
            assert name in errors, errors


class TestWebUiParams:
    def test_mapped_and_converted(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kw = build_config_kwargs({"local_think": "false", "local_num_ctx": "8192",
                                  "extraction_max_tokens": 1500, "extraction_max_chunks": "3"})
        assert kw == {"local_think": False, "local_num_ctx": 8192,
                      "extraction_max_tokens": 1500, "extraction_max_chunks": 3}
        cfg = create_config(**kw)
        assert cfg.api.local_think is False and cfg.research.extraction_max_chunks == 3

    def test_empty_values_leave_the_defaults(self):
        from deep_research_tool.webui.server import build_config_kwargs
        assert build_config_kwargs({"local_think": "", "local_num_ctx": "",
                                    "extraction_max_tokens": "", "extraction_max_chunks": ""}) == {}

    @pytest.mark.parametrize("params", [{"local_think": "maybe"}, {"local_num_ctx": 0},
                                        {"extraction_max_tokens": 1.5},
                                        {"extraction_max_chunks": 9}])
    def test_bad_values_name_the_field(self, params):
        from deep_research_tool.webui.server import FieldError, build_config_kwargs
        with pytest.raises(FieldError) as e:
            build_config_kwargs(params)
        assert e.value.field == next(iter(params))


class TestCli:
    def test_options_reach_create_config(self):
        from click.testing import CliRunner
        import deep_research_tool.cli as cli_mod
        seen = {}

        class _Stop(Exception):
            pass

        def fake_create_config(**kw):
            seen.update(kw)
            raise _Stop()
        with patch.object(cli_mod, "create_config", fake_create_config):
            CliRunner().invoke(cli_mod.cli, [
                "research", "q", "--provider", "local", "--local-think", "off",
                "--local-num-ctx", "8192", "--extraction-max-tokens", "1500",
                "--extraction-max-chunks", "3"])
        assert seen["local_think"] is False and seen["local_num_ctx"] == 8192
        assert seen["extraction_max_tokens"] == 1500 and seen["extraction_max_chunks"] == 3
        seen.clear()
        with patch.object(cli_mod, "create_config", fake_create_config):
            CliRunner().invoke(cli_mod.cli, ["research", "q"])
        assert seen["local_think"] is None and seen["local_num_ctx"] is None
        assert seen["extraction_max_tokens"] is None and seen["extraction_max_chunks"] is None

    def test_out_of_range_is_rejected(self):
        from click.testing import CliRunner
        import deep_research_tool.cli as cli_mod
        r = CliRunner().invoke(cli_mod.cli, ["research", "q", "--extraction-max-chunks", "9"])
        assert r.exit_code != 0 and "extraction-max-chunks" in r.output


class TestMainWiring:
    def test_local_main_client_gets_the_settings(self):
        from deep_research_tool.main import DeepResearchTool
        cfg = create_config(provider="local", local_base_url="http://127.0.0.1:9",
                            local_think=False, local_num_ctx=4096)
        tool = DeepResearchTool(cfg)
        assert tool.llm_client.think is False and tool.llm_client.num_ctx == 4096

    def test_delegated_local_client_gets_the_settings(self):
        from deep_research_tool.main import DeepResearchTool
        cfg = create_config(provider="openai", openai_api_key="sk-test",
                            local_llm_role="verify", local_base_url="http://127.0.0.1:9",
                            local_think=True, local_num_ctx=8192)
        tool = DeepResearchTool(cfg)
        local = tool.role_llm_clients["verification"].delegate
        assert local.think is True and local.num_ctx == 8192

    def test_researcher_gets_the_extraction_settings(self):
        import inspect
        import deep_research_tool.main as main_mod
        src = inspect.getsource(main_mod.DeepResearchTool)
        assert "extraction_max_tokens=self.config.research.extraction_max_tokens" in src
        assert "extraction_max_chunks=self.config.research.extraction_max_chunks" in src
        from deep_research_tool.research.researcher import Researcher
        r = Researcher(llm_client=MagicMock(), search_client=MagicMock(),
                       output_dir=Path(tempfile.mkdtemp()), language="ja",
                       extraction_max_tokens=1500, extraction_max_chunks=3)
        assert r.content_extractor.extraction_max_tokens == 1500
        assert r.content_extractor.MAX_CHUNKS == 3
        r = Researcher(llm_client=MagicMock(), search_client=MagicMock(),
                       output_dir=Path(tempfile.mkdtemp()), language="ja")
        assert r.content_extractor.extraction_max_tokens is None
        assert r.content_extractor.MAX_CHUNKS == ContentExtractor.MAX_CHUNKS == 8


# --------------------------------------------------------------------------
# page extraction
# --------------------------------------------------------------------------

GOOD = json.dumps({"processed_content": "炭素繊維の需要は航空機向けが中心である。" * 3,
                   "key_points": ["航空機向けが中心"], "quotes": [],
                   "relevance_score": 0.8, "extraction_notes": ""}, ensure_ascii=False)


class FakeLLM:
    """Records every generate() call (prompt and keyword arguments)."""

    def __init__(self, reply=GOOD):
        self.reply = reply
        self.calls = []
        self._lock = threading.Lock()
        self.model = "fake"

    def generate(self, prompt, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        return MagicMock(content=self.reply)


LONG_PAGE = ("炭素繊維の国内市場は航空機向けと自動車向けの需要で伸びている。" * 2000)   # ~64k chars


def _extract(ext, page=LONG_PAGE, title="資料"):
    return ext.extract_relevant_content(page, "https://example.com/a", title,
                                        "炭素繊維の需要", "炭素繊維市場")


class TestExtractionChunks:
    def test_default_reads_up_to_eight_chunks(self):
        llm = FakeLLM()
        _extract(ContentExtractor(llm, "ja"))
        assert len(llm.calls) == 8

    def test_max_chunks_three_means_at_most_three_calls(self):
        llm = FakeLLM()
        out = _extract(ContentExtractor(llm, "ja", max_chunks=3))
        assert 1 <= len(llm.calls) <= 3
        assert out.metadata["extraction_complete"] is True
        assert len(out.metadata["source_spans"]) == len(llm.calls)

    def test_with_the_researchers_content_budget(self):
        # Researcher sets max_selected_chars = max_content_length (50,000)
        llm = FakeLLM()
        ext = ContentExtractor(llm, "ja", max_chunks=3)
        ext.max_selected_chars = 50000
        _extract(ext)
        assert len(llm.calls) <= 3

    def test_the_chunk_count_is_part_of_the_cache_key(self):
        llm = FakeLLM()
        ext = ContentExtractor(llm, "ja", max_chunks=3)
        _extract(ext)
        n = len(llm.calls)
        _extract(ext)
        assert len(llm.calls) == n            # same settings: cached
        ext.MAX_CHUNKS = 2
        _extract(ext)
        assert len(llm.calls) == n + 2        # another chunk count: extracted again


class TestExtractionMaxTokens:
    def test_passed_to_every_chunk_call(self):
        llm = FakeLLM()
        _extract(ContentExtractor(llm, "ja", extraction_max_tokens=1500, max_chunks=3))
        assert llm.calls and all(kw == {"max_tokens": 1500} for kw in llm.calls)

    def test_not_passed_when_unset(self):
        llm = FakeLLM()
        _extract(ContentExtractor(llm, "ja"), page="短いページ。炭素繊維の需要。")
        assert llm.calls == [{}]

    def test_part_of_the_cache_key(self):
        llm = FakeLLM()
        ext = ContentExtractor(llm, "ja", extraction_max_tokens=1500)
        _extract(ext, page="短いページ。炭素繊維の需要。")
        ext.extraction_max_tokens = 3000
        _extract(ext, page="短いページ。炭素繊維の需要。")
        assert [kw.get("max_tokens") for kw in llm.calls] == [1500, 3000]

    def test_json_cut_at_the_cap_is_not_a_success(self):
        cut = GOOD[: len(GOOD) // 2]           # the reply stopped at max_tokens
        llm = FakeLLM(reply=cut)
        ext = ContentExtractor(llm, "ja", extraction_max_tokens=200)
        out = _extract(ext, page="短いページ。炭素繊維の需要。")
        assert out.metadata["extraction_complete"] is False
        assert out.metadata["source_spans"] == []
        assert "failed" in out.extraction_notes.lower()
        assert out.key_points == []
        # a failure is not cached: the next request tries again
        _extract(ext, page="短いページ。炭素繊維の需要。")
        assert len(llm.calls) == 2

    def test_a_cut_chunk_of_a_long_page_counts_as_failed(self):
        replies = iter([GOOD, GOOD[:40], GOOD])

        class Mixed(FakeLLM):
            def generate(self, prompt, **kwargs):
                with self._lock:
                    self.calls.append(kwargs)
                    return MagicMock(content=next(replies))
        llm = Mixed()
        ext = ContentExtractor(llm, "ja", extraction_max_tokens=300, max_chunks=3,
                               max_parallel_workers=1)
        out = _extract(ext)
        assert len(llm.calls) == 3
        assert out.metadata["extraction_complete"] is False
        assert len(out.metadata["failed_spans"]) == 1
        assert "Partial extraction" in out.extraction_notes


# --------------------------------------------------------------------------
# disk cache of extractions (cache_reuse)
# --------------------------------------------------------------------------

from deep_research_tool.research.fetch_cache import (  # noqa: E402
    DEFAULT_EXTRACTION_MAX_CHUNKS, EXTRACTION_PROMPT_VERSION, FetchCache,
    content_hash)

_KEY_ARGS = ("本文", "1. 章", "炭素繊維市場", "fake", EXTRACTION_PROMPT_VERSION)


class TestExtractionDiskCacheKey:
    def test_defaults_keep_the_old_key(self, tmp_path):
        cache = FetchCache(tmp_path / "c")
        # the key the versions before the setting wrote
        old = FetchCache._key("extract", content_hash("本文"), "1. 章",
                              "炭素繊維市場", "fake", EXTRACTION_PROMPT_VERSION)
        assert cache.extraction_key(*_KEY_ARGS) == old
        assert cache.extraction_key(*_KEY_ARGS, max_chunks=None,
                                    max_tokens=None) == old
        assert cache.extraction_key(*_KEY_ARGS, max_chunks=8,
                                    max_tokens=None) == old

    def test_the_default_matches_the_extractor_and_config(self):
        from deep_research_tool import config as cfg_mod
        assert DEFAULT_EXTRACTION_MAX_CHUNKS == ContentExtractor.MAX_CHUNKS
        assert DEFAULT_EXTRACTION_MAX_CHUNKS == cfg_mod.DEFAULT_EXTRACTION_MAX_CHUNKS

    def test_changed_settings_give_other_keys(self, tmp_path):
        cache = FetchCache(tmp_path / "c")
        keys = [
            cache.extraction_key(*_KEY_ARGS),
            cache.extraction_key(*_KEY_ARGS, max_chunks=3),
            cache.extraction_key(*_KEY_ARGS, max_chunks=2),
            cache.extraction_key(*_KEY_ARGS, max_tokens=3000),
            cache.extraction_key(*_KEY_ARGS, max_tokens=1500),
            cache.extraction_key(*_KEY_ARGS, max_chunks=3, max_tokens=3000),
        ]
        assert len(set(keys)) == len(keys)
        # same settings: same key
        assert cache.extraction_key(*_KEY_ARGS, max_chunks=3, max_tokens=3000) \
            == keys[-1]

    def _researcher(self, tmp_path, llm, **kw):
        from deep_research_tool.research.researcher import Researcher
        r = Researcher(llm_client=llm, search_client=MagicMock(),
                       output_dir=tmp_path / "out", language="ja",
                       fetch_cache=FetchCache(tmp_path / "cache"), **kw)
        r.content_extractor.eval_llm = llm
        return r

    def _run(self, r):
        return r._extract_cached(LONG_PAGE, "https://example.com/a", "資料",
                                 "1. 炭素繊維の需要", "炭素繊維市場")

    def test_three_chunk_result_is_not_reused_after_going_back_to_eight(self, tmp_path):
        llm = FakeLLM()
        out = self._run(self._researcher(tmp_path, llm, extraction_max_chunks=3))
        assert "[cache]" not in out.extraction_notes
        three = len(llm.calls)
        assert 1 <= three <= 3
        # same settings in a new run: read from the disk cache
        out = self._run(self._researcher(tmp_path, llm, extraction_max_chunks=3))
        assert "[cache]" in out.extraction_notes and len(llm.calls) == three
        # back to the default 8: extracted again, not the 3-chunk result
        out = self._run(self._researcher(tmp_path, llm))
        assert "[cache]" not in out.extraction_notes
        assert len(llm.calls) == three + 8

    def test_max_tokens_result_is_not_reused_without_it(self, tmp_path):
        llm = FakeLLM()
        self._run(self._researcher(tmp_path, llm, extraction_max_tokens=3000))
        n = len(llm.calls)
        out = self._run(self._researcher(tmp_path, llm))
        assert "[cache]" not in out.extraction_notes
        assert len(llm.calls) == n + 8
        assert all("max_tokens" not in kw for kw in llm.calls[n:])

    def test_default_run_reads_an_entry_stored_under_the_old_key(self, tmp_path):
        cache = FetchCache(tmp_path / "cache")
        old = FetchCache._key("extract", content_hash(LONG_PAGE), "1. 炭素繊維の需要",
                              "炭素繊維市場", "fake", EXTRACTION_PROMPT_VERSION)
        cache.put_extraction(old, {"processed_content": "以前の抽出結果",
                                   "key_points": [], "quotes": [],
                                   "relevance_score": 0.7,
                                   "extraction_notes": "", "metadata": {}})
        llm = FakeLLM()
        out = self._run(self._researcher(tmp_path, llm))
        assert out.processed_content == "以前の抽出結果"
        assert llm.calls == []
