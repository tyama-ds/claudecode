"""
The progress bar is ONE overall 0–100 scale:
- every stage reports its own 0–100 and the plan maps it onto that
  stage's slice (weights depend on the enabled features)
- 100 is reported once, when the run is complete — not when the research
  loop ends, not when the multilingual search finishes a section
- the Web UI job exposes the current stage and the live overall value
  (verification counters move the bar while claims are judged)
No paid API: fake clients only.
"""

from types import SimpleNamespace

import pytest

from deep_research_tool.utils.progress import ProgressPlan, STAGE_LABELS


def _config(*, v1=True, verification=True, deep_think=False, fermi=False, figures=False):
    return SimpleNamespace(
        report=SimpleNamespace(generator_version=SimpleNamespace(value="v1" if v1 else "v2"),
                               auto_figures=figures),
        deep_think=SimpleNamespace(enabled=deep_think),
        fermi_estimation=SimpleNamespace(enabled=fermi),
        enable_verification=verification,
    )


class TestPlan:

    def test_slices_cover_exactly_0_to_100_in_run_order(self):
        plan = ProgressPlan.for_config(_config(v1=True, verification=True, deep_think=True,
                                               fermi=True, figures=True))
        assert plan.order == ["plan", "research", "deep_think", "evidence_review", "fermi",
                              "figures", "verification", "writing", "rendering"]
        pos = 0.0
        for key in plan.order:
            start, end = plan.ranges[key]
            assert start == pytest.approx(pos) and end > start
            pos = end
        assert pos == 100.0
        # V2/V3: chapters are written BEFORE verification
        plan2 = ProgressPlan.for_config(_config(v1=False, verification=True))
        assert plan2.order == ["plan", "research", "writing", "verification", "rendering"]
        # disabled features are absent and the research slice grows
        plan3 = ProgressPlan.for_config(_config(v1=True, verification=False))
        assert "verification" not in plan3.ranges
        assert plan3.ranges["research"][1] > plan.ranges["research"][1]

    def test_stage_relative_values_map_into_the_slice_and_never_go_back(self):
        plan = ProgressPlan([("a", 20), ("b", 80)])
        seen = []
        cb = plan.callback(lambda m, p: seen.append((m, p)))
        cb("half of a", 50)
        cb("done a", 100)
        plan.enter("b")
        cb("start b", 0)
        cb("a note", None)
        cb("error", -1)
        cb("quarter b", 25)
        assert [p for _, p in seen] == [10.0, 20.0, 20.0, 20.0, -1.0, 40.0]
        # a lower stage value never lowers the bar
        cb("late report of 10", 10)
        assert seen[-1][1] == 40.0
        # entering an unknown (disabled) stage is ignored
        plan.enter("nope")
        assert plan.current == "b"

    def test_researcher_values_split_into_plan_and_research(self):
        plan = ProgressPlan.for_config(_config())
        seen = []
        cb = plan.research_callback(lambda m, p: seen.append(p))
        cb("Creating research plan...", 5)
        assert plan.current == "plan" and 0 < seen[-1] < plan.ranges["plan"][1]
        cb("Research plan created", 10)
        assert seen[-1] == pytest.approx(plan.ranges["plan"][1])
        cb("Waiting for plan review...", 10)
        assert plan.current == "plan"              # still planning
        cb("Researching: 1. 市場規模", 10)
        assert plan.current == "research"          # the first section starts at 10
        cb("Research completed!", 100)
        assert seen[-1] == pytest.approx(plan.ranges["research"][1])
        assert seen[-1] < 70                      # NOT the end of the bar

    def test_verification_counters_move_the_bar_inside_the_slice(self):
        plan = ProgressPlan.for_config(_config(v1=True, verification=True))
        plan.enter("verification")
        start, end = plan.ranges["verification"]
        assert plan.live(None) == pytest.approx(start)
        judging = {"phase": "judging", "round": 1, "max_rounds": 1,
                   "claims_done": 50, "claims_total": 100}
        value = plan.live(judging)
        assert start < value < end
        assert plan.verification_fraction(judging) == pytest.approx(0.15 + 0.375)
        assert plan.verification_fraction({"phase": "done"}) == 1.0
        assert plan.verification_fraction({"phase": "idle"}) == 0.0
        snap = plan.snapshot(judging)
        assert snap["stage"]["key"] == "verification"
        assert snap["stage"]["label"] == STAGE_LABELS["verification"]
        assert snap["stage"]["step"] == 3
        assert snap["overall"] == pytest.approx(value, abs=0.1)
        # a second round resets the counters (phase back to extracting,
        # 0 chunks done): the bar keeps the highest value it reached
        assert plan.live({"phase": "extracting", "round": 2, "max_rounds": 3,
                          "chunks_done": 0, "chunks_total": 9}) >= value
        # and the live value never exceeds the slice
        assert plan.live({"phase": "coverage", "round": 3, "max_rounds": 1}) <= end

    def test_done_is_the_only_100(self):
        plan = ProgressPlan.for_config(_config())
        seen = []
        cb = plan.callback(lambda m, p: seen.append(p))
        for key in plan.order:
            plan.enter(key)
            cb("end of " + key, 100)
        assert max(seen) < 100.0
        plan.done(lambda m, p: seen.append(p))
        assert seen[-1] == 100.0 and plan.current == "done"


class TestResearcherSubProgress:

    def test_multilingual_search_reports_inside_the_section_slice(self):
        from deep_research_tool.research.researcher import Researcher
        r = Researcher.__new__(Researcher)
        seen = []
        r.progress_callback = lambda m, p: seen.append((m, p))
        r._section_slice = (45.0, 7.0)             # 6th of 10 sections
        r._sub_progress("Translating queries...", 10)
        r._sub_progress("Multilingual search complete", 100)
        r._sub_progress("note", -1)
        assert [p for _, p in seen] == pytest.approx([45.0 + 0.42, 45.0 + 4.2, 45.0])
        assert all(p < 85 for _, p in seen)       # never the whole bar

    def test_multilingual_searcher_is_wired_to_the_section_scale(self):
        from deep_research_tool.config import MultilingualSearchConfig
        from deep_research_tool.research.researcher import Researcher
        from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
        r = Researcher(llm_client=FakeLLM(), search_client=FakeSearchClient(),
                       multilingual_config=MultilingualSearchConfig(enabled=True),
                       progress_callback=lambda m, p: None)
        assert r.multilingual_searcher is not None
        assert r.multilingual_searcher.progress_callback == r._sub_progress


class TestRun:

    def test_a_full_run_reports_a_single_final_100(self, tmp_path):
        from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
        from deep_research_tool.tests.test_finalization_integration import build_tool
        tool = build_tool(tmp_path, FakeLLM(), FakeSearchClient(),
                          enable_verification=True, auto_figures=False,
                          verification_profile="balanced", plan_review=False)
        rc = tool.config.research
        rc.use_enhanced_synthesis = False
        rc.max_gap_fill_rounds = 0
        rc.max_queries_per_iteration = 2
        rc.max_pages_per_query = 2
        rc.content_filter_mode = type(rc.content_filter_mode)("none")
        rc.min_iterations = 1
        rc.max_iterations = 1
        seen = []
        tool.run(query="炭素繊維の市場調査", requirements="目次は2章構成でよい",
                 progress_callback=lambda m, p: seen.append((m, p)))
        values = [p for _, p in seen if p >= 0]
        assert values == sorted(values)                     # monotonic
        assert values[-1] == 100.0 and values.count(100.0) == 1
        by_msg = {m: p for m, p in seen}
        assert by_msg["Research completed!"] < 70
        assert by_msg["Verifying final report body..."] > by_msg["Research completed!"]
        assert by_msg["Generating report..."] > by_msg["Verifying final report body..."]
        assert by_msg["Completed"] == 100.0
        snap = tool.progress_snapshot()
        assert snap["stage"]["key"] == "done" and snap["overall"] == 100.0
        # the researched plan stays available (Web UI viewer / history)
        plan = tool.current_plan()
        assert plan and len(plan["table_of_contents"]["items"]) == 2
        assert plan["search_queries"]

    def test_job_status_carries_the_live_plan(self):
        from deep_research_tool.webui.server import ResearchJob
        job = ResearchJob("job-plan", "q", params={"query": "q"})
        assert job.to_dict()["plan"] is None
        job.plan_source = lambda: {"title": "t", "table_of_contents": {"items": []},
                                   "search_queries": ["q1"]}
        assert job.to_dict()["plan"]["title"] == "t"
        # a reviewed plan wins over the live one while it is set
        job.plan = {"title": "reviewed", "table_of_contents": {"items": []}}
        assert job.to_dict()["plan"]["title"] == "reviewed"


class TestWebUI:

    def test_job_status_uses_the_live_plan_value_and_names_the_stage(self):
        from deep_research_tool.webui.server import ResearchJob
        job = ResearchJob("job-p", "q", params={"query": "q"})
        job.update("Verifying final report body...", 75.0)
        d = job.to_dict()
        assert d["progress"] == 75.0 and d["stage"] is None
        job.progress_source = lambda: {"overall": 83.2, "stage": {"key": "verification",
                                                                  "label": "検証", "step": 3}}
        d = job.to_dict()
        assert d["progress"] == 83.2 and d["stage"]["label"] == "検証"
        # the live value can never show 100 before the job finished
        job.progress_source = lambda: {"overall": 100.0, "stage": {"key": "done"}}
        assert job.to_dict()["progress"] == 99.5
        # a finished job shows its own final value and no live stage
        job.progress = 100.0
        job.finish("completed")
        d = job.to_dict()
        assert d["progress"] == 100.0 and d["stage"] is None
