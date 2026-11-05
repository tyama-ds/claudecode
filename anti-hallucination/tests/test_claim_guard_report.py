"""Rendering safeguards using synthetic records; no network or experiment data."""
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest

import claim_guard_report as inspector
import replication as rep
from factcheck.claim_guard import build_block_packet
from tests.test_claim_guard import fixture, review, finalize


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.text = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class ClaimGuardReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.output = self.base / "trial", self.base / "report"
        self.protocol = {
            "id": "inspector-fixture", "report_title": "Synthetic claim guard", "report_date": "2026-11-02",
            "documents": [{"id": identity, "path": f"{cohort}/{identity}", "cohort": cohort, "theme": identity,
                           "arms": ["current", "improved"] if cohort == "regression" else ["A", "B"]}
                          for identity, cohort in (("R1", "regression"), ("R2", "regression"), ("N1", "new"), ("N2", "new"))],
            "arm_mapping": {"N1": {"A": "current", "B": "improved"}, "N2": {"A": "improved", "B": "current"}},
        }
        self.write(self.root / "protocol.json", self.protocol)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rep._json_bytes(value))

    def records(self, doc_id="R1"):
        original, candidate, ledger = fixture()
        guard = review(original, candidate, ledger)
        guard["patch_reviews"][1]["necessity"] = "optional"
        gated = finalize(original, candidate, ledger, guard)
        doc = next(doc for doc in self.protocol["documents"] if doc["id"] == doc_id)
        arm = "improved" if doc["cohort"] == "regression" else next(arm for arm, variant in self.protocol["arm_mapping"][doc_id].items() if variant == "improved")
        folder = self.root / doc["path"]
        arm_folder = folder / "arms" / arm
        values = {"blocks": build_block_packet(original), "ledger": ledger, "candidate": candidate, "guard": guard,
                  "gate": gated, "summary": gated["summary"],
                  "final": {key: gated["final_applied"][key] for key in ("text", "sources")}}
        paths = {key: (folder if key == "blocks" else arm_folder) / name for key, name in inspector.FILES.items()}
        for key, path in paths.items():
            self.write(path, values[key])
        return paths, values

    def render(self):
        result = inspector.render_report(self.root, self.output)
        return result, (self.output / "claim_guard.html").read_text(encoding="utf-8")

    def test_all_planned_documents_stay_pending_without_outcomes_or_zero_success(self):
        result, text = self.render()
        page = Page(text)
        self.assertEqual(result["planned_documents"], 4)
        self.assertEqual(result["documents_with_pending_or_unreadable_records"], 4)
        self.assertEqual(sum(attrs.get("class") == "document" for _, attrs in page.tags), 4)
        self.assertIn("—（未取得）", text)
        self.assertIn("不採用件数は成功件数ではありません", text)
        self.assertNotIn("<strong>0</strong>", text)
        self.assertEqual(list(self.root.rglob("*.json")), [self.root / "protocol.json"])

    def test_escaped_records_exact_quotes_safe_source_links_and_filter_states(self):
        paths, values = self.records()
        attack = "<script>alert('x')</script> & \"quoted\""
        values["ledger"]["claims"][0]["report_scope"] = attack
        values["ledger"]["claims"][0]["reason"] = attack
        values["ledger"]["claims"][2]["status"] = {"malformed": True}
        values["ledger"]["claims"][0]["checks"]["time"]["status"] = "unknown"
        url = "https://example.org/data?a=1&b=%22exact%22#section"
        values["ledger"]["evidence"][0]["url"] = url
        for index, unsafe in enumerate(("javascript:alert(1)", "data:text/html,<script>bad</script>", "https://example.org/\nunsafe")):
            values["guard"]["evidence"].append({"evidence_id": f"BAD{index}", "url": unsafe, "summary": attack})
        for key in ("ledger", "guard"):
            self.write(paths[key], values[key])
        _, text = self.render()
        page = Page(text)
        self.assertNotIn(attack, text)
        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", text)
        self.assertEqual(sum(tag == "script" for tag, _ in page.tags), 1)  # Only the static filter script.
        sources = [attrs["href"] for tag, attrs in page.tags if tag == "a" and attrs.get("class") == "source-url"]
        self.assertEqual(sources, [url])
        self.assertIn(values["ledger"]["claims"][0]["quote"], page.text)
        statuses = [(attrs.get("data-kind"), attrs.get("data-status")) for _, attrs in page.tags if "filter-item" in attrs.get("class", "").split()]
        self.assertIn(("claim", "problem"), statuses)
        self.assertIn(("claim", "invalid"), statuses)
        self.assertIn(("patch", "accepted"), statuses)
        self.assertIn(("patch", "rejected"), statuses)
        self.assertIn("不明（AI判定）", text)
        self.assertIn("necessity is optional", text)
        self.assertIn("The location was East.", text)
        self.assertIn("追加された情報・引用", text)
        self.assertIn("Fixture evidence scope", text)

    def test_new_arm_mapping_rerender_and_unreadable_records_remain_explicit(self):
        self.render()
        paths, values = self.records("N2")
        self.assertIn("arms/A", paths["ledger"].as_posix())
        before = {path: path.read_bytes() for path in paths.values()}
        result, text = self.render()
        self.assertEqual(result["documents_with_pending_or_unreadable_records"], 3)
        self.assertEqual(before, {path: path.read_bytes() for path in paths.values()})
        self.assertIn("new/N2/arms/A/ledger.raw.json", text)
        paths["ledger"].write_text('{"claims":', encoding="utf-8")
        result, text = self.render()
        self.assertEqual(result["documents_with_pending_or_unreadable_records"], 4)
        self.assertIn("読み取りエラー", text)
        self.assertIn("主張台帳は未取得・未完了", text)

    def test_protocol_path_escape_is_rejected_and_output_file_is_supported(self):
        explicit = self.output / "custom.html"
        result = inspector.render_report(self.root, explicit)
        self.assertEqual(result["output"], str(explicit.resolve()))
        self.assertTrue(explicit.exists())
        self.protocol["documents"][0]["path"] = "../outside"
        self.write(self.root / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            inspector.render_report(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
