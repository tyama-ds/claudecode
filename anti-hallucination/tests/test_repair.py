import copy
import hashlib
import unittest

from factcheck.repair import apply_repair


class RepairTests(unittest.TestCase):
    def packet(self, text="alpha beta gamma"):
        return {
            "text": text,
            "title": "Original report",
            "sources": [{
                "citation_id": "old", "title": "Original source",
                "url": "http://example.org/original", "visible_source_entry": {"text": "kept"},
            }],
        }

    def patch(self, before="beta", after="B", patch_id="p1"):
        return {
            "patch_id": patch_id, "before": before, "after": after,
            "finding_ids": ["f1"], "reason": "Fixture repair",
            "source_urls": ["https://example.org/evidence"], "action": "replace",
        }

    def repair(self, *patches):
        return {"patches": list(patches), "added_sources": []}

    def test_multiple_patches_use_original_positions_and_preserve_untouched_text(self):
        original = self.packet("\ufeff前😀\r\nalpha\t beta  \n終gamma\r\n e\u0301  ")
        repair = self.repair(
            self.patch("gamma", "", "delete"),
            self.patch("alpha", "長い replacement", "expand"),
            self.patch("beta", "B", "shorten"),
        )
        result = apply_repair(original, repair)
        self.assertEqual(result["text"], "\ufeff前😀\r\n長い replacement\t B  \n終\r\n e\u0301  ")
        self.assertEqual([p["patch_id"] for p in result["patches"]], ["expand", "shorten", "delete"])
        original_cursor = revised_cursor = 0
        for patch in result["patches"]:
            start, end = patch["start_char"], patch["end_char"]
            after_start, after_end = patch["after_start_char"], patch["after_end_char"]
            self.assertEqual(original["text"][start:end], patch["before"])
            self.assertEqual(result["text"][after_start:after_end], patch["after"])
            self.assertEqual(original["text"][original_cursor:start], result["text"][revised_cursor:after_start])
            original_cursor, revised_cursor = end, after_end
        self.assertEqual(original["text"][original_cursor:], result["text"][revised_cursor:])
        self.assertEqual(result["patches"][0]["start_char"], 5)
        self.assertEqual(result["original_sha256_utf8"], hashlib.sha256(original["text"].encode("utf-8")).hexdigest())
        self.assertEqual(result["revised_sha256_utf8"], hashlib.sha256(result["text"].encode("utf-8")).hexdigest())
        self.assertEqual(result, apply_repair(original, repair))

    def test_inputs_and_nested_metadata_are_not_mutated_or_aliased(self):
        original = self.packet()
        repair = self.repair(self.patch())
        repair["patches"][0]["extra"] = {"tags": ["preserved"]}
        repair["added_sources"].append({"citation_id": "new", "title": "New source", "url": "https://example.org/new"})
        before_original, before_repair = copy.deepcopy(original), copy.deepcopy(repair)
        result = apply_repair(original, repair)
        self.assertEqual(original, before_original)
        self.assertEqual(repair, before_repair)
        self.assertEqual(result["sources"], original["sources"] + repair["added_sources"])
        self.assertEqual(result["patches"][0]["extra"], {"tags": ["preserved"]})
        result["sources"][0]["visible_source_entry"]["text"] = "changed"
        result["sources"][1]["title"] = "changed"
        result["patches"][0]["finding_ids"].append("changed")
        result["patches"][0]["extra"]["tags"].append("changed")
        self.assertEqual(original, before_original)
        self.assertEqual(repair, before_repair)

    def test_adjacent_patches_and_replacement_text_matching_later_patch(self):
        result = apply_repair(self.packet("AB"), self.repair(self.patch("B", "done", "p2"), self.patch("A", "B", "p1")))
        self.assertEqual(result["text"], "Bdone")
        self.assertEqual([(p["start_char"], p["end_char"], p["after_start_char"], p["after_end_char"])
                          for p in result["patches"]], [(0, 1, 0, 1), (1, 2, 1, 5)])

    def test_empty_repair_and_whitespace_only_before_are_valid(self):
        original = self.packet("")
        result = apply_repair(original, self.repair())
        self.assertEqual(result["text"], "")
        self.assertEqual(result["patches"], [])
        self.assertEqual(result["original_sha256_utf8"], result["revised_sha256_utf8"])
        result = apply_repair(self.packet("a\t\nb"), self.repair(self.patch("\t\n", "")))
        self.assertEqual(result["text"], "ab")

    def test_duplicate_patch_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate patch_id"):
            apply_repair(self.packet(), self.repair(self.patch("alpha"), self.patch("beta")))

    def test_empty_missing_and_ambiguous_before_are_rejected(self):
        for text, before, message in [
            ("alpha", "", "must not be empty"),
            ("alpha", "absent", "not found"),
            ("beta beta", "beta", "ambiguous"),
            ("aaa", "aa", "ambiguous"),
            ("é", "e\u0301", "not found"),
        ]:
            with self.subTest(text=text, before=before), self.assertRaisesRegex(ValueError, message):
                apply_repair(self.packet(text), self.repair(self.patch(before)))

    def test_overlapping_ranges_are_rejected(self):
        for first, second in [("abc", "bcd"), ("abcd", "bc"), ("abc", "abc")]:
            with self.subTest(first=first, second=second), self.assertRaisesRegex(ValueError, "overlap"):
                apply_repair(self.packet("abcd"), self.repair(self.patch(first), self.patch(second, patch_id="p2")))

    def test_missing_required_patch_fields_are_rejected(self):
        for field in self.patch():
            patch = self.patch()
            del patch[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_repair(self.packet(), self.repair(patch))

    def test_invalid_patch_field_types_are_rejected(self):
        for field, value in [
            ("patch_id", " "), ("patch_id", 1), ("before", None), ("after", 1),
            ("finding_ids", "f1"), ("finding_ids", [1]), ("finding_ids", [""]),
            ("reason", None), ("action", None), ("source_urls", "https://example.org"),
        ]:
            patch = self.patch()
            patch[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                apply_repair(self.packet(), self.repair(patch))

    def test_citation_id_collisions_are_rejected(self):
        for location in ("existing", "added", "both"):
            original = self.packet()
            repair = self.repair()
            source = {"citation_id": "new", "title": "New", "url": "https://example.org/new"}
            if location == "existing":
                original["sources"].append(copy.deepcopy(original["sources"][0]))
            elif location == "added":
                repair["added_sources"] = [source, copy.deepcopy(source)]
            else:
                source["citation_id"] = "old"
                repair["added_sources"] = [source]
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "duplicate citation_id"):
                apply_repair(original, repair)

    def test_integer_and_string_citation_ids_keep_their_input_types(self):
        original = self.packet()
        original["sources"][0]["citation_id"] = 1
        repair = self.repair()
        repair["added_sources"] = [
            {"citation_id": 20, "title": "Integer", "url": "https://example.org/new"},
            {"citation_id": "21", "title": "String", "url": "https://example.org/another"},
        ]
        result = apply_repair(original, repair)
        self.assertEqual([s["citation_id"] for s in result["sources"]], [1, 20, "21"])
        self.assertIs(type(result["sources"][0]["citation_id"]), int)
        self.assertIs(type(result["sources"][1]["citation_id"]), int)
        self.assertIs(type(result["sources"][2]["citation_id"]), str)
        self.assertIs(type(original["sources"][0]["citation_id"]), int)

    def test_integer_string_citation_id_collisions_are_rejected(self):
        for first, second in [(20, "20"), ("20", 20)]:
            for location in ("existing", "added", "both"):
                original = self.packet()
                repair = self.repair()
                sources = [
                    {"citation_id": first, "title": "First", "url": "https://example.org/first"},
                    {"citation_id": second, "title": "Second", "url": "https://example.org/second"},
                ]
                if location == "existing":
                    original["sources"] = sources
                elif location == "added":
                    repair["added_sources"] = sources
                else:
                    original["sources"] = sources[:1]
                    repair["added_sources"] = sources[1:]
                with self.subTest(first=first, location=location), self.assertRaisesRegex(ValueError, "duplicate citation_id"):
                    apply_repair(original, repair)

    def test_invalid_citation_ids_are_rejected(self):
        for citation_id in [True, False, None, 20.0, [], {}, "", "  "]:
            for location in ("existing", "added"):
                original = self.packet()
                repair = self.repair()
                source = {"citation_id": citation_id, "title": "Source", "url": "https://example.org"}
                if location == "existing":
                    original["sources"] = [source]
                else:
                    repair["added_sources"] = [source]
                with self.subTest(citation_id=citation_id, location=location), self.assertRaisesRegex(ValueError, "citation_id"):
                    apply_repair(original, repair)

    def test_invalid_urls_are_rejected_for_new_sources_and_patch_evidence(self):
        invalid_urls = [
            "http://example.org", "//example.org", "https:///path", "https://",
            "https://user:password@example.org", "https://@example.org",
            "https://example.org/a b", " https://example.org", "https://example.org/\npath",
            "https://example.org/\u3000", "https://example.org/\x00", "https://example.org/\x7f",
            "https://example.org\\path", "https://[broken", "https://example.org:wrong",
            "https://example.org:65536", None,
        ]
        for url in invalid_urls:
            for location in ("source", "patch"):
                repair = self.repair(self.patch())
                if location == "source":
                    repair["added_sources"] = [{"citation_id": "new", "title": "New", "url": url}]
                else:
                    repair["patches"][0]["source_urls"] = [url]
                with self.subTest(url=url, location=location), self.assertRaises(ValueError):
                    apply_repair(self.packet(), repair)

    def test_valid_https_url_is_preserved_without_normalization(self):
        url = "HTTPS://example.org:443/a%20b?q=東京#part"
        repair = self.repair(self.patch())
        repair["patches"][0]["source_urls"] = [url]
        repair["added_sources"] = [{"citation_id": "new", "title": "New", "url": url}]
        result = apply_repair(self.packet(), repair)
        self.assertEqual(result["sources"][-1]["url"], url)
        self.assertEqual(result["patches"][0]["source_urls"], [url])

    def test_invalid_packet_and_repair_shapes_are_rejected(self):
        cases = [
            (None, self.repair()), ({"text": "text"}, self.repair()),
            ({"text": 1, "sources": []}, self.repair()),
            ({"text": "text", "sources": {}}, self.repair()),
            (self.packet(), None), (self.packet(), {}),
            (self.packet(), {"patches": {}, "added_sources": []}),
            (self.packet(), {"patches": [], "added_sources": {}}),
            (self.packet(), {"patches": [None], "added_sources": []}),
            (self.packet(), {"patches": [], "added_sources": [None]}),
            (self.packet(), {"patches": [], "added_sources": [{"citation_id": "new"}]}),
        ]
        for original, repair in cases:
            with self.subTest(original=original, repair=repair), self.assertRaises(ValueError):
                apply_repair(original, repair)

    def test_invalid_utf8_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            apply_repair(self.packet("\ud800"), self.repair())
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            apply_repair(self.packet(), self.repair(self.patch(after="\ud800")))


if __name__ == "__main__":
    unittest.main()
