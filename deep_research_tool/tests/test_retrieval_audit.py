"""Offline regression cases for source identity, retrieval and bounded work."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deep_research_tool.config import MultilingualSearchConfig
from deep_research_tool.research.fast_crawler import FastCrawler, CrawledPage, EvaluationMode
from deep_research_tool.research.local_store import LocalDocumentStore
from deep_research_tool.search.base import PageContent, SearchResult
from deep_research_tool.search.multilingual import MultilingualSearcher, MultilingualSearchResult
from deep_research_tool.utils.document_reader import DocumentReader
from deep_research_tool.utils.helpers import chunk_text
from deep_research_tool.utils.retrieval import select_relevant_spans


def judgment(page=None, summary="Supported source content", score=0.8):
    result = {"processed_content": summary, "relevance_score": score, "key_points": [summary]}
    if page is not None:
        result["page"] = page
    return result


def llm_response(value):
    return SimpleNamespace(content=json.dumps(value))


def source(url="https://a.example/report", text="Source content", **kwargs):
    return CrawledPage(url, "Report", "", text, **kwargs)


def test_reordered_batch_keeps_facts_on_their_source_urls():
    llm = Mock()
    llm.generate.return_value = llm_response([judgment(2, "B revenue 200"), judgment(1, "A revenue 100")])
    crawler = FastCrawler(Mock(), llm)
    pages = [source("https://a.example"), source("https://b.example")]
    results = crawler._batch_evaluate(pages, "Revenue")
    assert [(p.url, p.processed_content) for p in results] == [
        ("https://a.example", "A revenue 100"), ("https://b.example", "B revenue 200")]
    assert llm.generate.call_count == 1


@pytest.mark.parametrize("bad", [
    [judgment(1), judgment(1)], [judgment(2)],
    [judgment(True), judgment(2)], [judgment(0), judgment(2)],
    [judgment(1, score=float("nan")), judgment(2)],
    [judgment(1, score="0.8"), judgment(2)],
    [judgment(1, score=True), judgment(2)],
    [judgment(1, score=1.5), judgment(2)],
    [dict(judgment(1), key_points="wrong type"), judgment(2)],
    [dict(judgment(1), processed_content=None), judgment(2)],
    {"page": 1}, [None, judgment(2)],
])
def test_invalid_batch_never_gets_an_accepted_fallback_score(bad):
    llm = Mock()
    llm.generate.return_value = llm_response(bad)
    crawler = FastCrawler(Mock(), llm)
    results = crawler._batch_evaluate([source(), source("https://b.example")], "Revenue")
    assert len(results) == 2
    assert all(p.error and p.relevance_score == 0 and not p.processed_content for p in results)
    assert llm.generate.call_count == 2


def test_one_schema_repair_is_bounded_and_can_recover():
    llm = Mock()
    llm.generate.side_effect = [SimpleNamespace(content="not json"), llm_response([judgment(1)])]
    result = FastCrawler(Mock(), llm)._batch_evaluate([source()], "Revenue")
    assert result[0].error is None and result[0].relevance_score == 0.8
    assert llm.generate.call_count == 2


@pytest.mark.parametrize("mode", list(EvaluationMode))
def test_all_modes_exclude_invalid_evaluations_even_with_zero_threshold(mode):
    llm = Mock()
    llm.generate.return_value = SimpleNamespace(content="invalid")
    crawler = FastCrawler(Mock(), llm, evaluation_mode=mode)
    crawler._parallel_fetch = Mock(return_value=[source()])
    crawler.content_filter = None
    result = crawler.crawl_and_evaluate(["revenue"], "Revenue", min_relevance_score=0)
    assert result.pages == [] and result.errors and result.pages_evaluated == 1


@pytest.mark.parametrize("mode", list(EvaluationMode))
def test_empty_crawl_makes_no_llm_calls(mode):
    llm = Mock()
    crawler = FastCrawler(Mock(), llm, evaluation_mode=mode)
    result = crawler.crawl_and_evaluate([], "Revenue", research_topic="long topic " * 100)
    assert result.pages == [] and result.pages_evaluated == 0
    llm.generate.assert_not_called()


def test_batched_llm_calls_overlap_within_the_worker_limit():
    barrier = threading.Barrier(2)
    llm = Mock()
    def generate(prompt):
        barrier.wait(timeout=3)
        return llm_response([judgment(1)])
    llm.generate.side_effect = generate
    crawler = FastCrawler(Mock(), llm, batch_size=1, evaluation_workers=4, max_parallel_workers=2)
    results = crawler._batch_evaluate([source(f"https://{i}.example") for i in range(4)], "Revenue")
    assert len(results) == 4 and all(p.error is None for p in results)


class RecordingSearch:
    supports_concurrent_search = True
    def __init__(self, *, serial=False, barrier=None):
        self.requires_serial_access = serial
        self.barrier = barrier
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.fetches = []
    def search(self, query, **kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier:
                self.barrier.wait(timeout=3)
            else:
                time.sleep(0.01)
            return [SearchResult(query, f"https://{query}.example/report", "snippet")]
        finally:
            with self.lock:
                self.active -= 1
    def get_page_content(self, url, **kwargs):
        self.fetches.append((url, kwargs))
        return PageContent(url, "Report", "Raw report content", metadata={"publisher": "Original publisher"})


def test_search_concurrency_respects_cap_and_serial_browser():
    parallel = RecordingSearch(barrier=threading.Barrier(2))
    pages = FastCrawler(parallel, Mock(), max_workers=8, max_parallel_workers=2)._parallel_fetch(["a", "b"], 1)
    assert len(pages) == 2 and parallel.peak == 2
    serial = RecordingSearch(serial=True)
    FastCrawler(serial, Mock(), max_workers=8)._parallel_fetch(["a", "b"], 1)
    assert serial.peak == 1


def test_fetch_preserves_metadata_and_bounded_document_links_with_query_strings():
    search = RecordingSearch()
    def get_page(url, **kwargs):
        search.fetches.append((url, kwargs))
        return PageContent(url, "Report", "Raw report content", metadata={"publisher": "Original"},
                           links=[{"url": "/primary.pdf?download=1#p2", "text": "Primary document"},
                                  {"url": "/other.pdf", "text": "Other"}])
    search.get_page_content = get_page
    crawler = FastCrawler(search, Mock(), max_document_links=1, max_document_pages=1, fetch_timeout=7)
    pages = crawler._parallel_fetch(["a", "b"], 1)
    assert len(pages) == 3
    assert pages[0].metadata["publisher"] == "Original" and pages[0].links
    assert pages[-1].url == "https://a.example/primary.pdf?download=1"
    assert pages[-1].metadata["parent_url"] == pages[0].url
    assert all(kwargs["timeout"] == 7 for _, kwargs in search.fetches)


def test_fast_crawler_uses_multilingual_searcher_when_configured():
    search = RecordingSearch()
    multilingual = Mock()
    multilingual.search_parallel.return_value = ([MultilingualSearchResult(
        "https://a.example", "Report", "Source", "de", "Umsatz", region="de")], None)
    result = FastCrawler(search, Mock(), multilingual_searcher=multilingual)._parallel_fetch(["revenue"], 1)
    multilingual.search_parallel.assert_called_once_with("revenue")
    assert result[0].metadata["source_language"] == "de"


def test_relevance_prompt_includes_matching_tail_span_of_long_document():
    llm = Mock()
    llm.generate.return_value = llm_response([judgment(1)])
    crawler = FastCrawler(Mock(), llm)
    content = "irrelevant introduction " * 5000 + " Revenue_2025_MARKER is 950 billion."
    crawler._evaluate_batch([source(text=content)], "Revenue_2025_MARKER")
    prompt = llm.generate.call_args.args[0]
    assert "Revenue_2025_MARKER is 950 billion" in prompt
    assert len(prompt) < 8000


@pytest.mark.parametrize("paragraphs", [False, True])
def test_chunker_hard_limit_overlap_coverage_and_termination(paragraphs):
    text = "abcde" * 701 + "\n\n" + "TAIL_MARKER"
    chunks = chunk_text(text, chunk_size=300, overlap=40, preserve_paragraphs=paragraphs)
    assert max(map(len, chunks)) <= 300
    reconstructed = chunks[0] + "".join(chunk[40:] for chunk in chunks[1:])
    assert reconstructed == text
    assert len(chunks) < 20


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, 7), (5, -1)])
def test_chunker_rejects_non_progressing_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=size, overlap=overlap)


def test_local_tail_retrieved_as_small_exact_source_chunk_with_offsets():
    text = "unrelated " * 10000 + "RevenueTailMarker=950"
    store = LocalDocumentStore()
    store.add_document("Long report", text, "long.txt", metadata={"publisher": "Original"})
    results = store.search("RevenueTailMarker", keywords=["RevenueTailMarker"])
    assert results and len(results[0].content) <= 3000
    assert "RevenueTailMarker=950" in results[0].content
    metadata = results[0].metadata
    assert text[metadata["start_offset"]:metadata["end_offset"]] == results[0].content
    assert metadata["publisher"] == "Original"


def test_identical_generic_titles_preserve_independent_sources():
    searcher = MultilingualSearcher(MultilingualSearchConfig(), Mock())
    sources = [MultilingualSearchResult(f"https://{host}.example/results", "Financial Results",
                                      f"{host} revenue", "en", "revenue") for host in ("a", "b")]
    assert len(searcher._deduplicate_results(sources)) == 2


def test_url_tracking_variants_merge_but_preserve_discovery_provenance():
    searcher = MultilingualSearcher(MultilingualSearchConfig(), Mock())
    results = [MultilingualSearchResult("https://a.example/report?utm_source=mail", "Report", "s", "en", "revenue"),
               MultilingualSearchResult("https://a.example/report#page2", "Report", "s", "ja", "売上")]
    merged = searcher._deduplicate_results(results)
    assert len(merged) == 1 and len(merged[0].metadata["discovered_in"]) == 2


def test_search_ranking_uses_query_and_original_serp_rank():
    searcher = MultilingualSearcher(MultilingualSearchConfig(), Mock())
    unrelated = MultilingualSearchResult("https://b.example", "Weather", "Forecast", "en", "revenue", search_rank=1)
    relevant = MultilingualSearchResult("https://a.example", "Revenue", "Revenue data", "en", "revenue", search_rank=2)
    assert searcher._score_results([unrelated, relevant])[0] is relevant


def test_localization_runs_in_parallel_and_successes_are_cached():
    barrier = threading.Barrier(2)
    llm = Mock()
    def generate(prompt, **kwargs):
        barrier.wait(timeout=3)
        return SimpleNamespace(content="localized revenue")
    llm.generate.side_effect = generate
    searcher = MultilingualSearcher(MultilingualSearchConfig(search_languages=["ja", "en"]), Mock(), llm,
                                    max_parallel_workers=2)
    first = searcher.translate_queries("revenue")
    second = searcher.translate_queries("revenue")
    assert len(first) == len(second) == 2
    assert llm.generate.call_count == 2


def test_docx_preserves_heading_and_table_order(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    for year, value in [(2024, 100), (2025, 200)]:
        doc.add_paragraph(f"{year} Revenue")
        doc.add_table(rows=1, cols=1).cell(0, 0).text = str(value)
    path = tmp_path / "results.docx"
    doc.save(path)
    result = DocumentReader(extract_images=False).read_document(path)
    assert result.content == "2024 Revenue\n\n100\n\n2025 Revenue\n\n200"
    assert [block["kind"] for block in result.metadata["block_spans"]] == ["paragraph", "table"] * 2


def test_pdf_missing_text_is_explicit_and_page_offsets_are_preserved(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / "partial.pdf"
    doc = canvas.Canvas(str(path))
    doc.drawString(72, 72, "2025 Revenue 200")
    doc.showPage()
    doc.showPage()
    doc.save()
    result = DocumentReader(extract_images=False).read_document(path)
    assert result.error is None
    assert result.metadata["needs_ocr"] is True
    assert result.metadata["missing_text_pages"] == [2]
    span = result.metadata["page_spans"][0]
    assert "2025 Revenue 200" in result.content[span["start_offset"]:span["end_offset"]]


def test_completely_empty_pdf_is_not_reported_as_success(tmp_path):
    from pypdf import PdfWriter
    doc = PdfWriter()
    doc.add_blank_page(width=612, height=792)
    path = tmp_path / "scan.pdf"
    doc.write(path)
    result = DocumentReader(extract_images=False).read_document(path)
    assert result.error and "OCR" in result.error


def test_pymupdf_ocr_recovers_text_and_does_not_decode_images(tmp_path, monkeypatch):
    import sys
    page = Mock()
    page.get_text.side_effect = ["", "OCR revenue 200"]
    page.get_images.return_value = [(17, 0, 100, 200)]
    doc = Mock()
    doc.metadata = {"title": "Scanned report"}
    class FakeDocument:
        metadata = doc.metadata
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return page
        def close(self):
            doc.close()
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda _: FakeDocument()))
    result = DocumentReader(enable_ocr=True)._read_pdf_pymupdf(tmp_path / "scan.pdf")
    assert result.content == "OCR revenue 200" and result.error is None
    assert result.metadata["needs_ocr"] is False
    assert result.images[0]["width"] == 100
    page.get_textpage_ocr.assert_called_once_with(language="jpn+eng")
    doc.close.assert_called_once()


def test_pymupdf_missing_ocr_backend_is_an_explicit_error(tmp_path, monkeypatch):
    import sys
    page = Mock()
    page.get_text.return_value = ""
    page.get_textpage_ocr.side_effect = RuntimeError("Tesseract data missing")
    class FakeDocument:
        metadata = {}
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return page
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda _: FakeDocument()))
    result = DocumentReader(extract_images=False, enable_ocr=True)._read_pdf_pymupdf(tmp_path / "scan.pdf")
    assert result.error and result.metadata["needs_ocr"]
    assert "Tesseract" in result.metadata["ocr_errors"][0]


class AuditCancelled(RuntimeError):
    pass


def cancellation_check(event):
    def check():
        if event.is_set():
            raise AuditCancelled("cancelled")
    return check


def test_cancelled_fast_crawler_never_starts_search_or_fetch():
    event = threading.Event()
    event.set()
    search, llm = Mock(), Mock()
    crawler = FastCrawler(search, llm, cancel_check=cancellation_check(event))
    with pytest.raises(AuditCancelled):
        crawler._parallel_fetch(["query"], 2)
    search.search.assert_not_called()
    search.get_page_content.assert_not_called()
    llm.generate.assert_not_called()


def test_cancellation_between_queries_prevents_next_search_and_all_fetches():
    event = threading.Event()
    search = Mock()
    def search_once(*args, **kwargs):
        event.set()
        return [SearchResult("Result", "https://a.example", "Source")]
    search.search.side_effect = search_once
    crawler = FastCrawler(search, Mock(), max_parallel_workers=1, cancel_check=cancellation_check(event))
    with pytest.raises(AuditCancelled):
        crawler._parallel_fetch(["first", "second"], 1)
    assert search.search.call_count == 1
    search.get_page_content.assert_not_called()


def test_cancellation_between_fetches_prevents_next_http_request():
    event = threading.Event()
    search = Mock()
    search.search.return_value = [SearchResult("Result", f"https://{i}.example", "Source") for i in range(2)]
    def fetch_once(url, **kwargs):
        event.set()
        return PageContent(url, "Result", "Original source")
    search.get_page_content.side_effect = fetch_once
    crawler = FastCrawler(search, Mock(), max_parallel_workers=1, cancel_check=cancellation_check(event))
    with pytest.raises(AuditCancelled):
        crawler._parallel_fetch(["query"], 2)
    assert search.get_page_content.call_count == 1


@pytest.mark.parametrize("method", ["_batch_evaluate", "_parallel_evaluate", "_sequential_evaluate"])
def test_cancellation_between_evaluations_prevents_next_llm_call(method):
    event = threading.Event()
    llm = Mock()
    def generate_once(prompt):
        event.set()
        return llm_response([judgment(1)] if method == "_batch_evaluate" else judgment())
    llm.generate.side_effect = generate_once
    crawler = FastCrawler(Mock(), llm, batch_size=1, max_parallel_workers=1,
                          cancel_check=cancellation_check(event))
    with pytest.raises(AuditCancelled):
        getattr(crawler, method)([source(), source("https://b.example")], "Revenue")
    assert llm.generate.call_count == 1


def test_cancellation_before_schema_retry_is_not_downgraded_to_source_error():
    event = threading.Event()
    llm = Mock()
    def malformed(prompt):
        event.set()
        return SimpleNamespace(content="not json")
    llm.generate.side_effect = malformed
    crawler = FastCrawler(Mock(), llm, cancel_check=cancellation_check(event))
    with pytest.raises(AuditCancelled):
        crawler._batch_evaluate([source()], "Revenue")
    assert llm.generate.call_count == 1


def test_repeated_context_uses_a_copy_of_run_cache_and_changed_topic_recomputes():
    llm = Mock()
    llm.generate.return_value = llm_response({"keywords": ["revenue"], "focus_areas": "revenue evidence"})
    crawler = FastCrawler(Mock(), llm)
    topic = "revenue and annual report evidence " * 8
    first = crawler._extract_research_context(topic)
    first["keywords"].append("caller mutation")
    second = crawler._extract_research_context(topic)
    assert second["keywords"] == ["revenue"]
    assert llm.generate.call_count == 1
    crawler._extract_research_context(topic + " different scope")
    assert llm.generate.call_count == 2
    crawler.context_cache.clear()
    crawler._extract_research_context(topic)
    assert llm.generate.call_count == 3


def test_failed_context_is_not_cached_and_precancelled_context_makes_no_call():
    llm = Mock()
    llm.generate.side_effect = [SimpleNamespace(content="invalid"),
                               llm_response({"keywords": ["revenue"], "focus_areas": "Revenue"})]
    event = threading.Event()
    crawler = FastCrawler(Mock(), llm, cancel_check=cancellation_check(event))
    topic = "revenue scope " * 20
    assert crawler._extract_research_context(topic)["keywords"] == []
    assert crawler._extract_research_context(topic)["keywords"] == ["revenue"]
    assert llm.generate.call_count == 2
    event.set()
    with pytest.raises(AuditCancelled):
        crawler._extract_research_context(topic)
    assert llm.generate.call_count == 2


@pytest.mark.parametrize("mode", list(EvaluationMode))
def test_streaming_evaluates_before_final_slow_fetch_and_keeps_search_order(mode):
    evaluation_started = threading.Event()
    slow_finished = threading.Event()
    search = Mock()
    search.search.return_value = [SearchResult(f"Report {i}", f"https://{i}.example", "Source") for i in range(3)]
    def fetch(url, **kwargs):
        if url == "https://2.example":
            if not evaluation_started.wait(timeout=3):
                raise RuntimeError("Evaluation waited for all fetches")
            slow_finished.set()
        return PageContent(url, url, "Original source content for " + url)
    search.get_page_content.side_effect = fetch
    llm = Mock()
    def generate(prompt):
        # The first evaluation must overlap the outstanding last HTTP fetch.
        if not evaluation_started.is_set():
            assert not slow_finished.is_set()
            evaluation_started.set()
        if mode == EvaluationMode.BATCH:
            count = prompt.count("=== PAGE ")
            # Deliberately reverse response IDs; streaming must retain mapping.
            return llm_response([judgment(i, f"page {i}") for i in range(count, 0, -1)])
        return llm_response(judgment())
    llm.generate.side_effect = generate
    crawler = FastCrawler(search, llm, evaluation_mode=mode, batch_size=2,
                          max_workers=3, max_parallel_workers=3)
    crawler.content_filter = None
    result = crawler.crawl_and_evaluate(["revenue"], "Revenue", max_pages_per_query=3)
    assert evaluation_started.is_set() and slow_finished.is_set()
    assert not result.errors and result.pages_fetched == result.pages_evaluated == 3
    assert [page.url for page in result.pages] == [f"https://{i}.example" for i in range(3)]
    assert result.total_wall_time > 0


def test_optional_page_stream_preserves_direct_list_interface_and_document_caps():
    search = Mock()
    search.search.return_value = [SearchResult("Landing", "https://a.example", "Source")]
    def fetch(url, **kwargs):
        return PageContent(url, "Report", "Original source content", links=[
            {"url": "/one.pdf", "text": "One"}, {"url": "/two.pdf", "text": "Two"}])
    search.get_page_content.side_effect = fetch
    seen = []
    crawler = FastCrawler(search, Mock(), max_document_pages=1, max_document_links=1)
    pages = crawler._parallel_fetch(["query"], 1, on_page=seen.append)
    assert [page.url for page in seen] == [page.url for page in pages] == [
        "https://a.example", "https://a.example/one.pdf"]


def test_streaming_retains_fetch_and_schema_errors_without_accepting_them():
    search = Mock()
    search.search.return_value = [SearchResult(str(i), f"https://{i}.example", "Source") for i in range(3)]
    def fetch(url, **kwargs):
        if url == "https://1.example":
            raise RuntimeError("Synthetic HTTP failure")
        return PageContent(url, "Report", "Original source content")
    search.get_page_content.side_effect = fetch
    llm = Mock()
    llm.generate.return_value = SimpleNamespace(content="not JSON")
    crawler = FastCrawler(search, llm, batch_size=1)
    crawler.content_filter = None
    result = crawler.crawl_and_evaluate(["revenue"], "Revenue", max_pages_per_query=3, min_relevance_score=0)
    assert result.pages == []
    assert result.pages_fetched == 3 and result.pages_evaluated == 2 and result.pages_filtered == 1
    assert len(result.errors) == 3
    assert any("Synthetic HTTP failure" in error for error in result.errors)


def test_streaming_does_not_evaluate_error_text_returned_as_page_content():
    search, llm = Mock(), Mock()
    search.search.return_value = [SearchResult("Report", "https://a.example", "Source")]
    search.get_page_content.return_value = PageContent(
        "https://a.example", "Report", "Long HTTP error response " * 30,
        metadata={"error": "upstream request failed"})
    crawler = FastCrawler(search, llm, batch_size=1)
    crawler.content_filter = None
    result = crawler.crawl_and_evaluate(["revenue"], "Revenue")
    assert result.pages == [] and result.pages_filtered == 1
    assert result.errors and "upstream request failed" in result.errors[0]
    llm.generate.assert_not_called()


def test_streaming_cancellation_blocks_later_evaluation_and_retry():
    event = threading.Event()
    search = Mock()
    search.search.return_value = [SearchResult(str(i), f"https://{i}.example", "Source") for i in range(6)]
    search.get_page_content.side_effect = lambda url, **kwargs: PageContent(url, "Report", "Original source content")
    llm = Mock()
    def first_evaluation(prompt):
        event.set()
        return SimpleNamespace(content="not JSON")
    llm.generate.side_effect = first_evaluation
    crawler = FastCrawler(search, llm, batch_size=1, max_parallel_workers=1,
                          cancel_check=cancellation_check(event))
    crawler.content_filter = None
    with pytest.raises(AuditCancelled):
        crawler.crawl_and_evaluate(["revenue"], "Revenue", max_pages_per_query=6)
    assert llm.generate.call_count == 1


def test_streaming_bounds_outstanding_evaluation_jobs():
    lock = threading.Lock()
    active = peak = 0
    barrier = threading.Barrier(2)
    search = Mock()
    search.search.return_value = [SearchResult(str(i), f"https://{i}.example", "Source") for i in range(6)]
    search.get_page_content.side_effect = lambda url, **kwargs: PageContent(url, "Report", "Original source content")
    llm = Mock()
    def generate(prompt):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=3)
            return llm_response([judgment(1)])
        finally:
            with lock:
                active -= 1
    llm.generate.side_effect = generate
    crawler = FastCrawler(search, llm, batch_size=1, evaluation_workers=2, max_parallel_workers=3)
    crawler.content_filter = None
    result = crawler.crawl_and_evaluate(["revenue"], "Revenue", max_pages_per_query=6)
    assert not result.errors and len(result.pages) == 6
    assert peak == 2
