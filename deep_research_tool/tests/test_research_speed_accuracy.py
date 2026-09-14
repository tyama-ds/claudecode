"""Offline regression coverage for source reuse and bounded research workers."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deep_research_tool.research.cache import CachedSearchClient, canonical_url
from deep_research_tool.research.content_extractor import ContentExtractor, ExtractedContent
from deep_research_tool.research.researcher import Researcher, ResearchSession
from deep_research_tool.research.query_generator import TableOfContentsItem
from deep_research_tool.evidence.locker import EvidenceLocker
from deep_research_tool.search.base import PageContent, SearchResult


def test_page_cache_single_flight_and_copy_isolation():
    client = Mock()
    def fetch(url, **kwargs):
        time.sleep(0.03)
        return PageContent(url, 'Title', 'Raw source', metadata={'value': 1})
    client.get_page_content.side_effect = fetch
    cached = CachedSearchClient(client)
    with ThreadPoolExecutor(max_workers=8) as ex:
        pages = list(ex.map(lambda _: cached.get_page_content('https://EXAMPLE.com/a?utm_source=x#top'), range(8)))
    assert client.get_page_content.call_count == 1
    pages[0].metadata['value'] = 99
    assert cached.get_page_content('https://example.com/a').metadata['value'] == 1
    cached.get_page_content('https://example.com/a', timeout=5)
    assert client.get_page_content.call_count == 1
    cached.clear_cache()
    cached.get_page_content('https://example.com/a')
    assert client.get_page_content.call_count == 2
    assert canonical_url('https://a.example/?year=2025') != canonical_url('https://a.example/?year=2026')


def test_failed_fetch_is_retried():
    client = Mock()
    client.get_page_content.side_effect = [RuntimeError('timeout'), PageContent('https://a.example', 'a', 'text')]
    cached = CachedSearchClient(client)
    with pytest.raises(RuntimeError):
        cached.get_page_content('https://a.example')
    assert cached.get_page_content('https://a.example').text_content == 'text'
    assert client.get_page_content.call_count == 2


def test_error_page_with_diagnostic_text_is_not_cached():
    client = Mock()
    client.get_page_content.side_effect = [
        PageContent('https://a.example', 'a', 'Failed to extract content: blocked', metadata={'error': 'waf_blocked'}),
        PageContent('https://a.example', 'a', 'Valid source text'),
    ]
    cached = CachedSearchClient(client)
    assert cached.get_page_content('https://a.example').metadata['error'] == 'waf_blocked'
    assert cached.get_page_content('https://a.example').text_content == 'Valid source text'
    assert client.get_page_content.call_count == 2


def test_single_flight_waiter_respects_own_timeout_without_cancelling_owner():
    started, release = threading.Event(), threading.Event()
    client = Mock()
    def fetch(url, **kwargs):
        started.set()
        assert release.wait(timeout=2)
        return PageContent(url, 'a', 'Valid source text')
    client.get_page_content.side_effect = fetch
    cached = CachedSearchClient(client)
    with ThreadPoolExecutor(max_workers=1) as ex:
        owner = ex.submit(cached.get_page_content, 'https://a.example', timeout=2)
        assert started.wait(timeout=1)
        try:
            with pytest.raises(TimeoutError):
                cached.get_page_content('https://a.example', timeout=0.01)
        finally:
            release.set()
        assert owner.result().text_content == 'Valid source text'
    assert cached.get_page_content('https://a.example').text_content == 'Valid source text'
    assert client.get_page_content.call_count == 1


def test_extraction_cache_and_tail_spans():
    llm = SimpleNamespace(model='test')
    extractor = ContentExtractor(llm, max_parallel_workers=2)
    extracted_chunks = []
    def extract(chunk, *args, **kwargs):
        extracted_chunks.append(chunk)
        return {'processed_content': chunk[:100], 'relevance_score': 0.8, 'key_points': [], 'quotes': []}
    extractor._extract_single_chunk = extract
    source = ('unrelated filler ' * 5500) + (' UNIQUE_TAIL_FINDING ' * 100)
    kwargs = dict(raw_content=source, source_url='https://a.example', source_title='Source',
                  section_context='Finding', research_query='UNIQUE_TAIL_FINDING')
    first = extractor.extract_relevant_content(**kwargs)
    assert any('UNIQUE_TAIL_FINDING' in chunk for chunk in extracted_chunks)
    assert len(extracted_chunks) <= 8
    for span in first.metadata['source_spans']:
        assert source[span['start_offset']:span['end_offset']] in extracted_chunks
    assert first.raw_content == source
    first.importance_score = 0.9
    calls = len(extracted_chunks)
    assert extractor.extract_relevant_content(**kwargs).importance_score == 0.0
    assert len(extracted_chunks) == calls
    extractor.extract_relevant_content(**{**kwargs, 'research_query': 'different question'})
    assert len(extracted_chunks) > calls
    calls = len(extracted_chunks)
    llm.model = 'new-model'
    extractor.extract_relevant_content(**kwargs)
    assert len(extracted_chunks) > calls


@pytest.mark.parametrize('budget,prefix', [(1000, 4000), (12000, 17500)])
def test_selection_budget_does_not_clip_away_ranked_match(budget, prefix):
    extractor = ContentExtractor(Mock())
    extractor.max_selected_chars = budget
    raw = 'x' * prefix + ' TAIL_MARKER ' + 'x' * 30
    spans = extractor._select_chunks(raw, '', 'TAIL_MARKER')
    assert any('TAIL_MARKER' in span['text'] for span in spans)
    assert sum(len(span['text']) for span in spans) <= budget
    assert [span['start_offset'] for span in spans] == sorted(span['start_offset'] for span in spans)
    for span in spans:
        assert raw[span['start_offset']:span['end_offset']] == span['text']


def test_enhanced_points_parallel_order_and_combined_summary():
    llm = Mock()
    llm.generate.return_value = SimpleNamespace(content='Body [SOURCE 1]\n===SECTION_META===\n{"summary":"Source-grounded summary"}')
    extractor = ContentExtractor(llm, language='en', max_parallel_workers=2)
    extractor._generate_section_outline = Mock(return_value=[{'title': str(i)} for i in range(4)])
    lock = threading.Lock()
    active = peak = 0
    def point(title, item, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02 * (4 - int(item['title'])))
        with lock:
            active -= 1
        return item['title'] * 120
    extractor._generate_point_content = point
    result = extractor.synthesize_section_content_enhanced('Title', 'Description',
        [ExtractedContent('https://a.example', 'Source', 'Raw', 'Processed')])
    assert peak == 2
    assert llm.generate.call_count == 1  # integration also supplies summary
    assert result['summary'] == 'Source-grounded summary'
    assert result['analysis_points'] == ['0', '1', '2', '3']


def test_standard_deduplicates_before_fetch_and_preserves_provenance(tmp_path):
    search = Mock()
    search.search.return_value = [SearchResult(str(i), f'https://example.com/{i}', 'snippet') for i in range(3)]
    search.get_page_content.side_effect = lambda url: PageContent(url, url, f'{url} Raw source ' * 30,
                                                                 metadata={'date_published': '2024-05-01'})
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none',
                   min_iterations=3, max_iterations=3, parallel_max_workers=2)
    r.session = ResearchSession(query='Topic')
    r.evidence_locker = EvidenceLocker(output_dir=tmp_path / 'evidence')
    r.content_extractor.extract_relevant_content = Mock(side_effect=lambda **kw: ExtractedContent(
        kw['source_url'], kw['source_title'], kw['raw_content'], 'Processed source', relevance_score=0.9))
    r.query_generator.identify_gaps = Mock(return_value=['gap'])
    r.query_generator.generate_follow_up_queries = Mock(return_value=['repeat'])
    r._generate_and_save_section_content = Mock()
    r._process_section_with_immediate_generation(TableOfContentsItem('1', 'Topic', ''), ['a', 'b', 'c'], 0, 1)
    assert search.get_page_content.call_count == 3
    assert r.content_extractor.extract_relevant_content.call_count == 6  # three URLs, two distinct query contexts
    assert len(r.session.iterations) == 2  # three distinct sources did not trigger the >=6 early exit
    parts = r._generate_and_save_section_content.call_args.args[1]
    assert [p.source_url for p in parts] == [f'https://example.com/{i}' for i in range(3)]
    evidence = r.evidence_locker.get_all_evidence()
    assert len(evidence) == 3
    assert evidence[0].metadata['search_queries'] == ['a', 'b', 'c', 'repeat']
    assert evidence[0].extracted_text.startswith('https://example.com/0 Raw source')


def test_standard_duplicate_content_counts_once(tmp_path):
    search = Mock()
    search.search.return_value = [SearchResult(str(i), f'https://mirror{i}.example/a', '') for i in range(3)]
    search.get_page_content.side_effect = lambda url: PageContent(url, 'Mirror', 'Same syndicated content ' * 30)
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none', max_iterations=1)
    r.session = ResearchSession(query='Topic')
    r.evidence_locker = EvidenceLocker(output_dir=tmp_path / 'evidence')
    r.content_extractor.extract_relevant_content = Mock(side_effect=lambda **kw: ExtractedContent(
        kw['source_url'], kw['source_title'], kw['raw_content'], 'Same finding', relevance_score=0.9))
    r._generate_and_save_section_content = Mock()
    r._process_section_with_immediate_generation(TableOfContentsItem('1', 'Topic', ''), ['a'], 0, 1)
    assert r.session.iterations[0].content_extracted == 1
    assert len(r._generate_and_save_section_content.call_args.args[1]) == 1


def test_new_question_reextracts_cached_document_tail_without_new_source_count(tmp_path):
    search = Mock()
    url = 'https://example.com/long-report'
    raw = ('OVERVIEW basic findings ' * 100) + 'x' * 24000 + ' TAIL_DISCOVERY: 42 '
    search.search.return_value = [SearchResult('Long report', url, '')]
    search.get_page_content.return_value = PageContent(url, 'Long report', raw)
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none',
                   min_iterations=3, max_iterations=3, max_content_length=1000)
    r.session = ResearchSession(query='General research')
    r.evidence_locker = EvidenceLocker(output_dir=tmp_path / 'evidence')
    def extract(chunk, *args, **kwargs):
        return {'processed_content': ('TAIL_DISCOVERY: 42' if 'TAIL_DISCOVERY' in chunk else 'Overview findings'),
                'relevance_score': 0.9, 'key_points': [], 'quotes': []}
    r.content_extractor._extract_single_chunk = Mock(side_effect=extract)
    r.query_generator.identify_gaps = Mock(return_value=['tail details'])
    r.query_generator.generate_follow_up_queries = Mock(return_value=['TAIL_DISCOVERY'])
    r._generate_and_save_section_content = Mock()
    r._process_section_with_immediate_generation(TableOfContentsItem('1', 'General', ''), ['OVERVIEW'], 0, 1)
    assert search.get_page_content.call_count == 1
    assert r.content_extractor._extract_single_chunk.call_count == 2
    parts = r._generate_and_save_section_content.call_args.args[1]
    assert len(parts) == 1
    assert 'Overview findings' in parts[0].processed_content
    assert 'TAIL_DISCOVERY: 42' in parts[0].processed_content
    assert len(parts[0].metadata['source_spans']) == 2
    assert r.session.iterations[1].content_extracted == 1  # new support, same independent source
    assert len(r.evidence_locker.get_all_evidence()) == 1


def test_parent_extraction_failure_preserves_usable_linked_document(tmp_path):
    search = Mock()
    parent, document = 'https://example.com/page', 'https://example.com/report.pdf'
    pages = {
        parent: PageContent(parent, 'Page', 'Parent raw source', links=[{'url': document, 'text': 'PDF'}]),
        document: PageContent(document, 'PDF', 'PDF contains usable original findings'),
    }
    search.get_page_content.side_effect = lambda url: pages[url]
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none')
    def extract(**kw):
        if kw['source_url'] == parent:
            raise RuntimeError('parent model response failed')
        return ExtractedContent(document, 'PDF', kw['raw_content'], 'PDF findings', relevance_score=0.9)
    r.content_extractor.extract_relevant_content = Mock(side_effect=extract)
    collected = r._fetch_section_result(SearchResult('Page', parent, ''), TableOfContentsItem('1', 'Topic', ''), ['query'])
    assert len(collected) == 1
    assert collected[0][0].source_url == document


def test_cancelled_research_starts_no_search(tmp_path):
    class StopNow(RuntimeError):
        pass
    search = Mock()
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none')
    r.session = ResearchSession(query='Topic')
    r.cancel_check = Mock(side_effect=StopNow('cancelled'))
    with pytest.raises(StopNow):
        r._process_section_with_immediate_generation(TableOfContentsItem('1', 'Topic', ''), ['query'], 0, 1)
    search.search.assert_not_called()


def test_cancelled_synthesis_starts_no_additional_points():
    class StopNow(RuntimeError):
        pass
    stopped = threading.Event()
    llm = Mock()
    extractor = ContentExtractor(llm, max_parallel_workers=1)
    extractor._generate_section_outline = Mock(return_value=[{'title': str(i)} for i in range(4)])
    def check():
        if stopped.is_set():
            raise StopNow('cancelled')
    extractor.cancel_check = check
    def point(*args, **kwargs):
        stopped.set()
        return 'Completed first point'
    extractor._generate_point_content = Mock(side_effect=point)
    with pytest.raises(StopNow):
        extractor.synthesize_section_content_enhanced('Topic', '', [ExtractedContent('url', 'title', 'raw')])
    assert extractor._generate_point_content.call_count == 1
    llm.generate.assert_not_called()


def test_standard_loop_retries_partial_extraction_for_same_question(tmp_path):
    search = Mock()
    search.search.return_value = [SearchResult('Source', 'https://example.com/source', '')]
    search.get_page_content.return_value = PageContent('https://example.com/source', 'Source', 'Full original text')
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none', min_iterations=3, max_iterations=3)
    r.session = ResearchSession(query='Topic')
    r.evidence_locker = EvidenceLocker(output_dir=tmp_path / 'evidence')
    r.content_extractor.extract_relevant_content = Mock(side_effect=[
        ExtractedContent('https://example.com/source', 'Source', 'Full original text', 'First finding',
                         relevance_score=0.9, metadata={'extraction_complete': False}),
        ExtractedContent('https://example.com/source', 'Source', 'Full original text', 'First and second findings',
                         relevance_score=0.9, metadata={'extraction_complete': True}),
    ])
    r.query_generator.identify_gaps = Mock(return_value=['missing fact'])
    r.query_generator.generate_follow_up_queries = Mock(return_value=['same question'])
    r._generate_and_save_section_content = Mock()
    r._process_section_with_immediate_generation(TableOfContentsItem('1', 'Topic', ''), ['same question'], 0, 1)
    assert r.content_extractor.extract_relevant_content.call_count == 2
    assert search.get_page_content.call_count == 1
    assert len(r.evidence_locker.get_all_evidence()) == 1
    assert 'second findings' in r.evidence_locker.get_all_evidence()[0].content_excerpt


def test_many_sources_still_respect_minimum_iterations(tmp_path):
    search = Mock()
    search.search.side_effect = lambda query: [SearchResult(str(i), f'https://example.com/{query}/{i}', '') for i in range(3)]
    search.get_page_content.side_effect = lambda url: PageContent(url, url, f'Unique source {url}')
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none', min_iterations=3, max_iterations=5)
    r.session = ResearchSession(query='Topic')
    r.evidence_locker = EvidenceLocker(output_dir=tmp_path / 'evidence')
    r.content_extractor.extract_relevant_content = Mock(side_effect=lambda **kw: ExtractedContent(
        kw['source_url'], kw['source_title'], kw['raw_content'], 'Finding ' + kw['source_url'], relevance_score=0.9))
    r.query_generator.identify_gaps = Mock(return_value=['gap'])
    r.query_generator.generate_follow_up_queries = Mock(side_effect=[['d'], ['e']])
    r._generate_and_save_section_content = Mock()
    r._process_section_with_immediate_generation(TableOfContentsItem('1', 'Topic', ''), ['a', 'b', 'c'], 0, 1)
    assert r.session.iterations[0].content_extracted == 9
    assert len(r.session.iterations) == 3
    assert r.session.iterations[-1].stop_reason == 'minimum_iterations_and_sources_satisfied'


def test_shared_detail_timings_count_operations_not_cache_hits(tmp_path):
    search = Mock()
    search.search.return_value = []
    search.get_page_content.side_effect = lambda url: PageContent(url, 'Source', 'private document text')
    r = Researcher(Mock(), search, output_dir=tmp_path, filter_mode='none', use_enhanced_synthesis=False)
    r.session = ResearchSession(query='private research query')
    r.search.search('private query')
    r.search.get_page_content('https://example.com/source')
    r.search.get_page_content('https://example.com/source')
    r.content_extractor._extract_single_chunk = Mock(return_value={
        'processed_content': 'Extracted evidence', 'relevance_score': 0.9})
    args = dict(raw_content='private document text', source_url='https://example.com/source',
                source_title='Private title', section_context='Topic', research_query='private query')
    part = r.content_extractor.extract_relevant_content(**args)
    r.content_extractor.extract_relevant_content(**args)
    r.content_extractor.synthesize_section_content = Mock(return_value={'content': 'Report'})
    assert r._synthesize_parts(TableOfContentsItem('1', 'Topic', ''), [part]) == {'content': 'Report'}
    search.search.side_effect = RuntimeError('search failed')
    with pytest.raises(RuntimeError):
        r.search.search('failed query')
    snapshot = r.performance_snapshot()
    assert {name: data['calls'] for name, data in snapshot['stages'].items()} == {
        'search': 2, 'fetch': 1, 'extract': 1, 'synthesis': 1}
    assert snapshot['stages']['search']['failures'] == 1
    assert all(stage['seconds'] >= 0 for stage in snapshot['stages'].values())
    assert snapshot['cache']['fetch'] == {'hits': 1, 'misses': 1}
    assert snapshot['cache']['extract'] == {'hits': 1, 'misses': 1}
    assert 'private' not in str(snapshot)
    assert r.search._timings is r.timings is r.content_extractor.timings
    assert '_timings' not in vars(search)  # wrapper instrumentation is never forwarded
    r._reset_timings()
    assert r.performance_snapshot()['stages'] == {}


def test_partial_chunk_failure_keeps_failed_spans_and_retries():
    extractor = ContentExtractor(Mock(), max_parallel_workers=1)
    good = {'processed_content': 'Grounded extracted content', 'relevance_score': 0.9}
    extractor._extract_single_chunk = Mock(side_effect=[None, good, good, good])
    args = dict(raw_content='x' * 8000, source_url='https://example.com/source',
                source_title='Source', section_context='Topic', research_query='query')
    partial = extractor.extract_relevant_content(**args)
    assert partial.metadata['extraction_complete'] is False
    assert len(partial.metadata['source_spans']) == 1
    assert partial.metadata['source_spans'][0]['start_offset'] == 5500
    assert partial.metadata['failed_spans'] == [{'start_offset': 0, 'end_offset': 6000, 'chunk_index': 0}]
    complete = extractor.extract_relevant_content(**args)
    assert extractor._extract_single_chunk.call_count == 4
    assert complete.metadata['extraction_complete'] is True
    assert len(complete.metadata['source_spans']) == 2
    assert complete.metadata['failed_spans'] == []
    extractor.extract_relevant_content(**args)
    assert extractor._extract_single_chunk.call_count == 4  # only complete source extraction cached
