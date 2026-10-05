"""
Local document store for local / hybrid research source modes.

Holds user-provided documents (already read into text by DocumentReader),
splits them into overlapping chunks, and serves per-section retrieval with
keyword scoring — a lightweight local counterpart to web search.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.helpers import chunk_text
from .site_crawler import score_relevance_simple


@dataclass
class LocalChunk:
    """A scored chunk of a local document."""
    doc_title: str
    doc_path: str
    content: str
    chunk_index: int
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        """Pseudo-URL used for evidence/citation bookkeeping."""
        base = self.doc_path or self.doc_title
        return f"local://{base}#chunk{self.chunk_index}"


class LocalDocumentStore:
    """
    Chunked keyword-searchable store of local documents.

    Usage:
        store = LocalDocumentStore()
        store.add_document(title="社内レポート", content=text, path="report.pdf")
        chunks = store.search("市場規模", top_k=5, keywords=["市場", "規模"])
    """

    def __init__(self, chunk_size: int = 3000, overlap: int = 200):
        """
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: List[LocalChunk] = []
        self._doc_titles: List[str] = []

    def add_document(self, title: str, content: str, path: str = "",
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a document, splitting it into chunks.

        Args:
            title: Document title
            content: Full text content
            path: Original file path (optional)

        Returns:
            Number of chunks added
        """
        if not content or not content.strip():
            return 0

        pieces = chunk_text(content, chunk_size=self.chunk_size, overlap=self.overlap)
        previous_end = 0
        for i, piece in enumerate(pieces):
            start = content.find(piece, max(0, previous_end - self.overlap))
            end = start + len(piece)
            chunk_metadata = dict(metadata or {})
            chunk_metadata.update(start_offset=start, end_offset=end,
                                  chunk_index=i, document_path=path)
            page_spans = chunk_metadata.get("page_spans", [])
            if page_spans:
                chunk_metadata["pages"] = [p["page"] for p in page_spans
                    if p["start_offset"] < end and p["end_offset"] > start]
            self._chunks.append(LocalChunk(
                doc_title=title or path or "document",
                doc_path=path,
                content=piece,
                chunk_index=i,
                metadata=chunk_metadata,
            ))
            previous_end = end
        self._doc_titles.append(title or path or "document")
        return len(pieces)

    @property
    def document_count(self) -> int:
        return len(self._doc_titles)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def is_empty(self) -> bool:
        return not self._chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        min_score: float = 0.05,
    ) -> List[LocalChunk]:
        """
        Retrieve the top-k chunks most relevant to a query.

        Args:
            query: Search query / topic text
            top_k: Maximum chunks to return
            keywords: Additional keywords for scoring
            min_score: Minimum keyword score to include

        Returns:
            Chunks sorted by score descending (fresh copies with scores set)
        """
        if self.is_empty():
            return []

        keywords = keywords or []
        scored: List[LocalChunk] = []
        for chunk in self._chunks:
            score = score_relevance_simple(
                content=chunk.content,
                title=chunk.doc_title,
                research_topic=query,
                keywords=keywords,
            )
            if score >= min_score:
                scored.append(LocalChunk(
                    doc_title=chunk.doc_title,
                    doc_path=chunk.doc_path,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    score=score,
                    metadata=dict(chunk.metadata),
                ))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]
