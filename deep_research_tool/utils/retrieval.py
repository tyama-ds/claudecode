"""Small deterministic source-span selector; no remote models or indexes."""
import re
from typing import List, Optional

from .helpers import chunk_text
from .japanese_text import extract_keywords


def select_relevant_spans(text: str, query: str, keywords: Optional[List[str]] = None,
                          max_chars: int = 2000) -> str:
    """Select original text windows across the entire source within a budget.

    Source order is preserved after ranking, and an ellipsis separates gaps.
    This is an input selector, not a generated summary or relevance verdict.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    terms = {t.casefold() for t in (keywords or []) if isinstance(t, str) and t.strip()}
    terms.update(t.casefold() for t in extract_keywords(query, max_keywords=30))
    # Identifiers and numbers often distinguish the precise requested evidence.
    terms.update(re.findall(r"[A-Za-z0-9_]{2,}", query.casefold()))
    window = min(700, max_chars)
    overlap = min(100, window // 5)
    chunks = chunk_text(text, chunk_size=window, overlap=overlap)
    ranked = []
    for index, chunk in enumerate(chunks):
        folded = chunk.casefold()
        matches = [term for term in terms if term in folded]
        score = sum(min(len(term), 12) for term in matches)
        ranked.append((score, index, chunk))
    # If no query term occurs, sample the whole source instead of implying the
    # beginning represents all content. Keep the first, middle and last windows.
    if not any(score for score, _, _ in ranked):
        indices = list(dict.fromkeys((0, len(chunks) // 2, len(chunks) - 1)))
        ranked = [(0, index, chunks[index]) for index in indices]
    else:
        ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    remaining = max_chars
    for _, index, chunk in ranked:
        separator = 5 if selected else 0
        if remaining <= separator:
            break
        piece = chunk[:remaining - separator]
        selected.append((index, piece))
        remaining -= len(piece) + separator
    selected.sort(key=lambda item: item[0])
    return "\n...\n".join(piece for _, piece in selected)
