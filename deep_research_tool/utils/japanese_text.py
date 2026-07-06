"""
Japanese text processing utilities using janome morphological analyzer.

Provides unified functions for:
- Tokenization (morphological analysis)
- Keyword extraction
- Sentence splitting
- Text normalization

All functions gracefully fall back to regex-based methods if janome is unavailable.
"""

import re
from typing import List, Optional, Tuple

# Lazy-load janome to avoid import errors when not installed
_tokenizer = None
_janome_available: Optional[bool] = None


def _get_tokenizer():
    """Lazy-load janome tokenizer (singleton)."""
    global _tokenizer, _janome_available
    if _janome_available is None:
        try:
            from janome.tokenizer import Tokenizer
            _tokenizer = Tokenizer()
            _janome_available = True
        except ImportError:
            _janome_available = False
            print("[japanese_text] janome not installed. Using regex fallback. "
                  "Install with: pip install janome")
    return _tokenizer


def is_japanese(text: str) -> bool:
    """Check if text contains Japanese characters."""
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))


# ============================================================================
# Tokenization
# ============================================================================

def tokenize(text: str) -> List[dict]:
    """
    Tokenize Japanese text into morphemes using janome.

    Returns list of dicts with keys:
        - surface: surface form (as-is text)
        - part_of_speech: part of speech (品詞)
        - base_form: dictionary/base form (原形)
        - reading: reading in katakana (読み)

    Falls back to character-based splitting if janome is unavailable.
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return _tokenize_fallback(text)

    tokens = []
    try:
        for token in tokenizer.tokenize(text):
            parts = token.part_of_speech.split(',')
            tokens.append({
                'surface': token.surface,
                'part_of_speech': parts[0] if parts else '',
                'part_of_speech_detail': parts[1] if len(parts) > 1 else '',
                'base_form': token.base_form if token.base_form != '*' else token.surface,
                'reading': token.reading if token.reading != '*' else '',
            })
    except Exception:
        return _tokenize_fallback(text)

    return tokens


def _tokenize_fallback(text: str) -> List[dict]:
    """Regex fallback for tokenization when janome is unavailable."""
    tokens = []
    # Split on common Japanese boundaries
    # This crude regex splits CJK, katakana, hiragana, latin characters as separate groups
    pattern = re.compile(
        r'([\u4E00-\u9FFF]+|'         # Kanji blocks
        r'[\u3040-\u309F]+|'           # Hiragana blocks
        r'[\u30A0-\u30FF]+|'           # Katakana blocks
        r'[A-Za-z0-9]+|'              # ASCII words
        r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\sA-Za-z0-9]+)'  # Punctuation etc.
    )
    for match in pattern.finditer(text):
        surface = match.group()
        if surface.strip():
            tokens.append({
                'surface': surface,
                'part_of_speech': 'unknown',
                'part_of_speech_detail': '',
                'base_form': surface,
                'reading': '',
            })
    return tokens


# ============================================================================
# Keyword Extraction
# ============================================================================

# Stop words (particles, auxiliary verbs, common function words)
_JA_STOP_POS = {'助詞', '助動詞', '接続詞', '記号', 'フィラー'}
_JA_STOP_POS_DETAIL = {'非自立', '接尾', '数'}
_JA_STOP_WORDS = {
    'の', 'に', 'は', 'を', 'が', 'と', 'で', 'も', 'や', 'から',
    'まで', 'より', 'など', 'について', 'における', 'として',
    'する', 'なる', 'ある', 'いる', 'れる', 'られる', 'こと', 'もの',
    'これ', 'それ', 'あれ', 'この', 'その', 'どの', 'ため', 'よう',
}
_EN_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "of", "to", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further",
    "then", "once", "and", "or", "but", "if", "than", "too",
    "very", "just", "only", "own", "same", "so", "more", "most",
    "other", "some", "such", "no", "nor", "not", "about", "what",
}


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    """
    Extract meaningful keywords from text (Japanese + English mixed).

    Uses janome to properly segment Japanese text into morphemes and
    extract content words (nouns, verbs, adjectives).

    Args:
        text: Input text (can be Japanese, English, or mixed)
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keyword strings
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return _extract_keywords_fallback(text, max_keywords)

    keywords = []
    try:
        for token in tokenizer.tokenize(text):
            surface = token.surface.strip()
            if not surface or len(surface) <= 1:
                continue

            parts = token.part_of_speech.split(',')
            pos = parts[0] if parts else ''
            pos_detail = parts[1] if len(parts) > 1 else ''

            # Skip stop parts of speech
            if pos in _JA_STOP_POS:
                continue
            if pos_detail in _JA_STOP_POS_DETAIL:
                continue

            # Skip known stop words
            if surface.lower() in _JA_STOP_WORDS or surface.lower() in _EN_STOP_WORDS:
                continue

            # Keep content words: nouns, verbs (base form), adjectives
            if pos in ('名詞', '動詞', '形容詞', '副詞') or pos == 'unknown':
                base = token.base_form if token.base_form != '*' else surface
                if base not in keywords and len(base) > 1:
                    keywords.append(base)

    except Exception:
        return _extract_keywords_fallback(text, max_keywords)

    return keywords[:max_keywords]


def _extract_keywords_fallback(text: str, max_keywords: int = 20) -> List[str]:
    """Regex-based keyword extraction fallback."""
    words = re.split(r'[\s、。，．・\-/()（）「」『』【】]+', text)
    all_stop = _JA_STOP_WORDS | _EN_STOP_WORDS
    keywords = [
        w for w in words
        if len(w) > 1 and w.lower() not in all_stop
    ]
    return keywords[:max_keywords]


def extract_content_words_set(text: str) -> set:
    """
    Extract a set of content words (base forms) for overlap/relevance checking.

    Useful for ToC validation where you need to check if topic keywords
    appear in section titles.

    Args:
        text: Input text

    Returns:
        Set of content word base forms (lowercase)
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        # Fallback: split and return
        words = re.split(r'[\s、。，．・\-/()（）「」『』【】]+', text)
        return {w.lower() for w in words if len(w) > 1 and w.lower() not in _JA_STOP_WORDS | _EN_STOP_WORDS}

    words = set()
    try:
        for token in tokenizer.tokenize(text):
            surface = token.surface.strip()
            if not surface or len(surface) <= 1:
                continue

            parts = token.part_of_speech.split(',')
            pos = parts[0] if parts else ''

            if pos in _JA_STOP_POS:
                continue

            if surface.lower() in _JA_STOP_WORDS or surface.lower() in _EN_STOP_WORDS:
                continue

            if pos in ('名詞', '動詞', '形容詞', '副詞', 'unknown'):
                base = token.base_form if token.base_form != '*' else surface
                words.add(base.lower())
    except Exception:
        words = re.split(r'[\s、。，．・]+', text)
        return {w.lower() for w in words if len(w) > 1}

    return words


# ============================================================================
# Sentence Splitting
# ============================================================================

def split_sentences(text: str, min_length: int = 5) -> List[str]:
    """
    Split Japanese/mixed text into sentences using morphological analysis.

    Uses janome to identify sentence boundaries more accurately than
    regex-only approaches. Handles:
    - Explicit punctuation (。！？)
    - Closing brackets after punctuation (。」)
    - Mixed Japanese/English text
    - Sentences ending without explicit punctuation (e.g., verb forms)

    Args:
        text: Input text
        min_length: Minimum sentence length to include

    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []

    if not is_japanese(text):
        return _split_sentences_non_ja(text, min_length)

    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return _split_sentences_regex(text, min_length)

    try:
        return _split_sentences_janome(text, tokenizer, min_length)
    except Exception:
        return _split_sentences_regex(text, min_length)


def _split_sentences_janome(text: str, tokenizer, min_length: int) -> List[str]:
    """Split Japanese sentences using janome morphological analysis."""
    sentences = []
    current_text = ""

    # Sentence-ending punctuation (by surface character)
    sentence_enders = {'。', '！', '？', '．', '…'}
    # English sentence endings
    en_sentence_enders = {'.', '!', '?'}
    # Closing brackets that can follow a sentence ender
    closing_brackets = {'」', '』', '）', ')', '】', '》'}
    # Paragraph/line break as sentence boundary
    newline_pattern = re.compile(r'\n+')

    def _find_next_content_token(tokens_list, start_idx):
        """Look ahead past whitespace tokens to find next content token."""
        j = start_idx
        while j < len(tokens_list):
            s = tokens_list[j].surface.strip()
            if s:
                return s
            j += 1
        return None

    # Pre-split by double newlines (paragraph boundaries)
    paragraphs = newline_pattern.split(text)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        tokens = list(tokenizer.tokenize(paragraph))
        skip_until = -1  # Index to skip already consumed tokens

        for i, token in enumerate(tokens):
            if i <= skip_until:
                continue

            surface = token.surface
            pos = token.part_of_speech.split(',')[0]

            current_text += surface

            is_ja_ender = surface in sentence_enders or (pos == '記号' and surface in sentence_enders)
            is_en_ender = surface in en_sentence_enders

            if is_ja_ender:
                # Look ahead for closing brackets
                j = i + 1
                while j < len(tokens) and tokens[j].surface in closing_brackets:
                    current_text += tokens[j].surface
                    skip_until = j
                    j += 1

                # Commit sentence
                sent = current_text.strip()
                if len(sent) >= min_length:
                    sentences.append(sent)
                current_text = ""
                continue

            # Handle English sentence endings in mixed text
            # janome may not classify "." as punctuation, so check surface directly
            if is_en_ender:
                # Look ahead past whitespace to find next content token
                next_content = _find_next_content_token(tokens, i + 1)
                if next_content is not None:
                    # New sentence if next content starts with uppercase or is Japanese
                    starts_new = (next_content[0].isupper() or is_japanese(next_content))
                    if starts_new:
                        sent = current_text.strip()
                        if len(sent) >= min_length:
                            sentences.append(sent)
                        current_text = ""
                        continue

        # End of paragraph - flush remaining text
        if current_text.strip():
            sent = current_text.strip()
            if len(sent) >= min_length:
                sentences.append(sent)
            current_text = ""

    return sentences


def _split_sentences_regex(text: str, min_length: int) -> List[str]:
    """Regex-based Japanese sentence splitting fallback."""
    # Split on Japanese and English sentence-ending punctuation
    pattern = r'([。！？．…]+[」』）\)】]?|(?<=[^0-9])[.!?]+(?=\s+[A-Z])|(?<=[^0-9])[.!?]+(?:\s|$))'

    parts = re.split(pattern, text)

    sentences = []
    current = ""

    for part in parts:
        if not part:
            continue

        if re.match(r'^[。！？．…]+[」』）\)】]?$', part) or re.match(r'^[.!?]+$', part.strip()):
            current += part
            if current.strip() and len(current.strip()) >= min_length:
                sentences.append(current.strip())
            current = ""
        else:
            current += part

    if current.strip() and len(current.strip()) >= min_length:
        sentences.append(current.strip())

    # Also split on double newlines
    final_sentences = []
    for sent in sentences:
        for sub in re.split(r'\n\n+', sent):
            sub = sub.strip()
            if sub and len(sub) >= min_length:
                final_sentences.append(sub)

    return final_sentences


def _split_sentences_non_ja(text: str, min_length: int) -> List[str]:
    """Split non-Japanese (English) text into sentences."""
    # Simple English sentence splitting
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\n)|(?<=[.!?])$'
    raw_parts = re.split(pattern, text)

    sentences = []
    for part in raw_parts:
        part = part.strip()
        if part and len(part) >= min_length:
            sentences.append(part)

    return sentences


# ============================================================================
# Title / ToC Utilities
# ============================================================================

def title_contains_topic_keywords(title: str, query: str) -> bool:
    """
    Check if a section title contains keywords from the research query.

    Uses morphological analysis for accurate word matching across Japanese
    and English text.

    Args:
        title: Section title
        query: Research query/topic

    Returns:
        True if title contains at least one topic keyword
    """
    query_words = extract_content_words_set(query)
    title_words = extract_content_words_set(title)

    # Check for overlap
    overlap = query_words & title_words
    return len(overlap) > 0


def is_generic_title_morphological(title: str, generic_titles: List[str]) -> bool:
    """
    Check if a title is generic using morphological analysis.

    More accurate than simple substring matching because it compares
    base forms of words.

    Args:
        title: Section title to check
        generic_titles: List of generic title strings

    Returns:
        True if title is generic
    """
    title_lower = title.lower().strip()

    # Direct match first
    for generic in generic_titles:
        if generic in title_lower or title_lower == generic:
            return True

    # Use morphological analysis for base form comparison
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return False

    try:
        title_bases = set()
        for token in tokenizer.tokenize(title):
            base = token.base_form if token.base_form != '*' else token.surface
            title_bases.add(base.lower())

        for generic in generic_titles:
            generic_bases = set()
            for token in tokenizer.tokenize(generic):
                base = token.base_form if token.base_form != '*' else token.surface
                generic_bases.add(base.lower())

            # If all generic title base forms are in the title, it's generic
            if generic_bases and generic_bases.issubset(title_bases):
                return True

    except Exception:
        pass

    return False
