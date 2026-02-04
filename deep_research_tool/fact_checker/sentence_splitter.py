"""
Sentence splitter for breaking text into individual sentences.

Uses a hybrid approach:
- Rule-based splitting for reliability (primary)
- Optional NLP-based splitting using spaCy
- Optional LLM-based splitting for complex cases

Supports Japanese, English, and mixed-language text.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class SplitMethod(str, Enum):
    """Sentence splitting methods."""
    RULE_BASED = "rule_based"
    SPACY = "spacy"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class Sentence:
    """A single sentence extracted from text."""
    text: str
    index: int
    start_pos: int
    end_pos: int
    language: str = "unknown"
    has_reference: bool = False
    reference_urls: List[str] = field(default_factory=list)


class SentenceSplitter:
    """
    Split text into individual sentences.

    Supports multiple splitting strategies:
    - Rule-based: Fast and reliable for most cases
    - spaCy: NLP-based for better accuracy
    - LLM: For complex cases (slower but more accurate)
    - Hybrid: Combines rule-based with LLM for ambiguous cases
    """

    # Japanese sentence endings
    JA_ENDINGS = ['。', '！', '？', '．', '…']

    # English/Western sentence endings
    EN_ENDINGS = ['.', '!', '?', '...']

    # Common abbreviations (should not split after these)
    ABBREVIATIONS = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'inc', 'ltd', 'corp',
        'vs', 'etc', 'eg', 'ie', 'no', 'vol', 'pp', 'ed', 'eds',
        'fig', 'figs', 'ref', 'refs', 'sec', 'secs', 'ch', 'chs',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    }

    # URL pattern for reference extraction
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[^\s]*)?'
    )

    def __init__(
        self,
        method: SplitMethod = SplitMethod.RULE_BASED,
        min_sentence_length: int = 5,
        llm_client=None,
        spacy_model: str = "ja_core_news_sm",
    ):
        """
        Initialize sentence splitter.

        Args:
            method: Splitting method to use
            min_sentence_length: Minimum characters for a valid sentence
            llm_client: LLM client for LLM-based splitting
            spacy_model: spaCy model name for NLP-based splitting
        """
        self.method = method
        self.min_sentence_length = min_sentence_length
        self.llm_client = llm_client
        self.spacy_model_name = spacy_model
        self._spacy_nlp = None

    def _get_spacy_nlp(self):
        """Lazy load spaCy model."""
        if self._spacy_nlp is None:
            try:
                import spacy
                self._spacy_nlp = spacy.load(self.spacy_model_name)
            except ImportError:
                raise ImportError(
                    "spaCy is required for NLP-based splitting. "
                    "Install with: pip install spacy && python -m spacy download ja_core_news_sm"
                )
            except OSError:
                # Model not found, try downloading
                import spacy
                from spacy.cli import download
                download(self.spacy_model_name)
                self._spacy_nlp = spacy.load(self.spacy_model_name)
        return self._spacy_nlp

    def split(self, text: str) -> List[Sentence]:
        """
        Split text into sentences.

        Args:
            text: Input text to split

        Returns:
            List of Sentence objects
        """
        if not text or not text.strip():
            return []

        # Normalize whitespace
        text = self._normalize_text(text)

        if self.method == SplitMethod.RULE_BASED:
            return self._split_rule_based(text)
        elif self.method == SplitMethod.SPACY:
            return self._split_spacy(text)
        elif self.method == SplitMethod.LLM:
            return self._split_llm(text)
        elif self.method == SplitMethod.HYBRID:
            return self._split_hybrid(text)
        else:
            return self._split_rule_based(text)

    def _normalize_text(self, text: str) -> str:
        """Normalize text before splitting."""
        # Replace multiple whitespace with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _split_rule_based(self, text: str) -> List[Sentence]:
        """
        Rule-based sentence splitting.

        Handles:
        - Japanese sentences (ending with 。！？)
        - English sentences (ending with .!?)
        - Mixed language text
        - Abbreviations
        - Numbers with decimals
        - URLs and references
        """
        sentences = []
        current_pos = 0

        # Detect language
        has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

        if has_japanese:
            # Japanese-aware splitting
            raw_sentences = self._split_japanese(text)
        else:
            # English/Western text splitting
            raw_sentences = self._split_english(text)

        # Process each sentence
        for idx, sent_text in enumerate(raw_sentences):
            sent_text = sent_text.strip()
            if len(sent_text) < self.min_sentence_length:
                continue

            # Find position in original text
            start_pos = text.find(sent_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(sent_text)
            current_pos = end_pos

            # Extract references/URLs
            urls = self.URL_PATTERN.findall(sent_text)

            sentence = Sentence(
                text=sent_text,
                index=len(sentences),
                start_pos=start_pos,
                end_pos=end_pos,
                language="ja" if has_japanese else "en",
                has_reference=bool(urls),
                reference_urls=urls,
            )
            sentences.append(sentence)

        return sentences

    def _split_japanese(self, text: str) -> List[str]:
        """Split Japanese text into sentences."""
        # Pattern for Japanese sentence endings
        # Handle 。」 (closing quote after period) as single ending
        pattern = r'([。！？．…]+[」』）\)】]?|(?<=[^0-9])[.!?]+(?:\s|$))'

        parts = re.split(pattern, text)

        sentences = []
        current = ""

        for i, part in enumerate(parts):
            if not part:
                continue

            if re.match(r'^[。！？．…]+[」』）\)】]?$', part) or re.match(r'^[.!?]+$', part):
                # This is an ending, append to current
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part

        # Don't forget the last part
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _split_english(self, text: str) -> List[str]:
        """Split English text into sentences."""
        # Pre-process: protect abbreviations
        protected_text = text
        for abbr in self.ABBREVIATIONS:
            # Case insensitive replacement
            pattern = rf'\b({abbr})\.'
            protected_text = re.sub(
                pattern,
                rf'\1<PERIOD>',
                protected_text,
                flags=re.IGNORECASE
            )

        # Protect decimal numbers
        protected_text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', protected_text)

        # Protect URLs
        url_placeholders = {}
        for i, match in enumerate(self.URL_PATTERN.finditer(protected_text)):
            placeholder = f'<URL_{i}>'
            url_placeholders[placeholder] = match.group()
            protected_text = protected_text.replace(match.group(), placeholder, 1)

        # Split on sentence endings
        pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\n)|(?<=[.!?])$'
        raw_parts = re.split(pattern, protected_text)

        sentences = []
        for part in raw_parts:
            if not part.strip():
                continue

            # Restore protected elements
            restored = part.replace('<PERIOD>', '.').replace('<DECIMAL>', '.')
            for placeholder, url in url_placeholders.items():
                restored = restored.replace(placeholder, url)

            sentences.append(restored.strip())

        return sentences

    def _split_spacy(self, text: str) -> List[Sentence]:
        """Split using spaCy NLP."""
        nlp = self._get_spacy_nlp()
        doc = nlp(text)

        sentences = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if len(sent_text) < self.min_sentence_length:
                continue

            urls = self.URL_PATTERN.findall(sent_text)

            # Detect language for this sentence
            has_ja = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', sent_text))

            sentence = Sentence(
                text=sent_text,
                index=len(sentences),
                start_pos=sent.start_char,
                end_pos=sent.end_char,
                language="ja" if has_ja else "en",
                has_reference=bool(urls),
                reference_urls=urls,
            )
            sentences.append(sentence)

        return sentences

    def _split_llm(self, text: str) -> List[Sentence]:
        """Split using LLM for complex cases."""
        if self.llm_client is None:
            raise ValueError("LLM client required for LLM-based splitting")

        prompt = f"""Split the following text into individual sentences.
Return ONLY a JSON array of sentence strings. Do not include any explanation.

Text:
{text[:8000]}

Return format: ["sentence 1", "sentence 2", ...]"""

        response = self.llm_client.generate(prompt)

        try:
            import json
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                raw_sentences = json.loads(content[start:end])
            else:
                # Fallback to rule-based
                return self._split_rule_based(text)
        except (json.JSONDecodeError, ValueError):
            return self._split_rule_based(text)

        # Convert to Sentence objects
        sentences = []
        current_pos = 0

        for sent_text in raw_sentences:
            sent_text = sent_text.strip()
            if len(sent_text) < self.min_sentence_length:
                continue

            start_pos = text.find(sent_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(sent_text)
            current_pos = end_pos

            urls = self.URL_PATTERN.findall(sent_text)
            has_ja = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', sent_text))

            sentence = Sentence(
                text=sent_text,
                index=len(sentences),
                start_pos=start_pos,
                end_pos=end_pos,
                language="ja" if has_ja else "en",
                has_reference=bool(urls),
                reference_urls=urls,
            )
            sentences.append(sentence)

        return sentences

    def _split_hybrid(self, text: str) -> List[Sentence]:
        """
        Hybrid splitting: rule-based first, then LLM for validation/refinement.
        """
        # First pass: rule-based splitting
        rule_sentences = self._split_rule_based(text)

        if self.llm_client is None:
            return rule_sentences

        # Only use LLM if text might have ambiguous splits
        # Check for potential issues
        needs_llm = False
        for sent in rule_sentences:
            # Very long sentences might need re-splitting
            if len(sent.text) > 500:
                needs_llm = True
                break
            # Sentences with multiple internal periods might be mis-split
            if sent.text.count('.') > 3 or sent.text.count('。') > 3:
                needs_llm = True
                break

        if not needs_llm:
            return rule_sentences

        # Use LLM to refine
        return self._split_llm(text)

    def split_with_context(
        self,
        text: str,
        context_window: int = 1
    ) -> List[Tuple[Sentence, List[Sentence], List[Sentence]]]:
        """
        Split text and return sentences with surrounding context.

        Args:
            text: Input text
            context_window: Number of sentences before/after to include

        Returns:
            List of (sentence, prev_sentences, next_sentences) tuples
        """
        sentences = self.split(text)
        result = []

        for i, sent in enumerate(sentences):
            prev_start = max(0, i - context_window)
            next_end = min(len(sentences), i + context_window + 1)

            prev_sentences = sentences[prev_start:i]
            next_sentences = sentences[i+1:next_end]

            result.append((sent, prev_sentences, next_sentences))

        return result
