"""
Layer 1: Content Sanitizer.

Neutralizes hidden instructions and prompt injection payloads
in external content (HTML, PDF text, Markdown) before it reaches
the LLM.  Does NOT attempt to understand semantics — operates
purely on structural / character-level patterns.
"""

import re
import unicodedata
from typing import Optional

from .config import SecurityConfig


# Zero-width and directional override characters
_INVISIBLE_CHARS = set(
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202a"  # LEFT-TO-RIGHT EMBEDDING
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202c"  # POP DIRECTIONAL FORMATTING
    "\u202d"  # LEFT-TO-RIGHT OVERRIDE
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\ufff9"  # INTERLINEAR ANNOTATION ANCHOR
    "\ufffa"  # INTERLINEAR ANNOTATION SEPARATOR
    "\ufffb"  # INTERLINEAR ANNOTATION TERMINATOR
)

# HTML comment pattern
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->", re.DOTALL)

# CSS hidden-text patterns (inline style)
_CSS_HIDDEN_RE = re.compile(
    r"<[^>]+style\s*=\s*[\"'][^\"']*"
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden|"
    r"font-size\s*:\s*0|opacity\s*:\s*0|"
    r"color\s*:\s*(?:white|#fff(?:fff)?|rgba?\([^)]*,\s*0\s*\)))"
    r"[^\"']*[\"'][^>]*>[\s\S]*?</[^>]+>",
    re.IGNORECASE | re.DOTALL,
)

# Markdown link injection: [text](javascript:...) or [](hidden instruction)
_MD_LINK_INJECTION_RE = re.compile(
    r"\[([^\]]*)\]\(\s*(?:javascript:|data:|vbscript:)[^\)]*\)",
    re.IGNORECASE,
)

# Markdown hidden instruction in empty links: [](some instruction here)
_MD_HIDDEN_INSTRUCTION_RE = re.compile(
    r"\[\s*\]\([^\)]{20,}\)",
)

# HTML <script>, <style>, <iframe>, <object>, <embed> tags
_DANGEROUS_TAGS_RE = re.compile(
    r"<\s*(?:script|style|iframe|object|embed|applet|form|input|button|textarea)"
    r"[^>]*>[\s\S]*?</\s*(?:script|style|iframe|object|embed|applet|form|input|button|textarea)\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Standalone dangerous tags (self-closing or unclosed)
_DANGEROUS_TAGS_SELF_RE = re.compile(
    r"<\s*(?:script|iframe|object|embed|applet|form|input|meta\s+http-equiv)[^>]*/?\s*>",
    re.IGNORECASE,
)

# Common PI trigger phrases (multi-language)
_PI_TRIGGER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?",
        r"disregard\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?",
        r"forget\s+(?:all\s+)?(?:previous|your)\s+instructions?",
        r"you\s+are\s+now\s+(?:a|an|in)\s+",
        r"new\s+instructions?\s*:",
        r"system\s*(?:prompt|message)\s*:",
        r"<\s*system\s*>",
        r"\[SYSTEM\]",
        r"以前の指示を(?:無視|忘れ)",
        r"新しい指示\s*[:：]",
        r"あなたは今から",
    ]
]


class ContentSanitizer:
    """
    Sanitizes external content to remove hidden instructions and
    prompt injection payloads.

    This is a structural/character-level defense — it removes
    content that is invisible to humans but visible to LLMs.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._pi_detections: list[dict] = []

    def sanitize(self, content: str, source_url: str = "") -> str:
        """
        Sanitize external content before passing to LLM.

        Args:
            content: Raw content from external source
            source_url: URL of the source (for logging)

        Returns:
            Sanitized content string
        """
        if not content or not self.config.sanitize_external_content:
            return content

        original_len = len(content)

        # Truncate oversized content
        if len(content) > self.config.max_content_length:
            content = content[: self.config.max_content_length]

        # Strip dangerous HTML tags
        content = _DANGEROUS_TAGS_RE.sub("", content)
        content = _DANGEROUS_TAGS_SELF_RE.sub("", content)

        # Strip CSS-hidden text
        if self.config.strip_hidden_text:
            content = _CSS_HIDDEN_RE.sub("", content)

        # Strip HTML comments
        if self.config.strip_html_comments:
            content = _HTML_COMMENT_RE.sub("", content)

        # Strip control/invisible characters
        if self.config.strip_control_chars:
            content = self._strip_invisible_chars(content)

        # Strip Markdown injection patterns
        if self.config.strip_markdown_injection:
            content = _MD_LINK_INJECTION_RE.sub(r"[\1](removed)", content)
            content = _MD_HIDDEN_INSTRUCTION_RE.sub("", content)

        # Detect (and optionally flag) PI trigger phrases
        content = self._scan_pi_triggers(content, source_url)

        return content

    def sanitize_html(self, html: str, source_url: str = "") -> str:
        """
        Sanitize raw HTML with additional HTML-specific processing.

        Strips all tags and returns plain text, then applies
        standard sanitization.
        """
        if not html:
            return html

        # Remove all HTML tags, keep text content
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return self.sanitize(text, source_url)

    def get_pi_detections(self) -> list[dict]:
        """Return list of detected PI trigger patterns."""
        return list(self._pi_detections)

    def reset_detections(self):
        """Clear detection history."""
        self._pi_detections.clear()

    def _strip_invisible_chars(self, text: str) -> str:
        """Remove zero-width, directional override, and other invisible chars."""
        result = []
        for ch in text:
            if ch in _INVISIBLE_CHARS:
                continue
            # Also strip C0/C1 control chars except whitespace
            cat = unicodedata.category(ch)
            if cat.startswith("C") and ch not in ("\n", "\r", "\t"):
                continue
            result.append(ch)
        return "".join(result)

    def _scan_pi_triggers(self, content: str, source_url: str) -> str:
        """
        Scan for common prompt injection trigger phrases.

        In PARANOID mode, strips the phrases. Otherwise, logs them
        as warnings but leaves content intact (to avoid false positives
        breaking legitimate content).
        """
        for pattern in _PI_TRIGGER_PATTERNS:
            match = pattern.search(content)
            if match:
                self._pi_detections.append({
                    "pattern": pattern.pattern,
                    "matched_text": match.group()[:100],
                    "source_url": source_url,
                })
                if self.config.block_on_pi_detection:
                    content = pattern.sub("[REMOVED:PI]", content)

        return content
