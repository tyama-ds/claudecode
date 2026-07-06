"""
Helper utilities for Deep Research Tool.
"""

import json
import logging
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ResearchWarnings – thread-safe collector for fallback / degradation notices
# ---------------------------------------------------------------------------

class ResearchWarnings:
    """Collect warnings emitted when fallback mechanisms trigger.

    Warnings are categorised by severity so that they can be filtered and
    rendered in the final report.

    Severity levels:
        CRITICAL – silent data loss; output likely incorrect or incomplete
        HIGH     – an entire optional feature was skipped
        MEDIUM   – partial data loss within a feature
        LOW      – cosmetic / formatting degradation
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    _instance: Optional["ResearchWarnings"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._warnings: List[Dict[str, str]] = []
        self._lock_inst = threading.Lock()

    # --- singleton access (so every module can record warnings) ---
    @classmethod
    def get_instance(cls) -> "ResearchWarnings":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (call at the start of each run)."""
        with cls._lock:
            cls._instance = cls()

    # --- recording ---
    def add(self, severity: str, source: str, message: str) -> None:
        """Add a warning.

        Args:
            severity: One of CRITICAL / HIGH / MEDIUM / LOW
            source: Short component identifier, e.g. "ReportGeneratorV2"
            message: Human-readable description of what happened
        """
        entry = {
            "severity": severity,
            "source": source,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock_inst:
            self._warnings.append(entry)
        # Also emit to console for immediate visibility
        print(f"[WARNING:{severity}] [{source}] {message}")

    # --- querying ---
    def get_all(self) -> List[Dict[str, str]]:
        with self._lock_inst:
            return list(self._warnings)

    def has_warnings(self) -> bool:
        with self._lock_inst:
            return len(self._warnings) > 0

    def count(self, min_severity: str = None) -> int:
        if min_severity is None:
            return len(self._warnings)
        order = {self.CRITICAL: 0, self.HIGH: 1, self.MEDIUM: 2, self.LOW: 3}
        threshold = order.get(min_severity, 3)
        with self._lock_inst:
            return sum(1 for w in self._warnings
                       if order.get(w["severity"], 3) <= threshold)

    # --- rendering for report footer ---
    def to_report_section(self, language: str = "ja") -> str:
        """Render a markdown section suitable for appending to a report."""
        with self._lock_inst:
            if not self._warnings:
                return ""

        order = {self.CRITICAL: 0, self.HIGH: 1, self.MEDIUM: 2, self.LOW: 3}
        sorted_warnings = sorted(self._warnings,
                                 key=lambda w: order.get(w["severity"], 3))

        if language == "ja":
            lines = ["\n\n---\n", "## 処理中の警告・注意事項\n",
                     "以下のフォールバックが発生しました。出力品質に影響がある可能性があります。\n"]
        else:
            lines = ["\n\n---\n", "## Processing Warnings\n",
                     "The following fallbacks occurred during processing. "
                     "Output quality may be affected.\n"]

        for w in sorted_warnings:
            lines.append(f"- **[{w['severity']}]** `{w['source']}`: {w['message']}")

        lines.append("")
        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict[str, str]]:
        return self.get_all()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: str = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("deep_research_tool")

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Default format
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def format_timestamp(dt: datetime = None, format: str = "iso") -> str:
    """
    Format a datetime object.

    Args:
        dt: Datetime object (default: now)
        format: Format type ("iso", "date", "time", "full", "filename")

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()

    formats = {
        "iso": "%Y-%m-%dT%H:%M:%S",
        "date": "%Y-%m-%d",
        "time": "%H:%M:%S",
        "full": "%Y-%m-%d %H:%M:%S",
        "filename": "%Y%m%d_%H%M%S",
    }

    fmt = formats.get(format, formats["iso"])
    return dt.strftime(fmt)


def truncate_text(
    text: str,
    max_length: int = 1000,
    suffix: str = "...",
    preserve_words: bool = True,
) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        preserve_words: Try to break at word boundaries

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)

    if preserve_words:
        # Find last space before truncate point
        last_space = text.rfind(" ", 0, truncate_at)
        if last_space > truncate_at * 0.5:  # Only if not too far back
            truncate_at = last_space

    return text[:truncate_at].rstrip() + suffix


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        filename: Original filename
        max_length: Maximum length for the filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Replace multiple spaces/underscores with single
    import re
    filename = re.sub(r'[_\s]+', '_', filename)

    # Truncate if needed
    if len(filename) > max_length:
        # Preserve extension if present
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
            name = name[:max_length - len(ext) - 1]
            filename = f"{name}.{ext}"
        else:
            filename = filename[:max_length]

    return filename.strip("_")


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.

    This is a rough estimate based on average token length.
    For precise counts, use the tokenizer for the specific model.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Rough estimate: ~4 characters per token for English
    # ~2 characters per token for Japanese/Chinese
    has_cjk = any(ord(char) > 0x4E00 for char in text[:100])

    if has_cjk:
        return len(text) // 2
    else:
        return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 200,
    preserve_paragraphs: bool = True,
) -> list:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Target size for each chunk (in characters)
        overlap: Overlap between chunks
        preserve_paragraphs: Try to break at paragraph boundaries

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []

    if preserve_paragraphs:
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap
                overlap_paras = []
                overlap_size = 0
                for p in reversed(current_chunk):
                    if overlap_size + len(p) > overlap:
                        break
                    overlap_paras.insert(0, p)
                    overlap_size += len(p)

                current_chunk = overlap_paras
                current_size = overlap_size

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

    else:
        # Simple character-based chunking
        start = 0
        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Try to find a good break point
                for sep in ["\n\n", "\n", ". ", " "]:
                    break_point = text.rfind(sep, start, end)
                    if break_point > start + chunk_size * 0.5:
                        end = break_point + len(sep)
                        break

            chunks.append(text[start:end])
            start = end - overlap

    return chunks


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from LLM response text.

    Handles common LLM output patterns:
    - JSON wrapped in markdown code blocks (```json ... ``` or ``` ... ```)
    - JSON with surrounding text
    - Plain JSON

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValueError: If no valid JSON object is found
    """
    if not text:
        raise ValueError("Empty response text")

    # Strip markdown code blocks first
    # Match ```json\n...\n``` or ```\n...\n```
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object by matching braces
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found in response")

    # Find matching closing brace by tracking nesting
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try continuing to find another valid JSON
                    break

    # Fallback: try first '{' to last '}'
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response")


def extract_json_array_from_response(text: str) -> List[Any]:
    """
    Extract a top-level JSON array from LLM response text.

    Companion to extract_json_from_response for responses whose top-level
    structure is a JSON array rather than an object. Handles markdown code
    blocks and surrounding text.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as a list

    Raises:
        ValueError: If no valid JSON array is found
    """
    if not text:
        raise ValueError("Empty response text")

    # Strip markdown code blocks first
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Find JSON array by matching brackets
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in response")

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    break

    # Fallback: first '[' to last ']'
    end = text.rfind("]") + 1
    if end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON array found in response")


def split_prose_and_meta(text: str, delimiter: str) -> Tuple[str, Dict[str, Any]]:
    """
    Split an LLM response into (prose_body, metadata_dict).

    Supports the "prose + delimiter + metadata JSON" output convention used
    for report text generation, while remaining backward compatible with
    legacy responses that wrap the prose in a JSON object.

    Fallback chain:
      1. If the delimiter is present: body is everything before it (code
         fences stripped), metadata is parsed from everything after it.
         Metadata parse failure yields an empty dict, never an error.
      2. Else, if the whole text parses as a JSON object with a "content"
         key: legacy JSON mode -- content becomes the body, the remaining
         keys become the metadata.
      3. Else: the whole text (code fences stripped) is the body, metadata
         is empty.

    Args:
        text: Raw LLM response text
        delimiter: Delimiter line separating prose from metadata JSON

    Returns:
        Tuple of (prose body, metadata dict)
    """
    if not text:
        return "", {}

    def _strip_fences(s: str) -> str:
        s = s.strip()
        # Remove a wrapping code fence if the entire text is fenced
        fence_match = re.fullmatch(r'```(?:\w+)?\s*\n(.*?)\n?\s*```', s, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        return s

    if delimiter in text:
        body_part, _, meta_part = text.partition(delimiter)
        body = _strip_fences(body_part)
        try:
            meta = extract_json_from_response(meta_part)
        except (ValueError, json.JSONDecodeError):
            meta = {}
        return body, meta

    # Legacy JSON mode: whole response is a JSON object with "content"
    try:
        data = extract_json_from_response(text)
        if isinstance(data, dict) and "content" in data:
            body = data.pop("content") or ""
            return body, data
    except (ValueError, json.JSONDecodeError):
        pass

    return _strip_fences(text), {}


def merge_dicts(*dicts, deep: bool = True) -> dict:
    """
    Merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge
        deep: Perform deep merge for nested dicts

    Returns:
        Merged dictionary
    """
    result = {}

    for d in dicts:
        if not isinstance(d, dict):
            continue

        for key, value in d.items():
            if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value

    return result
