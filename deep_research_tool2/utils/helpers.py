"""
Helper utilities for Deep Research Tool.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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
    logger = logging.getLogger("deep_research_tool2")

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
