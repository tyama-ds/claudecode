"""
Base class for web search clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    content: str = ""
    images: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "images": self.images,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class PageContent:
    """Extracted content from a web page."""
    url: str
    title: str
    text_content: str
    html_content: str = ""
    images: List[Dict[str, str]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "text_content": self.text_content,
            "html_content": self.html_content,
            "images": self.images,
            "links": self.links,
            "metadata": self.metadata,
            "extracted_at": self.extracted_at.isoformat(),
        }


class BaseSearchClient(ABC):
    """Abstract base class for web search clients."""

    def __init__(
        self,
        max_results: int = 10,
        timeout: int = 30,
        extract_images: bool = True,
        max_images: int = 5,
    ):
        """
        Initialize the search client.

        Args:
            max_results: Maximum number of search results
            timeout: Request timeout in seconds
            extract_images: Whether to extract images from pages
            max_images: Maximum number of images to extract per page
        """
        self.max_results = max_results
        self.timeout = timeout
        self.extract_images = extract_images
        self.max_images = max_images
        # Optional run-scoped concurrency limiter (utils.concurrency
        # RunLimits): leaf HTTP operations take one composed permit
        self.concurrency_limiter = None

    def _leaf_permit(self):
        from ..utils.concurrency import maybe_permit
        return maybe_permit(getattr(self, "concurrency_limiter", None))

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Perform a web search.

        Args:
            query: The search query
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def get_page_content(self, url: str, **kwargs) -> PageContent:
        """
        Extract content from a specific URL.

        Args:
            url: The URL to fetch
            **kwargs: Additional parameters

        Returns:
            Extracted page content
        """
        pass

    def search_and_extract(
        self,
        query: str,
        max_pages: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search and extract content from top results.

        Args:
            query: The search query
            max_pages: Maximum number of pages to extract content from
            **kwargs: Additional parameters

        Returns:
            List of search results with extracted content
        """
        results = self.search(query, **kwargs)

        for i, result in enumerate(results[:max_pages]):
            try:
                page_content = self.get_page_content(result.url)
                result.content = page_content.text_content
                result.images = [img.get("src", "") for img in page_content.images]
                result.metadata.update(page_content.metadata)
            except Exception as e:
                result.metadata["extraction_error"] = str(e)

        return results

    @staticmethod
    def simplify_query(query: str, level: int = 1) -> str:
        """
        Simplify a search query by progressively removing detail.

        Level 1: Remove text inside brackets/parentheses and after colons.
        Level 2: Strip stop-words, keep keywords only.
        Level 3: Keep at most 3 keywords (broadest).

        Args:
            query: Original query string
            level: Simplification level (1-3)

        Returns:
            Simplified query string. Returns original if simplification
            produces an empty string.
        """
        import re

        if not query or not query.strip():
            return query

        text = query

        # ---- Level 1: remove parenthesized / bracketed content & colon-suffix ----
        if level >= 1:
            # Remove content inside various bracket types (including brackets)
            text = re.sub(r'[（(][^）)]*[）)]', ' ', text)
            text = re.sub(r'[「][^」]*[」]', ' ', text)
            text = re.sub(r'[『][^』]*[』]', ' ', text)
            text = re.sub(r'[【][^】]*[】]', ' ', text)
            # Remove everything after a colon (often a subtitle/detail)
            text = re.sub(r'[：:].*', '', text)

        # ---- Level 2: extract keywords only ----
        if level >= 2:
            # Split on punctuation and whitespace
            tokens = re.split(r'[\s、。・／/,;，；\-–—]+', text)
            tokens = [t.strip() for t in tokens if t.strip()]

            # Japanese / English stop-words
            stop_words = {
                # Japanese particles & connectors
                'の', 'を', 'に', 'は', 'が', 'と', 'で', 'から', 'まで', 'より',
                'へ', 'も', 'や', 'な', 'だ', 'です', 'ます', 'する', 'ある',
                'について', 'に関する', 'における', 'および', 'ならびに',
                'または', 'もしくは', 'および', 'ただし',
                'その', 'この', 'あの', 'どの', 'それ', 'これ', 'あれ',
                'よる', 'よって', 'おける', 'つい', 'ため',
                # Common filler words in queries
                'まとめ', '一覧', '概要', '詳細', '比較', '解説', '紹介',
                '主要', '主な', '各種', '基本', '基礎',
                # English stop-words
                'the', 'a', 'an', 'of', 'in', 'to', 'and', 'for', 'on',
                'about', 'with', 'by', 'from', 'as', 'at', 'or', 'but',
                'is', 'are', 'was', 'were', 'be', 'been',
                'summary', 'overview', 'comparison', 'introduction',
            }

            tokens = [t for t in tokens if t.lower() not in stop_words and len(t) > 1]
            text = ' '.join(tokens)

        # ---- Level 3: keep at most 3 keywords ----
        if level >= 3:
            tokens = text.split()
            tokens = tokens[:3]
            text = ' '.join(tokens)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Never return empty — fall back to original
        return text if text else query

    def _clean_text(self, text: str) -> str:
        """Clean extracted text content."""
        if not text:
            return ""

        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        if not url:
            return False

        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg')
        url_lower = url.lower()

        # Check file extension
        if any(url_lower.endswith(ext) for ext in valid_extensions):
            return True

        # Check for image-related URL patterns
        if any(pattern in url_lower for pattern in ['image', 'img', 'photo', 'picture']):
            return True

        return False

    def save_image(
        self,
        url: str,
        save_dir: Path,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download and save an image.

        Args:
            url: Image URL
            save_dir: Directory to save the image
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved image or None if failed
        """
        import requests
        from urllib.parse import urlparse

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Determine filename
            if not filename:
                parsed = urlparse(url)
                filename = Path(parsed.path).name or "image.jpg"

            # Ensure save directory exists
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save image
            save_path = save_dir / filename
            with open(save_path, "wb") as f:
                f.write(response.content)

            return save_path

        except Exception:
            return None
