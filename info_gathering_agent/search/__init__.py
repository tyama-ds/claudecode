"""
Web search modules for Deep Research Tool.
"""

from .base import BaseSearchClient, SearchResult
from .duckduckgo import DuckDuckGoSearch
from .selenium_browser import SeleniumBrowser

__all__ = [
    "BaseSearchClient",
    "SearchResult",
    "DuckDuckGoSearch",
    "SeleniumBrowser",
]


def get_search_client(method: str = "duckduckgo", **kwargs):
    """
    Factory function to get the appropriate search client.

    Args:
        method: 'duckduckgo' or 'selenium'
        **kwargs: Additional configuration options

    Returns:
        Configured search client instance
    """
    if method.lower() == "duckduckgo":
        return DuckDuckGoSearch(**kwargs)
    elif method.lower() == "selenium":
        return SeleniumBrowser(**kwargs)
    else:
        raise ValueError(f"Unsupported search method: {method}")
