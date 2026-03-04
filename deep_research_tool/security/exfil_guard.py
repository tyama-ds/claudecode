"""
Layer 4: Exfiltration Guard.

Prevents data exfiltration by checking that URLs accessed during
research do not contain user query data, session information,
or other sensitive context in their parameters.

Key principle from OpenAI: "Don't let the model construct arbitrary
URLs" and "Don't encode conversation data into URL parameters."
"""

import re
from typing import Optional
from urllib.parse import urlparse, parse_qs

from .config import SecurityConfig


class ExfilGuard:
    """
    Checks outbound URLs for signs of data exfiltration.

    Exfiltration attacks work by tricking the LLM into embedding
    sensitive data (user queries, API keys, internal content) into
    URL query parameters, then fetching those URLs to send data
    to an attacker-controlled server.
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        context_tokens: Optional[list[str]] = None,
    ):
        """
        Args:
            config: Security configuration
            context_tokens: Sensitive strings to watch for in URLs
                           (e.g., user query terms, API key prefixes)
        """
        self.config = config or SecurityConfig()
        # Lowercase tokens for case-insensitive matching
        self._context_tokens = [
            t.lower() for t in (context_tokens or []) if len(t) >= 4
        ]
        self._blocked_urls: list[dict] = []
        self._url_count = 0

    def check_url(self, url: str) -> tuple[bool, str]:
        """
        Check if a URL is safe to fetch.

        Args:
            url: URL to check

        Returns:
            (is_safe, reason) tuple. is_safe=True means OK to fetch.
        """
        if not self.config.exfil_guard:
            return True, ""

        self._url_count += 1

        # Session URL limit
        if self._url_count > self.config.max_urls_per_session:
            reason = f"URL limit exceeded ({self._url_count}/{self.config.max_urls_per_session})"
            self._record_block(url, reason)
            return False, reason

        try:
            parsed = urlparse(url)
        except Exception:
            self._record_block(url, "malformed_url")
            return False, "Malformed URL"

        # Block data: URIs
        if self.config.block_data_urls and parsed.scheme == "data":
            self._record_block(url, "data_uri")
            return False, "Data URI blocked"

        # Check query parameters for context data leakage
        if self.config.check_url_params and parsed.query:
            params = parse_qs(parsed.query)
            for key, values in params.items():
                for value in values:
                    if self._contains_context_data(value):
                        reason = f"Context data detected in URL param '{key}'"
                        self._record_block(url, reason)
                        return False, reason

        # Check for suspiciously long query strings (data smuggling)
        if parsed.query and len(parsed.query) > 500:
            reason = f"Suspiciously long query string ({len(parsed.query)} chars)"
            self._record_block(url, reason)
            return False, reason

        # Check for base64-encoded payloads in path or query
        if self._has_base64_payload(parsed.path + "?" + parsed.query if parsed.query else parsed.path):
            reason = "Possible base64-encoded payload in URL"
            self._record_block(url, reason)
            return False, reason

        return True, ""

    def update_context_tokens(self, tokens: list[str]):
        """
        Update the list of sensitive context tokens to watch for.

        Call this when the user query or session data changes.
        """
        self._context_tokens = [
            t.lower() for t in tokens if len(t) >= 4
        ]

    def get_blocked_urls(self) -> list[dict]:
        """Return list of blocked URLs with reasons."""
        return list(self._blocked_urls)

    def get_url_count(self) -> int:
        """Return total number of URLs checked this session."""
        return self._url_count

    def reset(self):
        """Reset counters and blocked list."""
        self._blocked_urls.clear()
        self._url_count = 0

    def _contains_context_data(self, value: str) -> bool:
        """Check if a URL parameter value contains context tokens."""
        if not self._context_tokens:
            return False
        value_lower = value.lower()
        for token in self._context_tokens:
            if token in value_lower:
                return True
        return False

    def _has_base64_payload(self, text: str) -> bool:
        """
        Detect base64-encoded payloads in URL components.

        Looks for suspiciously long base64-like strings that could
        be used to exfiltrate data.
        """
        # Match base64 strings 40+ chars (short enough to avoid false positives
        # on legitimate hashes, long enough to carry meaningful data)
        b64_pattern = re.compile(r"[A-Za-z0-9+/=]{40,}")
        matches = b64_pattern.findall(text)
        return any(len(m) > 60 for m in matches)

    def _record_block(self, url: str, reason: str):
        """Record a blocked URL."""
        self._blocked_urls.append({
            "url": url[:200],  # Truncate for logging
            "reason": reason,
        })
