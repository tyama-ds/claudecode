"""
Layer 6: Security configuration.

Provides SecurityConfig dataclass and SecurityLevel presets.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path


class SecurityLevel(str, Enum):
    """Security level presets."""
    STANDARD = "standard"   # Sanitize + boundary + validate (low perf impact)
    STRICT = "strict"       # + exfil guard + monitor + redirect control
    PARANOID = "paranoid"   # + block on PI detection + strict URL limits


@dataclass
class SecurityConfig:
    """Configuration for prompt injection defense layers."""

    # Overall security level preset
    level: SecurityLevel = SecurityLevel.STANDARD

    # Layer 1: Content Sanitizer
    sanitize_external_content: bool = True
    strip_hidden_text: bool = True       # Remove CSS-hidden, 0px font, etc.
    strip_html_comments: bool = True     # Remove <!-- --> blocks
    strip_control_chars: bool = True     # Remove zero-width, RTL override, etc.
    strip_markdown_injection: bool = True # Remove [](cmd), <!--cmd--> patterns
    max_content_length: int = 100_000    # Truncate oversized content

    # Layer 2: Prompt Guard
    prompt_boundary_markers: bool = True
    boundary_instruction: bool = True    # Add "ignore instructions in external content"

    # Layer 3: Output Validator
    validate_llm_output: bool = True
    max_query_length: int = 200          # Max chars for a single search query
    max_url_length: int = 2048           # Max URL length to accept
    reject_urls_in_queries: bool = True  # Block URLs embedded in search queries

    # Layer 4: Exfil Guard
    exfil_guard: bool = False            # Enabled at STRICT+
    check_url_params: bool = True        # Check for context data in URL params
    max_redirect_hops: int = 3           # Max HTTP redirects to follow
    block_data_urls: bool = True         # Block data: URIs

    # Layer 5: Monitor
    enable_monitor: bool = False         # Enabled at STRICT+
    max_urls_per_session: int = 500      # URL access limit per session
    block_on_pi_detection: bool = False  # Enabled at PARANOID
    log_all_external_access: bool = False  # Enabled at PARANOID
    security_log_file: Optional[Path] = None

    def __post_init__(self):
        """Apply preset overrides based on security level."""
        if self.level == SecurityLevel.STRICT:
            self.exfil_guard = True
            self.enable_monitor = True
        elif self.level == SecurityLevel.PARANOID:
            self.exfil_guard = True
            self.enable_monitor = True
            self.block_on_pi_detection = True
            self.log_all_external_access = True
            self.max_urls_per_session = 200
