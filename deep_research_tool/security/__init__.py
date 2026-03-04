"""
Security module for Deep Research Tool - Prompt Injection defense.

Multi-layer defense:
- Layer 1: ContentSanitizer  - Neutralize hidden instructions in external content
- Layer 2: PromptGuard       - Mark trust boundaries in LLM prompts
- Layer 3: OutputValidator    - Schema-validate LLM outputs
- Layer 4: ExfilGuard        - Prevent data exfiltration via URLs
- Layer 5: SecurityMonitor   - Runtime anomaly detection and logging
- Layer 6: SecurityConfig    - Centralized security settings
"""

from .sanitizer import ContentSanitizer
from .prompt_guard import PromptGuard
from .output_validator import OutputValidator
from .exfil_guard import ExfilGuard
from .monitor import SecurityMonitor, SecurityEvent
from .config import SecurityConfig, SecurityLevel

__all__ = [
    "ContentSanitizer",
    "PromptGuard",
    "OutputValidator",
    "ExfilGuard",
    "SecurityMonitor",
    "SecurityEvent",
    "SecurityConfig",
    "SecurityLevel",
]

__version__ = "1.0.0"
