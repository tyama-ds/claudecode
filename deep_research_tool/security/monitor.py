"""
Layer 5: Security Monitor.

Runtime event logging, anomaly detection, and blocking for the
research pipeline.  Integrates with ResearchWarnings for user-facing
alerts.
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from pathlib import Path

from .config import SecurityConfig


logger = logging.getLogger("deep_research_tool.security")


class EventType(str, Enum):
    """Security event types."""
    URL_ACCESS = "url_access"
    PI_DETECTED = "pi_detected"
    EXFIL_BLOCKED = "exfil_blocked"
    VALIDATION_FAIL = "validation_fail"
    QUERY_GENERATED = "query_generated"
    REDIRECT = "redirect"
    CONTENT_SANITIZED = "content_sanitized"
    SESSION_LIMIT = "session_limit"


class Severity(str, Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """A single security event."""
    event_type: EventType
    severity: Severity
    message: str
    detail: str = ""
    source_url: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail[:500],
            "source_url": self.source_url[:200],
            "timestamp": self.timestamp,
        }


class SecurityMonitor:
    """
    Collects security events during a research session and provides
    anomaly detection and optional blocking.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._events: list[SecurityEvent] = []
        self._url_count = 0
        self._pi_count = 0
        self._exfil_count = 0
        self._log_handler: Optional[logging.FileHandler] = None

        if self.config.security_log_file:
            self._setup_file_logging(self.config.security_log_file)

    def record(self, event: SecurityEvent):
        """Record a security event."""
        if not self.config.enable_monitor:
            return

        self._events.append(event)

        # Update counters
        if event.event_type == EventType.URL_ACCESS:
            self._url_count += 1
        elif event.event_type == EventType.PI_DETECTED:
            self._pi_count += 1
        elif event.event_type == EventType.EXFIL_BLOCKED:
            self._exfil_count += 1

        # Log based on severity
        if event.severity == Severity.CRITICAL:
            logger.warning(
                "[SECURITY] %s: %s (detail: %s)",
                event.event_type.value, event.message, event.detail[:200],
            )
        elif event.severity == Severity.WARNING:
            logger.info(
                "[SECURITY] %s: %s",
                event.event_type.value, event.message,
            )

        # File logging if enabled
        if self._log_handler and self.config.log_all_external_access:
            self._write_to_file(event)

    def record_url_access(self, url: str, status_code: int = 0, redirected: bool = False):
        """Convenience: record a URL access event."""
        self.record(SecurityEvent(
            event_type=EventType.URL_ACCESS,
            severity=Severity.INFO,
            message=f"Accessed URL (status={status_code})",
            source_url=url,
            detail=f"redirected={redirected}",
        ))

    def record_pi_detection(self, pattern: str, source_url: str, matched_text: str):
        """Convenience: record a prompt injection detection."""
        self.record(SecurityEvent(
            event_type=EventType.PI_DETECTED,
            severity=Severity.CRITICAL,
            message=f"Prompt injection pattern detected",
            source_url=source_url,
            detail=f"pattern={pattern}, text={matched_text[:100]}",
        ))

    def record_exfil_block(self, url: str, reason: str):
        """Convenience: record a blocked exfiltration attempt."""
        self.record(SecurityEvent(
            event_type=EventType.EXFIL_BLOCKED,
            severity=Severity.CRITICAL,
            message=f"Exfiltration attempt blocked: {reason}",
            source_url=url,
        ))

    def record_content_sanitized(self, source_url: str, removed_count: int):
        """Convenience: record content sanitization."""
        if removed_count > 0:
            self.record(SecurityEvent(
                event_type=EventType.CONTENT_SANITIZED,
                severity=Severity.WARNING,
                message=f"Sanitized {removed_count} suspicious elements",
                source_url=source_url,
            ))

    def check_session_limits(self) -> tuple[bool, str]:
        """
        Check if session-level limits have been exceeded.

        Returns:
            (within_limits, message) tuple
        """
        if self._url_count > self.config.max_urls_per_session:
            return False, f"URL limit exceeded ({self._url_count}/{self.config.max_urls_per_session})"

        if self.config.block_on_pi_detection and self._pi_count > 0:
            return False, f"PI detected ({self._pi_count} occurrences), blocking enabled"

        return True, ""

    def get_summary(self) -> dict:
        """Return a summary of security events for the session."""
        return {
            "total_events": len(self._events),
            "url_accesses": self._url_count,
            "pi_detections": self._pi_count,
            "exfil_blocks": self._exfil_count,
            "critical_events": sum(
                1 for e in self._events if e.severity == Severity.CRITICAL
            ),
            "warning_events": sum(
                1 for e in self._events if e.severity == Severity.WARNING
            ),
        }

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        severity: Optional[Severity] = None,
    ) -> list[SecurityEvent]:
        """Get filtered events."""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]
        return events

    def get_warnings_for_report(self) -> list[str]:
        """
        Get security warnings formatted for inclusion in
        ResearchWarnings output.
        """
        warnings = []
        if self._pi_count > 0:
            warnings.append(
                f"[Security] {self._pi_count} prompt injection pattern(s) detected "
                f"in external content during research."
            )
        if self._exfil_count > 0:
            warnings.append(
                f"[Security] {self._exfil_count} potential data exfiltration "
                f"attempt(s) blocked."
            )
        summary = self.get_summary()
        if summary["critical_events"] > 0:
            warnings.append(
                f"[Security] {summary['critical_events']} critical security "
                f"event(s) recorded. Check security log for details."
            )
        return warnings

    def reset(self):
        """Reset all counters and events."""
        self._events.clear()
        self._url_count = 0
        self._pi_count = 0
        self._exfil_count = 0

    def _setup_file_logging(self, log_path: Path):
        """Set up file-based security logging."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [SECURITY] %(message)s"
        )
        self._log_handler.setFormatter(formatter)
        logger.addHandler(self._log_handler)

    def _write_to_file(self, event: SecurityEvent):
        """Write event to security log file."""
        try:
            logger.debug(json.dumps(event.to_dict(), ensure_ascii=False))
        except Exception:
            pass  # Don't let logging errors break the pipeline

    def __del__(self):
        """Clean up file handler."""
        if self._log_handler:
            logger.removeHandler(self._log_handler)
            self._log_handler.close()
