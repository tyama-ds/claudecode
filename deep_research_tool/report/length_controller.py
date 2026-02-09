"""
Content Length Controller - Control output length by pages or characters.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re


@dataclass
class LengthTarget:
    """Target length specification."""
    target_pages: Optional[int] = None
    target_characters: Optional[int] = None
    chars_per_page: int = 2500  # Default characters per page (A4, standard font)

    def get_target_characters(self) -> Optional[int]:
        """Get target character count."""
        if self.target_characters:
            return self.target_characters
        if self.target_pages:
            return self.target_pages * self.chars_per_page
        return None

    def has_target(self) -> bool:
        """Check if any target is set."""
        return self.target_pages is not None or self.target_characters is not None


@dataclass
class LengthInfo:
    """Information about content length."""
    total_characters: int
    estimated_pages: float
    section_lengths: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_characters": self.total_characters,
            "estimated_pages": round(self.estimated_pages, 1),
            "section_lengths": self.section_lengths,
        }


@dataclass
class ExpansionRequirement:
    """Requirements for content expansion."""
    needs_expansion: bool
    target_characters: int
    current_characters: int
    expansion_ratio: float
    sections_to_expand: List[str]
    characters_needed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "needs_expansion": self.needs_expansion,
            "target_characters": self.target_characters,
            "current_characters": self.current_characters,
            "expansion_ratio": round(self.expansion_ratio, 2),
            "sections_to_expand": self.sections_to_expand,
            "characters_needed": self.characters_needed,
        }


class ContentLengthController:
    """
    Control content length to meet page or character targets.

    Provides methods to:
    - Calculate current content length
    - Estimate page count
    - Adjust content to meet targets
    """

    # Default characters per page for different formats
    CHARS_PER_PAGE = {
        "pdf": 2500,      # A4, 11pt font
        "docx": 2800,     # Word document, standard margins
        "markdown": 3000, # Plain text/markdown
        "html": 2500,     # HTML rendered
    }

    # Minimum content ratios (don't shrink beyond these)
    MIN_SECTION_RATIO = 0.2       # Minimum 20% of original
    MIN_EXEC_SUMMARY_RATIO = 0.5  # Keep at least 50% of exec summary

    def __init__(
        self,
        target: LengthTarget = None,
        format_type: str = "pdf",
        language: str = "ja",
    ):
        """
        Initialize ContentLengthController.

        Args:
            target: Length target specification
            format_type: Output format type
            language: Content language
        """
        self.target = target or LengthTarget()
        self.format_type = format_type
        self.language = language

        # Set chars per page based on format
        if self.target.chars_per_page == 2500:  # Default value
            self.target.chars_per_page = self.CHARS_PER_PAGE.get(format_type, 2500)

    def calculate_length(
        self,
        section_contents: Dict[str, Dict[str, Any]],
    ) -> LengthInfo:
        """
        Calculate current content length.

        Args:
            section_contents: Dictionary of section contents

        Returns:
            LengthInfo with length statistics
        """
        section_lengths = {}
        total = 0

        for section_id, content in section_contents.items():
            if isinstance(content, dict):
                text = content.get("content", "") or ""
                # Add summary if present
                summary = content.get("summary", "") or ""
                length = len(text) + len(summary)
            else:
                length = len(str(content))

            section_lengths[section_id] = length
            total += length

        estimated_pages = total / self.target.chars_per_page

        return LengthInfo(
            total_characters=total,
            estimated_pages=estimated_pages,
            section_lengths=section_lengths,
        )

    def needs_adjustment(
        self,
        section_contents: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, float]:
        """
        Check if content needs adjustment.

        Args:
            section_contents: Dictionary of section contents

        Returns:
            Tuple of (needs_adjustment, adjustment_ratio)
            - adjustment_ratio > 1 means expand
            - adjustment_ratio < 1 means shrink
        """
        if not self.target.has_target():
            return False, 1.0

        length_info = self.calculate_length(section_contents)
        target_chars = self.target.get_target_characters()

        if target_chars is None:
            return False, 1.0

        ratio = target_chars / max(length_info.total_characters, 1)

        # Allow 10% tolerance
        if 0.9 <= ratio <= 1.1:
            return False, 1.0

        return True, ratio

    def adjust_content(
        self,
        section_contents: Dict[str, Dict[str, Any]],
        adjustment_ratio: float = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Adjust content to meet target length.

        Args:
            section_contents: Dictionary of section contents
            adjustment_ratio: Ratio to adjust by (calculated if not provided)

        Returns:
            Adjusted section contents
        """
        if not self.target.has_target():
            return section_contents

        if adjustment_ratio is None:
            _, adjustment_ratio = self.needs_adjustment(section_contents)

        # No adjustment needed
        if 0.9 <= adjustment_ratio <= 1.1:
            return section_contents

        adjusted = {}

        for section_id, content in section_contents.items():
            if not isinstance(content, dict):
                adjusted[section_id] = content
                continue

            # Determine min ratio based on section type
            if section_id == "_executive_summary":
                min_ratio = self.MIN_EXEC_SUMMARY_RATIO
            else:
                min_ratio = self.MIN_SECTION_RATIO

            # Apply ratio with minimum limit
            effective_ratio = max(adjustment_ratio, min_ratio)

            adjusted[section_id] = self._adjust_section(content, effective_ratio)

        return adjusted

    def _adjust_section(
        self,
        content: Dict[str, Any],
        ratio: float,
    ) -> Dict[str, Any]:
        """
        Adjust a single section's content.

        Args:
            content: Section content dictionary
            ratio: Adjustment ratio

        Returns:
            Adjusted section content
        """
        adjusted = content.copy()

        if ratio < 1.0:
            # Shrink content
            if "content" in adjusted and adjusted["content"]:
                adjusted["content"] = self._shrink_text(
                    adjusted["content"],
                    ratio,
                )

            # Limit key findings/points
            if "key_findings" in adjusted:
                max_items = max(2, int(len(adjusted["key_findings"]) * ratio))
                adjusted["key_findings"] = adjusted["key_findings"][:max_items]

            if "key_points" in adjusted:
                max_items = max(2, int(len(adjusted["key_points"]) * ratio))
                adjusted["key_points"] = adjusted["key_points"][:max_items]

        elif ratio > 1.0:
            # Note: Expanding content would require LLM
            # For now, we just return as-is and note that expansion is limited
            pass

        return adjusted

    def _shrink_text(self, text: str, ratio: float) -> str:
        """
        Shrink text to meet ratio.

        Uses intelligent truncation:
        1. Keep first and last paragraphs
        2. Trim middle paragraphs
        3. Preserve sentence boundaries

        Args:
            text: Text to shrink
            ratio: Target ratio (0 < ratio < 1)

        Returns:
            Shrunk text
        """
        if not text or ratio >= 1.0:
            return text

        target_length = int(len(text) * ratio)

        if len(text) <= target_length:
            return text

        # Split into paragraphs
        paragraphs = text.split('\n\n')

        if len(paragraphs) <= 2:
            # Few paragraphs - truncate at sentence boundary
            return self._truncate_at_sentence(text, target_length)

        # Keep first paragraph fully
        result_parts = [paragraphs[0]]
        current_length = len(paragraphs[0])

        # Calculate remaining budget for middle and end
        remaining_budget = target_length - current_length - len(paragraphs[-1])

        if remaining_budget > 0:
            # Add middle paragraphs until budget exhausted
            middle_paras = paragraphs[1:-1]
            for para in middle_paras:
                if current_length + len(para) + 4 <= target_length - len(paragraphs[-1]):
                    result_parts.append(para)
                    current_length += len(para) + 4  # +4 for \n\n
                else:
                    # Truncate this paragraph
                    remaining = target_length - current_length - len(paragraphs[-1]) - 8
                    if remaining > 50:
                        truncated = self._truncate_at_sentence(para, remaining)
                        if truncated:
                            result_parts.append(truncated)
                    break

        # Add last paragraph
        if len(paragraphs) > 1:
            result_parts.append(paragraphs[-1])

        return '\n\n'.join(result_parts)

    def _truncate_at_sentence(self, text: str, max_length: int) -> str:
        """
        Truncate text at the nearest sentence boundary.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text ending at a sentence
        """
        if len(text) <= max_length:
            return text

        # Find sentence boundaries (。.!? followed by space or end)
        sentence_ends = []
        for match in re.finditer(r'[。.!?！？]\s*', text[:max_length]):
            sentence_ends.append(match.end())

        if sentence_ends:
            # Use the last complete sentence
            return text[:sentence_ends[-1]].strip()

        # No sentence boundary found - truncate at word boundary
        truncated = text[:max_length]

        # Find last space
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.5:
            return truncated[:last_space].strip() + "..."

        # For Japanese text, just truncate
        return truncated.strip() + "..."

    def get_adjustment_summary(
        self,
        original_length: LengthInfo,
        adjusted_length: LengthInfo,
    ) -> Dict[str, Any]:
        """
        Get summary of adjustments made.

        Args:
            original_length: Original length info
            adjusted_length: Adjusted length info

        Returns:
            Summary dictionary
        """
        return {
            "original_characters": original_length.total_characters,
            "adjusted_characters": adjusted_length.total_characters,
            "original_pages": original_length.estimated_pages,
            "adjusted_pages": adjusted_length.estimated_pages,
            "reduction_percentage": round(
                (1 - adjusted_length.total_characters / max(original_length.total_characters, 1)) * 100,
                1
            ),
            "target_met": self._check_target_met(adjusted_length),
        }

    def _check_target_met(self, length_info: LengthInfo) -> bool:
        """Check if target is met (within 10% tolerance)."""
        target_chars = self.target.get_target_characters()
        if target_chars is None:
            return True

        ratio = length_info.total_characters / target_chars
        return 0.9 <= ratio <= 1.1

    def get_expansion_requirement(
        self,
        section_contents: Dict[str, Dict[str, Any]],
    ) -> ExpansionRequirement:
        """
        Calculate expansion requirements.

        Args:
            section_contents: Dictionary of section contents

        Returns:
            ExpansionRequirement with expansion details
        """
        if not self.target.has_target():
            return ExpansionRequirement(
                needs_expansion=False,
                target_characters=0,
                current_characters=0,
                expansion_ratio=1.0,
                sections_to_expand=[],
                characters_needed=0,
            )

        length_info = self.calculate_length(section_contents)
        target_chars = self.target.get_target_characters()

        if target_chars is None:
            return ExpansionRequirement(
                needs_expansion=False,
                target_characters=0,
                current_characters=length_info.total_characters,
                expansion_ratio=1.0,
                sections_to_expand=[],
                characters_needed=0,
            )

        ratio = target_chars / max(length_info.total_characters, 1)

        # Need expansion if content is less than 90% of target
        if ratio <= 1.1:
            return ExpansionRequirement(
                needs_expansion=False,
                target_characters=target_chars,
                current_characters=length_info.total_characters,
                expansion_ratio=ratio,
                sections_to_expand=[],
                characters_needed=0,
            )

        # Calculate how much more content is needed
        characters_needed = target_chars - length_info.total_characters

        # Identify sections that could be expanded
        # Prioritize sections with less content relative to others
        sections_to_expand = self._identify_sections_to_expand(
            section_contents,
            length_info,
            characters_needed,
        )

        return ExpansionRequirement(
            needs_expansion=True,
            target_characters=target_chars,
            current_characters=length_info.total_characters,
            expansion_ratio=ratio,
            sections_to_expand=sections_to_expand,
            characters_needed=characters_needed,
        )

    def _identify_sections_to_expand(
        self,
        section_contents: Dict[str, Dict[str, Any]],
        length_info: LengthInfo,
        characters_needed: int,
    ) -> List[str]:
        """
        Identify which sections should be expanded.

        Prioritizes:
        1. Sections with low confidence
        2. Sections with identified gaps
        3. Shorter sections relative to average

        Args:
            section_contents: Dictionary of section contents
            length_info: Current length information
            characters_needed: How many characters needed

        Returns:
            List of section IDs to expand
        """
        sections_to_expand = []
        regular_sections = {
            k: v for k, v in section_contents.items()
            if not k.startswith("_") and isinstance(v, dict)
        }

        if not regular_sections:
            return sections_to_expand

        # Calculate average section length
        section_lengths = [
            length_info.section_lengths.get(k, 0)
            for k in regular_sections.keys()
        ]
        avg_length = sum(section_lengths) / len(section_lengths) if section_lengths else 0

        # Score sections for expansion priority
        section_scores = []
        for section_id, content in regular_sections.items():
            score = 0
            current_length = length_info.section_lengths.get(section_id, 0)

            # Lower confidence = higher priority for expansion
            confidence = content.get("confidence", "medium")
            if confidence == "low":
                score += 3
            elif confidence == "medium":
                score += 1

            # Has gaps = higher priority
            gaps = content.get("gaps", [])
            if gaps:
                score += len(gaps)

            # Shorter than average = higher priority
            if current_length < avg_length * 0.8:
                score += 2
            elif current_length < avg_length:
                score += 1

            section_scores.append((section_id, score, current_length))

        # Sort by score (higher first), then by length (shorter first)
        section_scores.sort(key=lambda x: (-x[1], x[2]))

        # Select sections until we have enough potential expansion targets
        # Estimate ~500-1000 chars per additional research iteration
        chars_per_iteration = 750
        iterations_needed = max(1, characters_needed // chars_per_iteration)

        # Select top sections, but at least cover all if iterations needed > sections
        num_sections = min(
            max(iterations_needed // 2, 1),
            len(section_scores)
        )

        # Always include at least 1 section, up to 3
        num_sections = max(1, min(3, num_sections))

        sections_to_expand = [s[0] for s in section_scores[:num_sections]]

        return sections_to_expand

    def estimate_additional_iterations(
        self,
        expansion_requirement: ExpansionRequirement,
    ) -> int:
        """
        Estimate how many additional iterations are needed.

        Args:
            expansion_requirement: Expansion requirements

        Returns:
            Estimated number of additional iterations
        """
        if not expansion_requirement.needs_expansion:
            return 0

        # Estimate ~500-1000 characters per iteration per section
        chars_per_iteration = 750
        sections = max(1, len(expansion_requirement.sections_to_expand))

        iterations = expansion_requirement.characters_needed // (chars_per_iteration * sections)

        # At least 1, at most 5 additional iterations
        return max(1, min(5, iterations))


def estimate_page_count(
    content: str,
    format_type: str = "pdf",
) -> float:
    """
    Estimate page count for content.

    Args:
        content: Text content
        format_type: Output format

    Returns:
        Estimated page count
    """
    chars_per_page = ContentLengthController.CHARS_PER_PAGE.get(format_type, 2500)
    return len(content) / chars_per_page


def get_length_summary(
    section_contents: Dict[str, Dict[str, Any]],
    format_type: str = "pdf",
    language: str = "ja",
) -> str:
    """
    Get human-readable length summary.

    Args:
        section_contents: Dictionary of section contents
        format_type: Output format
        language: Output language

    Returns:
        Summary string
    """
    controller = ContentLengthController(format_type=format_type, language=language)
    length_info = controller.calculate_length(section_contents)

    if language == "ja":
        return (
            f"文字数: {length_info.total_characters:,}文字\n"
            f"推定ページ数: {length_info.estimated_pages:.1f}ページ\n"
            f"セクション数: {len(length_info.section_lengths)}"
        )
    else:
        return (
            f"Characters: {length_info.total_characters:,}\n"
            f"Estimated Pages: {length_info.estimated_pages:.1f}\n"
            f"Sections: {len(length_info.section_lengths)}"
        )
