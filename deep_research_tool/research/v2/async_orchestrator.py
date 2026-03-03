"""
Async Research Orchestrator - Parallel section processing.

Enables concurrent research across independent sections using asyncio,
while respecting dependencies between parent/child sections and
rate limits of search APIs.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from ..query_generator import TableOfContentsItem


@dataclass
class SectionGroup:
    """A group of sections that can be processed in parallel."""

    sections: List[TableOfContentsItem] = field(default_factory=list)
    parent_level: str = ""  # e.g., "1" for sections 1.1, 1.2, 1.3


@dataclass
class ParallelResearchResult:
    """Result of parallel section processing."""

    section_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_time: float = 0.0
    parallel_groups: int = 0
    errors: Dict[str, str] = field(default_factory=dict)


class AsyncResearchOrchestrator:
    """
    Orchestrator for parallel section research.

    Analyzes section dependencies and processes independent sections
    concurrently using asyncio.to_thread for compatibility with the
    existing synchronous Researcher API.
    """

    def __init__(
        self,
        max_concurrent_sections: int = 3,
        progress_callback: Callable[[str, float], None] = None,
    ):
        self.max_concurrent = max_concurrent_sections
        self.progress_callback = progress_callback

    def analyze_dependencies(
        self,
        sections: List[TableOfContentsItem],
    ) -> List[SectionGroup]:
        """
        Analyze section dependencies and group into parallel-executable batches.

        Sections at the same level under the same parent can run in parallel.
        Example: [1.1, 1.2, 1.3] can run in parallel, but 1 must finish
        before 1.1 starts (if 1 has content).

        Returns:
            List of SectionGroup, where each group can be processed in parallel.
        """
        if not sections:
            return []

        # Group by parent section number
        groups_by_parent: Dict[str, List[TableOfContentsItem]] = {}
        for section in sections:
            parent = self._get_parent_key(section.section)
            if parent not in groups_by_parent:
                groups_by_parent[parent] = []
            groups_by_parent[parent].append(section)

        # Convert to ordered SectionGroups
        # Process top-level sections first, then their children
        result = []
        processed_parents = set()

        for section in sections:
            parent = self._get_parent_key(section.section)
            if parent in processed_parents:
                continue
            processed_parents.add(parent)

            group = SectionGroup(
                sections=groups_by_parent[parent],
                parent_level=parent,
            )
            result.append(group)

        return result

    async def process_sections_parallel(
        self,
        section_groups: List[SectionGroup],
        process_func: Callable,
        available_queries: List[str],
        total_sections: int,
    ) -> ParallelResearchResult:
        """
        Process section groups, running sections within each group in parallel.

        Args:
            section_groups: Groups of sections from analyze_dependencies()
            process_func: The synchronous function to process a single section.
                          Signature: process_func(section, queries, section_idx, total) -> None
            available_queries: Initial search queries from research plan.
            total_sections: Total number of sections (for progress tracking).

        Returns:
            ParallelResearchResult with timing and error information.
        """
        result = ParallelResearchResult()
        start_time = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        section_counter = 0

        for group_idx, group in enumerate(section_groups):
            result.parallel_groups += 1

            if len(group.sections) == 1:
                # Single section: run directly (no overhead)
                section = group.sections[0]
                section_counter += 1
                queries = self._get_queries_for_section(
                    available_queries, section_counter
                )
                try:
                    await asyncio.to_thread(
                        process_func, section, queries,
                        section_counter - 1, total_sections,
                    )
                except Exception as e:
                    result.errors[section.section] = str(e)
                    print(f"[AsyncOrch] Error in section {section.section}: {e}")
            else:
                # Multiple sections: run in parallel with semaphore
                tasks = []
                for section in group.sections:
                    section_counter += 1
                    queries = self._get_queries_for_section(
                        available_queries, section_counter
                    )
                    task = self._run_with_semaphore(
                        semaphore,
                        process_func,
                        section, queries,
                        section_counter - 1, total_sections,
                    )
                    tasks.append((section.section, task))

                # Gather results
                gathered = await asyncio.gather(
                    *[t for _, t in tasks],
                    return_exceptions=True,
                )
                for (section_id, _), task_result in zip(tasks, gathered):
                    if isinstance(task_result, Exception):
                        result.errors[section_id] = str(task_result)
                        print(f"[AsyncOrch] Error in section {section_id}: {task_result}")

                if self.progress_callback:
                    progress = (section_counter / total_sections) * 70 + 10
                    self.progress_callback(
                        f"Group {group_idx + 1}/{len(section_groups)} complete",
                        progress,
                    )

        result.total_time = time.time() - start_time
        return result

    async def _run_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        process_func: Callable,
        section: TableOfContentsItem,
        queries: List[str],
        section_idx: int,
        total_sections: int,
    ):
        """Run a section processing function with semaphore-based concurrency control."""
        async with semaphore:
            return await asyncio.to_thread(
                process_func, section, queries,
                section_idx, total_sections,
            )

    def _get_parent_key(self, section_number: str) -> str:
        """Get the parent level key for grouping.

        "1" -> ""  (top level)
        "1.1" -> "1"
        "1.1.1" -> "1.1"
        """
        parts = section_number.split(".")
        if len(parts) <= 1:
            return ""
        return ".".join(parts[:-1])

    def _get_queries_for_section(
        self,
        available_queries: List[str],
        section_counter: int,
        queries_per_section: int = 3,
    ) -> List[str]:
        """Get a slice of available queries for a specific section."""
        start = (section_counter - 1) * queries_per_section
        end = start + queries_per_section
        return available_queries[start:end] if start < len(available_queries) else []
