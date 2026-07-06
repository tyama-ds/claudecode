"""Patent search clients for multiple patent databases."""

from .patent_search import PatentSearchClient
from .google_patents import GooglePatentsClient
from .jplatpat import JPlatPatClient
from .espacenet import EspacenetClient
from .patent_merger import PatentMerger
from .academic_search import AcademicSearchClient
from .examination_search import ExaminationSearchClient
from .business_search import BusinessSearchClient


def __getattr__(name):
    """Lazy import for SearchOrchestrator to avoid circular imports."""
    if name == "SearchOrchestrator":
        from .search_orchestrator import SearchOrchestrator
        return SearchOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PatentSearchClient",
    "GooglePatentsClient",
    "JPlatPatClient",
    "EspacenetClient",
    "PatentMerger",
    "AcademicSearchClient",
    "ExaminationSearchClient",
    "BusinessSearchClient",
    "SearchOrchestrator",
]
