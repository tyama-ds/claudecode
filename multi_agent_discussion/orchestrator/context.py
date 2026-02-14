"""Pipeline context for sharing data between stages."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class ResearchResult:
    """Result from a single agent's research."""
    agent_name: str
    perspective: str
    report_content: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    session_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "perspective": self.perspective,
            "report_content": self.report_content,
            "evidence": self.evidence,
            "session_id": self.session_id,
            "session_path": self.session_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    stage_type: str
    output: Any = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self):
        self.completed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class PipelineContext:
    """
    Shared context passed between pipeline stages.

    Each stage reads from and writes to this context, enabling
    data flow through the pipeline.
    """
    # Identity
    pipeline_id: str = field(default_factory=lambda: str(uuid4())[:8])
    topic: str = ""

    # Phase 1: Research results
    research_results: Dict[str, ResearchResult] = field(default_factory=dict)

    # Phase 2: Discussion / processing
    discussion_transcript: str = ""
    discussion_evaluation: Dict[str, Any] = field(default_factory=dict)

    # Phase 3: Synthesized / refined content
    synthesized_report: str = ""
    refined_reports: List[str] = field(default_factory=list)

    # Phase 4: Competitive evaluation
    competitive_rankings: List[Dict[str, Any]] = field(default_factory=list)
    best_report: str = ""

    # Phase 5: Fact-checking
    fact_check_results: Dict[str, Any] = field(default_factory=dict)

    # Final output
    final_report: str = ""
    final_report_path: Optional[str] = None

    # Stage history
    stage_results: List[StageResult] = field(default_factory=list)

    # General metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_research_result(self, result: ResearchResult):
        """Add a research result from an agent."""
        self.research_results[result.agent_name] = result

    def get_all_evidence(self) -> List[Dict[str, Any]]:
        """Get all evidence from all research results."""
        evidence = []
        for result in self.research_results.values():
            evidence.extend(result.evidence)
        return evidence

    def get_all_report_contents(self) -> Dict[str, str]:
        """Get all research report contents keyed by agent name."""
        return {
            name: result.report_content
            for name, result in self.research_results.items()
        }

    def get_latest_report(self) -> str:
        """Get the most recent version of the report."""
        if self.final_report:
            return self.final_report
        if self.refined_reports:
            return self.refined_reports[-1]
        if self.best_report:
            return self.best_report
        if self.synthesized_report:
            return self.synthesized_report
        # Fallback: concatenate research results
        parts = []
        for name, result in self.research_results.items():
            parts.append(f"## {name}\n\n{result.report_content}")
        return "\n\n".join(parts)

    def record_stage(self, result: StageResult):
        """Record a completed stage result."""
        self.stage_results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "topic": self.topic,
            "research_results": {
                k: v.to_dict() for k, v in self.research_results.items()
            },
            "discussion_transcript": self.discussion_transcript,
            "discussion_evaluation": self.discussion_evaluation,
            "synthesized_report": self.synthesized_report,
            "refined_reports": self.refined_reports,
            "competitive_rankings": self.competitive_rankings,
            "best_report": self.best_report,
            "fact_check_results": self.fact_check_results,
            "final_report": self.final_report,
            "final_report_path": self.final_report_path,
            "stage_results": [s.to_dict() for s in self.stage_results],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineContext":
        ctx = cls(
            pipeline_id=data.get("pipeline_id", ""),
            topic=data.get("topic", ""),
            discussion_transcript=data.get("discussion_transcript", ""),
            discussion_evaluation=data.get("discussion_evaluation", {}),
            synthesized_report=data.get("synthesized_report", ""),
            refined_reports=data.get("refined_reports", []),
            competitive_rankings=data.get("competitive_rankings", []),
            best_report=data.get("best_report", ""),
            fact_check_results=data.get("fact_check_results", {}),
            final_report=data.get("final_report", ""),
            final_report_path=data.get("final_report_path"),
            metadata=data.get("metadata", {}),
        )
        for name, rd in data.get("research_results", {}).items():
            ctx.research_results[name] = ResearchResult.from_dict(rd)
        return ctx

    def save(self, filepath: Optional[Path] = None, output_dir: str = "./pipeline_output"):
        """Save context to JSON."""
        if filepath is None:
            dir_path = Path(output_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            filepath = dir_path / f"pipeline_{self.pipeline_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "PipelineContext":
        """Load context from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
