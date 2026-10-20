"""Stage checkpoints of one run — ``run_state_<session>.json``.

The research loop already checkpoints every completed section; the
stages AFTER it (DeepThink, evidence review, Fermi, figures, writing,
verification) used to leave nothing behind, so a run that died in
verification had to redo everything. The state file records which
stages completed and the artifacts they produced; a resumed run skips
the completed ones and continues from the first stage that did not
finish.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# the order the stages run in DeepResearchTool.run()
STAGE_ORDER = ("research", "deep_think", "evidence_review", "fermi", "figures",
               "writing", "verification", "rendering")


class RunState:
    """Which stages of a run completed, and where their artifacts are."""

    def __init__(self, path: Optional[Path] = None, data: Optional[Dict[str, Any]] = None):
        self.path = Path(path) if path is not None else None
        self.data: Dict[str, Any] = data or {}
        self.data.setdefault("stages", {})
        self.data.setdefault("artifacts", {})
        self.resumed_from: List[str] = []    # stages skipped by a resume

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path) -> "RunState":
        """The saved state, or an empty one when the file is missing or
        unreadable (a broken state file never blocks a resume)."""
        p = Path(path)
        data: Dict[str, Any] = {}
        if p.is_file():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        return cls(p, data)

    # ------------------------------------------------------------------
    def done(self, stage: str) -> bool:
        return stage in self.data["stages"]

    def completed_stages(self) -> List[str]:
        return [s for s in STAGE_ORDER if s in self.data["stages"]]

    def artifact(self, key: str) -> Optional[str]:
        value = self.data["artifacts"].get(key)
        return str(value) if value else None

    def mark(self, stage: str, **artifacts: Any) -> None:
        """Record a completed stage (and its artifact paths) and save.
        Marking a stage drops the marks of every LATER stage: they are
        about to be (re)run on this body."""
        later = set(STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]) \
            if stage in STAGE_ORDER else set()
        for s in list(self.data["stages"]):
            if s in later:
                self.data["stages"].pop(s, None)
        self.data["stages"][stage] = {"at": datetime.now().isoformat()}
        for key, value in artifacts.items():
            if value is not None:
                self.data["artifacts"][key] = str(value)
        self.data["updated_at"] = datetime.now().isoformat()
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as e:                 # never breaks the run
            print(f"[RunState] save failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {"completed": self.completed_stages(),
                "artifacts": dict(self.data["artifacts"]),
                "resumed_from": list(self.resumed_from)}
