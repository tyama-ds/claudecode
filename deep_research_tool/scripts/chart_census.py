"""
Count how often past research runs had data in the shape each proposed
chart form needs — without any LLM call or network access.

It reads what the auto-chart step left in each run's figures folder:
``chart_data_<chapter>.json`` (the per-chapter data catalog: label arrays
and series with values, units and kinds) and ``chart_yield.json`` (which
forms were drawn). Nothing is written.

    python -m deep_research_tool.scripts.chart_census ./output
    python -m deep_research_tool.scripts.chart_census ./output --json

Shapes counted (candidates only: the real gates come with each form):
- donut      a comparison in % whose values add up to 100 (±1.5) over 3-6 items
- donut_part a comparison in % adding up to 50-98.5 (would need a gap)
- ridge      3-8 time series of one metric and unit (entities side by side)
- slope      a time series with exactly 2 points (dropped today: < 3 points)
- quarterly  a time series whose labels are quarters / halves / months
- scatter    two comparisons of the same year with different metrics or
             units sharing >= 4 entities
- heatmap    >= 4 time series of one metric and unit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_SUBYEAR = re.compile(r"(Q[1-4]|H[12]|上期|下期|上半期|下半期|[0-9]{4}[-/][01]?[0-9]$|月)", re.I)
SHAPES = ("donut", "donut_part", "ridge", "slope", "quarterly", "scatter", "heatmap")


def _numbers(values: Iterable[Any]) -> List[float]:
    out = []
    for v in values or []:
        if isinstance(v, dict):
            v = v.get("y")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out


def _metric_of(series: Dict[str, Any]) -> str:
    """The series name without its entity (time series are named
    '<entity> <metric>')."""
    name = str(series.get("name") or "")
    ent = str(series.get("entity") or "")
    if ent and name.startswith(ent):
        name = name[len(ent):]
    return name.strip()


def census_chapter(data: Dict[str, Any]) -> Dict[str, int]:
    """Candidate counts for one chapter's catalog (chart_data_*.json)."""
    counts = dict.fromkeys(SHAPES, 0)
    series = {k: v for k, v in data.items() if isinstance(v, dict) and "values" in v}
    labels = {k: v for k, v in data.items() if isinstance(v, list)}
    time_groups: Dict[tuple, int] = {}
    comparisons = []
    for s in series.values():
        nums = _numbers(s.get("values"))
        labs = labels.get(s.get("labels_ref"), [])
        unit = str(s.get("unit") or "")
        if s.get("kind") == "time":
            if len(nums) == 2:
                counts["slope"] += 1
            if any(_SUBYEAR.search(str(l)) for l in labs):
                counts["quarterly"] += 1
            key = (_metric_of(s), unit)
            time_groups[key] = time_groups.get(key, 0) + 1
        else:
            comparisons.append((s, labs, nums, unit))
            if unit.strip() in ("%", "％") and 3 <= len(nums) <= 6 and all(v >= 0 for v in nums):
                total = sum(nums)
                if abs(total - 100.0) <= 1.5:
                    counts["donut"] += 1
                elif 50.0 <= total < 98.5:
                    counts["donut_part"] += 1
    for n in time_groups.values():
        if 3 <= n <= 8:
            counts["ridge"] += 1
        if n >= 4:
            counts["heatmap"] += 1
    for i, (a, la, _na, ua) in enumerate(comparisons):
        for b, lb, _nb, ub in comparisons[i + 1:]:
            if a.get("year") is None or a.get("year") != b.get("year"):
                continue
            if a.get("name") == b.get("name") and ua == ub:
                continue
            if len(set(la) & set(lb)) >= 4:
                counts["scatter"] += 1
    return counts


def census_run(figures_dir: Path) -> Dict[str, Any]:
    """Counts for one run's figures folder."""
    shapes = dict.fromkeys(SHAPES, 0)
    chapters = 0
    for f in sorted(figures_dir.glob("chart_data_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        chapters += 1
        for k, v in census_chapter(data).items():
            shapes[k] += v
    drawn: Dict[str, int] = {}
    try:
        yld = json.loads((figures_dir / "chart_yield.json").read_text(encoding="utf-8"))
        for ch in (yld.get("chapters") or {}).values():
            for p in ch.get("panels") or []:
                if p.get("rendered"):
                    form = p.get("form_rendered") or p.get("form") or "?"
                    drawn[form] = drawn.get(form, 0) + 1
    except Exception:
        pass
    return {"figures_dir": str(figures_dir), "chapters": chapters, "candidates": shapes, "drawn": drawn}


def find_runs(root: Path) -> List[Path]:
    """Every folder under ``root`` that holds chart_data_*.json."""
    return sorted({p.parent for p in Path(root).rglob("chart_data_*.json")})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", nargs="?", default="./output", help="出力フォルダ（既定: ./output）")
    ap.add_argument("--json", action="store_true", help="JSON で出力する")
    args = ap.parse_args(argv)
    runs = [census_run(d) for d in find_runs(Path(args.root))]
    total = dict.fromkeys(SHAPES, 0)
    drawn_total: Dict[str, int] = {}
    for r in runs:
        for k, v in r["candidates"].items():
            total[k] += v
        for k, v in r["drawn"].items():
            drawn_total[k] = drawn_total.get(k, 0) + v
    if args.json:
        print(json.dumps({"runs": runs, "total": total, "drawn_total": drawn_total},
                         ensure_ascii=False, indent=1))
        return 0
    if not runs:
        print(f"{args.root} に chart_data_*.json が見つかりません（図表を作った調査の出力フォルダを指定してください）")
        return 1
    print(f"調査 {len(runs)} 件・章 {sum(r['chapters'] for r in runs)} 件")
    print("候補の数（形ごと）: " + "、".join(f"{k} {v}" for k, v in total.items()))
    if drawn_total:
        print("実際に描いた形: " + "、".join(f"{k} {v}" for k, v in sorted(drawn_total.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
