from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Union

from .loader import discover_tool_path


def compute_exact_match_metrics(results: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    items = results.get("results", []) if isinstance(results, dict) else results
    if not isinstance(items, list):
        return {"with_expected": 0, "exact_matches": 0, "exact_match_rate": "0%", "mismatches": []}

    with_expected = 0
    exact_matches = 0
    mismatches: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        expected = item.get("expected_answer")
        if expected is None:
            continue
        with_expected += 1
        expected_text = str(expected).strip()
        actual_text = str(item.get("final_answer", "")).strip()
        if actual_text == expected_text:
            exact_matches += 1
        else:
            mismatches.append(
                {
                    "task_id": item.get("task_id"),
                    "expected_answer": expected_text,
                    "final_answer": actual_text,
                }
            )

    return {
        "with_expected": with_expected,
        "exact_matches": exact_matches,
        "exact_match_rate": f"{(exact_matches / with_expected * 100):.1f}%" if with_expected else "0%",
        "mismatches": mismatches,
    }


def compute_tool_coverage(tasks: Iterable[Dict[str, Any]], tools_dir: Path) -> Dict[str, Any]:
    covered = 0
    missing: List[Dict[str, str]] = []
    seen_keys: Set[str] = set()

    for task in tasks:
        if not isinstance(task, dict):
            continue
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        rel = str(meta.get("relative_path", "")).strip()
        task_id = str(task.get("id", "")).strip()
        if not rel or not task_id:
            continue
        unique_key = f"{rel}::{task_id}"
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        tool_path = discover_tool_path(tools_dir, rel, task_id)
        if tool_path is not None and tool_path.exists():
            covered += 1
        else:
            missing.append({"task_id": task_id, "relative_path": rel})

    total = covered + len(missing)
    return {
        "tools_dir": str(tools_dir),
        "tasks_considered": total,
        "covered_tasks": covered,
        "missing_tasks": len(missing),
        "coverage_rate": f"{(covered / total * 100):.1f}%" if total else "0%",
        "missing_examples": missing[:50],
    }
