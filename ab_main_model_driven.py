#!/usr/bin/env python3
import argparse
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.evaluation import compute_exact_match_metrics, compute_tool_coverage
from agent.model_driven_core import ModelDrivenToolAgent
from agent.llm_client import LLMClient
from agent.loader import find_task_files, load_task
from agent.utils import logger
from config.settings import LLM_MODEL, LLM_TEMPERATURE, TASKS_DIR, TOOLS_DIR


def _load_tasks_from_cli(tasks_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    all_tasks: List[Dict[str, Any]] = []
    for task_file in find_task_files(tasks_dir, pattern):
        loaded_tasks = load_task(task_file)
        for task in loaded_tasks:
            task.setdefault("_meta", {})["tasks_root"] = str(tasks_dir.resolve())
        all_tasks.extend(loaded_tasks)
    return all_tasks


def _execute_single_task(task: Dict[str, Any], tools_dir: Path, max_rounds: int) -> Dict[str, Any]:
    llm = LLMClient(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = ModelDrivenToolAgent(tools_dir=tools_dir, llm=llm, max_rounds=max_rounds)
    return agent.execute_task(task)


def _run_batch_parallel(tasks: List[Dict[str, Any]], tools_dir: Path, max_rounds: int, max_workers: int) -> Dict[str, Any]:
    logger.info(f"📦 Running batch(model-driven) of {len(tasks)} tasks with max_workers={max_workers}")
    started = time.time()
    indexed_tasks = list(enumerate(tasks))
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(indexed_tasks)
    success_count = 0

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_index: Dict[Any, int] = {
            executor.submit(_execute_single_task, task, tools_dir, max_rounds): index
            for index, task in indexed_tasks
        }
        for completed, future in enumerate(as_completed(future_to_index), 1):
            index = future_to_index[future]
            result = future.result()
            ordered_results[index] = result
            if result.get("success"):
                success_count += 1
            logger.info(f"[{completed}/{len(indexed_tasks)}] Completed")

    results = [item for item in ordered_results if item is not None]
    return {
        "total": len(tasks),
        "success": success_count,
        "failed": len(tasks) - success_count,
        "success_rate": f"{success_count/len(tasks)*100:.1f}%" if tasks else "0%",
        "results": results,
        "duration_sec_total": round(time.time() - started, 3),
    }


def _parse_success_rate(rate_text: str) -> float:
    try:
        return float(str(rate_text).replace("%", "").strip())
    except Exception:
        return 0.0


def _build_task_key(task: Dict[str, Any]) -> Optional[str]:
    rel = str(task.get("_meta", {}).get("relative_path", "")).strip()
    task_id = str(task.get("id", "")).strip()
    if not rel or not task_id:
        return None
    rel_path = Path(rel)
    task_type = rel_path.parts[0] if rel_path.parts else ""
    if task_type not in {"sequential", "parallel", "mixture"}:
        return None
    if len(rel_path.parts) < 3:
        return None
    main_topic = rel_path.parts[1]
    subtopic = rel_path.stem
    return f"{task_type}/{main_topic}/{subtopic}/{task_id}"


def _load_hint_catalog(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        logger.warning(f"Hint catalog not found: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Failed to load hint catalog {path}: {e}")
    return {}


def _resolve_hint_for_task(task: Dict[str, Any], hint_catalog: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    task_key = _build_task_key(task)
    if task_key and task_key in hint_catalog:
        return hint_catalog[task_key], task_key
    task_id = str(task.get("id", "")).strip()
    if task_id:
        candidates = [k for k in hint_catalog.keys() if k.endswith(f"/{task_id}")]
        if len(candidates) == 1:
            key = candidates[0]
            return hint_catalog.get(key), key
    return None, task_key


def _prepare_tasks_for_mode(
    base_tasks: List[Dict[str, Any]],
    mode: str,
    hint_catalog: Dict[str, Dict[str, Any]],
    hint_injection_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    tasks = copy.deepcopy(base_tasks)
    stats = {"total": len(tasks), "hint_attached": 0, "hint_missing": 0}
    if mode != "with_hint":
        return tasks, stats
    for task in tasks:
        task_hint, resolved_key = _resolve_hint_for_task(task, hint_catalog)
        if isinstance(task_hint, dict) and task_hint:
            meta = task.setdefault("_meta", {})
            meta["ab_hint_key"] = resolved_key
            meta["ab_hint_catalog_entry"] = task_hint
            meta["ab_hint_injection_mode"] = hint_injection_mode
            stats["hint_attached"] += 1
        else:
            stats["hint_missing"] += 1
    return tasks, stats


def _run_mode(mode: str, tasks: List[Dict[str, Any]], tools_dir: Path, max_rounds: int, max_workers: int) -> Dict[str, Any]:
    started = time.time()
    summary = _run_batch_parallel(tasks, tools_dir, max_rounds, max_workers)
    summary["exact_match"] = compute_exact_match_metrics(summary)
    summary["coverage"] = compute_tool_coverage(tasks, tools_dir)
    summary["mode"] = mode
    summary["duration_sec_total"] = round(time.time() - started, 3)
    return summary


def _result_index(results: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in results.get("results", []):
        task_id = str(item.get("task_id", ""))
        prompt = str(item.get("user_prompt", ""))
        key = (task_id, prompt.split("\n", 1)[0])
        index[key] = item
    return index


def _build_ab_comparison(no_hint: Dict[str, Any], with_hint: Dict[str, Any]) -> Dict[str, Any]:
    idx_a = _result_index(no_hint)
    idx_b = _result_index(with_hint)
    keys = sorted(set(idx_a.keys()) | set(idx_b.keys()))
    per_task = []
    newly_recovered = 0
    regressions = 0
    for key in keys:
        a = idx_a.get(key, {})
        b = idx_b.get(key, {})
        a_success = bool(a.get("success", False))
        b_success = bool(b.get("success", False))
        if (not a_success) and b_success:
            newly_recovered += 1
        if a_success and (not b_success):
            regressions += 1
        per_task.append(
            {
                "task_id": key[0],
                "prompt_head": key[1],
                "expected_answer": a.get("expected_answer") or b.get("expected_answer"),
                "no_hint": {
                    "success": a_success,
                    "final_answer": a.get("final_answer"),
                    "duration_sec": a.get("duration_sec"),
                    "error": a.get("error"),
                },
                "with_hint": {
                    "success": b_success,
                    "final_answer": b.get("final_answer"),
                    "duration_sec": b.get("duration_sec"),
                    "error": b.get("error"),
                },
            }
        )
    rate_a = _parse_success_rate(no_hint.get("success_rate", "0%"))
    rate_b = _parse_success_rate(with_hint.get("success_rate", "0%"))
    exact_a = _parse_success_rate(no_hint.get("exact_match", {}).get("exact_match_rate", "0%"))
    exact_b = _parse_success_rate(with_hint.get("exact_match", {}).get("exact_match_rate", "0%"))
    return {
        "metrics": {
            "no_hint_success_rate": no_hint.get("success_rate", "0%"),
            "with_hint_success_rate": with_hint.get("success_rate", "0%"),
            "recovery_gain_percent_points": round(rate_b - rate_a, 2),
            "exact_match_no_hint": no_hint.get("exact_match", {}).get("exact_match_rate", "0%"),
            "exact_match_with_hint": with_hint.get("exact_match", {}).get("exact_match_rate", "0%"),
            "exact_match_gain_percent_points": round(exact_b - exact_a, 2),
            "newly_recovered_tasks": newly_recovered,
            "regressions": regressions,
        },
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Model-driven A/B executor for no_hint vs with_hint")
    parser.add_argument("--tasks-dir", type=Path, default=TASKS_DIR)
    parser.add_argument("--tools-dir", type=Path, default=TOOLS_DIR)
    parser.add_argument("--pattern", type=str, default="**/*.json")
    parser.add_argument("--task-id", type=str)
    parser.add_argument("--hint-catalog", type=Path, default=Path("tools_exception/exception_hints_catalog.json"))
    parser.add_argument("--ab-output", type=Path, default=Path("tools_exception/ab_result_model_driven.json"))
    parser.add_argument("--ab-modes", type=str, default="no_hint,with_hint")
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=1, help="Max parallel task workers per mode")
    parser.add_argument(
        "--with-hint-injection-mode",
        type=str,
        default="from_start",
        choices=["from_start", "deferred_on_first_error"],
        help="How with_hint mode injects hints: from start for design AB, or deferred once after first failure for realistic testing.",
    )
    args = parser.parse_args()

    all_tasks = _load_tasks_from_cli(args.tasks_dir, args.pattern)
    if args.task_id:
        matched = [t for t in all_tasks if t.get("id") == args.task_id]
        if not matched:
            logger.error(f"Task not found: {args.task_id}")
            return 1
        if len(matched) > 1:
            logger.error(f"Task ID is ambiguous: {args.task_id}. Narrow --tasks-dir or --pattern")
            return 1
        all_tasks = [matched[0]]

    logger.info(f"Found {len(all_tasks)} tasks for model-driven A/B")
    modes = [m.strip() for m in args.ab_modes.split(",") if m.strip()]
    if not modes:
        modes = ["no_hint", "with_hint"]

    hint_catalog = _load_hint_catalog(args.hint_catalog)
    mode_outputs: Dict[str, Dict[str, Any]] = {}
    mode_prep_stats: Dict[str, Dict[str, int]] = {}
    for mode in modes:
        logger.info(f"▶ Running mode: {mode}")
        tasks_for_mode, prep_stats = _prepare_tasks_for_mode(
            all_tasks,
            mode,
            hint_catalog,
            args.with_hint_injection_mode,
        )
        mode_prep_stats[mode] = prep_stats
        mode_outputs[mode] = _run_mode(mode, tasks_for_mode, args.tools_dir, args.max_rounds, args.max_workers)

    comparison: Dict[str, Any] = {}
    if "no_hint" in mode_outputs and "with_hint" in mode_outputs:
        comparison = _build_ab_comparison(mode_outputs["no_hint"], mode_outputs["with_hint"])

    output_payload = {
        "meta": {
            "tasks_dir": str(args.tasks_dir),
            "tools_dir": str(args.tools_dir),
            "pattern": args.pattern,
            "task_id": args.task_id,
            "hint_catalog": str(args.hint_catalog),
            "with_hint_injection_mode": args.with_hint_injection_mode,
            "modes": modes,
            "max_rounds": args.max_rounds,
            "max_workers": args.max_workers,
            "started_at": int(time.time()),
            "env": {
                "FAIL_SEED": os.getenv("FAIL_SEED"),
                "INJECTION_CONFIG_JSON": bool(os.getenv("INJECTION_CONFIG_JSON")),
            },
        },
        "prep_stats": mode_prep_stats,
        "mode_results": mode_outputs,
        "comparison": comparison,
    }

    args.ab_output.parent.mkdir(parents=True, exist_ok=True)
    with args.ab_output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"💾 Model-driven A/B report saved to {args.ab_output}")
    if comparison:
        metrics = comparison.get("metrics", {})
        print(
            f"\n📊 A/B Summary: no_hint={metrics.get('no_hint_success_rate')} "
            f"vs with_hint={metrics.get('with_hint_success_rate')}, "
            f"gain={metrics.get('recovery_gain_percent_points')}pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
