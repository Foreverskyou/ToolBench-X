#!/usr/bin/env python3
import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.evaluation import compute_exact_match_metrics, compute_tool_coverage
from agent.model_driven_core import ModelDrivenToolAgent
from agent.llm_client import LLMClient
from agent.loader import find_task_files, load_task
from agent.utils import logger
from config.settings import LLM_MODEL, LLM_TEMPERATURE, TASKS_DIR, TOOLS_DIR


def _compute_metrics_any(results_obj):
    return compute_exact_match_metrics(results_obj)


def _execute_single_task(task, tools_dir: Path, max_rounds: int):
    llm = LLMClient(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = ModelDrivenToolAgent(tools_dir=tools_dir, llm=llm, max_rounds=max_rounds)
    return agent.execute_task(task)


def _run_batch_parallel(tasks, tools_dir: Path, max_rounds: int, max_workers: int):
    logger.info(f"📦 Running batch(model-driven) of {len(tasks)} tasks with max_workers={max_workers}")
    started = time.time()
    indexed_tasks = list(enumerate(tasks))
    ordered_results = [None] * len(indexed_tasks)
    success_count = 0

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_index = {
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
    summary = {
        "total": len(tasks),
        "success": success_count,
        "failed": len(tasks) - success_count,
        "success_rate": f"{success_count/len(tasks)*100:.1f}%" if tasks else "0%",
        "results": results,
        "duration_sec_total": round(time.time() - started, 3),
    }
    logger.info(f"📊 Batch complete(model-driven): {summary['success_rate']} success rate")
    return summary


def _load_tasks_from_cli(tasks_dir: Path, pattern: str):
    all_tasks = []
    for task_file in find_task_files(tasks_dir, pattern):
        loaded_tasks = load_task(task_file)
        for task in loaded_tasks:
            task.setdefault("_meta", {})["tasks_root"] = str(tasks_dir.resolve())
        all_tasks.extend(loaded_tasks)
    return all_tasks


def _build_task_key(task):
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


def _load_hint_catalog(path: Path):
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load hint catalog {path}: {e}")
        return {}


def _resolve_hint_for_task(task, hint_catalog):
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


def _attach_hints(tasks, hint_catalog, hint_injection_mode):
    prepared = copy.deepcopy(tasks)
    stats = {"total": len(prepared), "hint_attached": 0, "hint_missing": 0}
    if not hint_catalog:
        return prepared, stats
    for task in prepared:
        task_hint, resolved_key = _resolve_hint_for_task(task, hint_catalog)
        if isinstance(task_hint, dict) and task_hint:
            meta = task.setdefault("_meta", {})
            meta["ab_hint_key"] = resolved_key
            meta["ab_hint_catalog_entry"] = task_hint
            meta["ab_hint_injection_mode"] = hint_injection_mode
            stats["hint_attached"] += 1
        else:
            stats["hint_missing"] += 1
    return prepared, stats


def main():
    parser = argparse.ArgumentParser(description="Model-driven LLM Agent Tool Executor")
    parser.add_argument("--tasks-dir", type=Path, default=TASKS_DIR, help="Tasks directory")
    parser.add_argument("--tools-dir", type=Path, default=TOOLS_DIR, help="Tools directory")
    parser.add_argument("--pattern", type=str, default="**/*.json", help="Task file pattern")
    parser.add_argument("--task-id", type=str, help="Execute single task by ID")
    parser.add_argument("--output", type=Path, help="Save results to file")
    parser.add_argument("--max-rounds", type=int, default=30, help="Max model decision rounds per task")
    parser.add_argument("--max-workers", type=int, default=1, help="Max parallel task workers in batch mode")
    parser.add_argument("--hint-catalog", type=Path, help="Optional hint catalog for with-hint execution")
    parser.add_argument(
        "--hint-injection-mode",
        type=str,
        default="deferred_on_first_error",
        choices=["from_start", "deferred_on_first_error"],
        help="Hint injection mode when --hint-catalog is provided.",
    )
    args = parser.parse_args()

    if args.task_id:
        all_tasks = _load_tasks_from_cli(args.tasks_dir, args.pattern)
        matched_tasks = [t for t in all_tasks if t.get("id") == args.task_id]
        if not matched_tasks:
            logger.error(f"Task not found: {args.task_id}")
            return 1
        if len(matched_tasks) > 1:
            logger.error(f"Task ID is ambiguous: {args.task_id}. Narrow the --tasks-dir or --pattern.")
            return 1
        selected_tasks = [matched_tasks[0]]
        batch_mode = False
    else:
        selected_tasks = _load_tasks_from_cli(args.tasks_dir, args.pattern)
        batch_mode = True
        logger.info(f"Found {len(selected_tasks)} tasks")

    hint_catalog = _load_hint_catalog(args.hint_catalog) if args.hint_catalog else {}
    selected_tasks, hint_stats = _attach_hints(selected_tasks, hint_catalog, args.hint_injection_mode)
    coverage = compute_tool_coverage(selected_tasks, args.tools_dir)

    if not batch_mode:
        result = _execute_single_task(selected_tasks[0], args.tools_dir, args.max_rounds)
        results = [result]
    else:
        results = _run_batch_parallel(selected_tasks, args.tools_dir, args.max_rounds, args.max_workers)

    metrics = _compute_metrics_any(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            payload = {
                "meta": {
                    "tasks_dir": str(args.tasks_dir),
                    "tools_dir": str(args.tools_dir),
                    "pattern": args.pattern,
                    "task_id": args.task_id,
                    "hint_catalog": str(args.hint_catalog) if args.hint_catalog else None,
                    "hint_injection_mode": args.hint_injection_mode if args.hint_catalog else None,
                    "max_workers": args.max_workers if batch_mode else 1,
                },
                "hint_stats": hint_stats,
                "coverage": coverage,
                "results": results,
                "exact_match": metrics,
            }
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 Results saved to {args.output}")

    if isinstance(results, dict) and "success_rate" in results:
        print(f"\n📊 Summary: {results['success_rate']} success rate")
        print(f"🎯 Exact Match: {metrics['exact_match_rate']} ({metrics['exact_matches']}/{metrics['with_expected']})")
        print(f"🧩 Tool Coverage: {coverage['coverage_rate']} ({coverage['covered_tasks']}/{coverage['tasks_considered']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
