#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from config.settings import LLM_MODEL
from evalscope_adapter import run_from_task_config


def _build_task_cfg(args: argparse.Namespace) -> Any:
    generation_config = {
        "retries": args.generation_retries,
        "retry_interval": args.retry_interval,
        "max_tokens": args.max_tokens,
        "request_timeout": args.request_timeout,
    }
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.disable_thinking:
        generation_config["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    try:
        from evalscope import TaskConfig  # type: ignore

        return TaskConfig(
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            datasets=args.datasets,
            eval_batch_size=args.max_workers,
            generation_config=generation_config,
        )
    except Exception:
        return {
            "model": args.model,
            "api_url": args.api_url,
            "api_key": args.api_key,
            "datasets": args.datasets,
            "eval_batch_size": args.max_workers,
            "generation_config": generation_config,
        }


def _mode_output_path(args: argparse.Namespace, mode: str) -> Path | None:
    if args.output:
        if mode == args.mode or args.mode != "all":
            return args.output
        return args.output.parent / f"{args.output.stem}_{mode}{args.output.suffix or '.json'}"
    if not args.output_dir:
        return None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args.output_dir / f"{mode}.json"


def _clean_output_path(args: argparse.Namespace, mode: str, output_path: Path | None) -> Path | None:
    if not args.openai_trajectory_clean_output_dir:
        return None
    args.openai_trajectory_clean_output_dir.mkdir(parents=True, exist_ok=True)
    if output_path:
        return args.openai_trajectory_clean_output_dir / f"{output_path.stem}_openai_trajectory_clean.json"
    return args.openai_trajectory_clean_output_dir / f"{mode}_openai_trajectory_clean.json"


def _maybe_log_clean_output(args: argparse.Namespace, mode: str, output_path: Path | None) -> None:
    clean_path = _clean_output_path(args, mode, output_path)
    if not clean_path:
        return
    tqdm.write(f"🧼 Clean OpenAI trajectory output written: {clean_path}")


def _extract_summary(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    meta = payload.get("meta", {})
    if mode == "ab":
        comparison = payload.get("comparison", {})
        metrics = comparison.get("metrics", {})
        return {
            "mode": mode,
            "tasks_dir": meta.get("tasks_dir"),
            "tools_dir": meta.get("tools_dir"),
            "hint_catalog": meta.get("hint_catalog"),
            "hint_injection_mode": meta.get("hint_injection_mode"),
            "max_rounds": meta.get("max_rounds"),
            "max_workers": meta.get("max_workers"),
            "no_hint_success_rate": metrics.get("no_hint_success_rate"),
            "with_hint_success_rate": metrics.get("with_hint_success_rate"),
            "recovery_gain_percent_points": metrics.get("recovery_gain_percent_points"),
            "exact_match_no_hint": metrics.get("exact_match_no_hint"),
            "exact_match_with_hint": metrics.get("exact_match_with_hint"),
            "exact_match_gain_percent_points": metrics.get("exact_match_gain_percent_points"),
            "newly_recovered_tasks": metrics.get("newly_recovered_tasks"),
            "regressions": metrics.get("regressions"),
        }
    exact = payload.get("exact_match", {})
    coverage = payload.get("coverage", {})
    results = payload.get("results", {})
    return {
        "mode": mode,
        "tasks_dir": meta.get("tasks_dir"),
        "tools_dir": meta.get("tools_dir"),
        "hint_catalog": meta.get("hint_catalog"),
        "hint_injection_mode": meta.get("hint_injection_mode"),
        "max_rounds": meta.get("max_rounds"),
        "max_workers": meta.get("max_workers"),
        "success_rate": results.get("success_rate"),
        "exact_match_rate": exact.get("exact_match_rate"),
        "coverage_rate": coverage.get("coverage_rate"),
        "hint_stats": payload.get("hint_stats"),
        "oracle_label_stats": payload.get("oracle_label_stats"),
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def _resolve_modes(mode: str) -> List[str]:
    if mode == "all":
        return ["baseline", "ab"]
    return [mode]


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal EvalScope-style runner for the local evaluation runtime")
    parser.add_argument(
        "--mode",
        choices=["baseline", "with_hint", "oracle_label", "ab", "all"],
        default="baseline",
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="**/*.json")
    parser.add_argument("--task-id", type=str)
    parser.add_argument("--include-uncovered", action="store_true", help="Disable the default covered-only filter and include tasks even if no tool resolves under --tools-dir")
    parser.add_argument("--hint-catalog", type=Path)
    parser.add_argument(
        "--hazard-manifest",
        type=Path,
        help="Balanced-task manifest containing exception_type labels for oracle_label mode",
    )
    parser.add_argument("--hint-injection-mode", choices=["from_start", "deferred_on_first_error"], default="deferred_on_first_error")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--openai-trajectory-clean-output-dir", type=Path, help="Also write a clean OpenAI trajectory JSON for each generated eval payload")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--agent-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING", help="Logging level for agent internals; use ERROR to suppress recoverable fallback warnings")
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fail-seed", type=str)
    parser.add_argument("--summary-only", action="store_true")

    parser.add_argument("--model", type=str, default=LLM_MODEL)
    parser.add_argument("--api-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--datasets", nargs="*", default=["general_fc"])
    parser.add_argument("--generation-retries", type=int, default=5)
    parser.add_argument("--retry-interval", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature. Omit to let the model/API use its default.")
    parser.add_argument("--disable-thinking", action="store_true", help="Pass extra_body.chat_template_kwargs.enable_thinking=false for Qwen-style OpenAI-compatible endpoints")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per LLM HTTP request timeout in seconds")
    args = parser.parse_args()

    if args.fail_seed:
        os.environ["FAIL_SEED"] = args.fail_seed

    # Keep progress bars clean by suppressing verbose logs from the underlying runtime.
    logging.getLogger("agent").setLevel(getattr(logging, args.agent_log_level))

    if args.mode in {"with_hint", "ab", "all"} and not args.hint_catalog:
        parser.error("--hint-catalog is required for with_hint, ab, or all mode")
    if args.mode == "oracle_label" and not args.hazard_manifest:
        parser.error("--hazard-manifest is required for oracle_label mode")

    task_cfg = _build_task_cfg(args)
    mode_payloads: Dict[str, Dict[str, Any]] = {}
    mode_summaries: Dict[str, Dict[str, Any]] = {}
    resolved_modes = _resolve_modes(args.mode)

    tqdm.write(f"🚀 Starting formal evaluation runner with mode={args.mode}, resolved_modes={resolved_modes}")
    for mode_index, mode in enumerate(resolved_modes, 1):
        tqdm.write(f"\n▶ Mode {mode_index}/{len(resolved_modes)}: {mode}")
        output_path = _mode_output_path(args, mode)
        payload = run_from_task_config(
            task_cfg,
            tasks_dir=args.tasks_dir,
            tools_dir=args.tools_dir,
            mode=mode,
            pattern=args.pattern,
            hint_catalog=args.hint_catalog,
            hazard_manifest=args.hazard_manifest,
            hint_injection_mode=args.hint_injection_mode,
            task_id=args.task_id,
            output_path=output_path,
            clean_output_path=_clean_output_path(args, mode, output_path),
            max_rounds=args.max_rounds,
            covered_only=not args.include_uncovered,
            request_timeout=args.request_timeout,
        )
        _maybe_log_clean_output(args, mode, output_path)
        mode_payloads[mode] = payload
        mode_summaries[mode] = _extract_summary(payload, mode)
        tqdm.write(f"✅ Mode {mode_index}/{len(resolved_modes)} complete: {mode}")

    combined_summary = {
        "meta": {
            "tasks_dir": str(args.tasks_dir),
            "tools_dir": str(args.tools_dir),
            "pattern": args.pattern,
            "task_id": args.task_id,
            "covered_only": not args.include_uncovered,
            "hint_catalog": str(args.hint_catalog) if args.hint_catalog else None,
            "hazard_manifest": str(args.hazard_manifest) if args.hazard_manifest else None,
            "hint_injection_mode": args.hint_injection_mode,
            "max_rounds": args.max_rounds,
            "max_workers": args.max_workers,
            "fail_seed": os.getenv("FAIL_SEED"),
            "runner_mode": args.mode,
        },
        "summaries": mode_summaries,
    }

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.summary_only:
        _print_summary(combined_summary)
    elif args.mode == "all":
        _print_summary(combined_summary)
    else:
        _print_summary(mode_summaries[args.mode])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
