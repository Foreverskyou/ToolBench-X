#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from agent.llm_client import LLMClient
from agent.model_driven_core import ModelDrivenToolAgent
from config.settings import LLM_MODEL
from main_model_driven import _load_tasks_from_cli


KEEP_TASK_FIELDS = [
    "task_id",
    "task_type",
    "topic",
    "subtopic",
    "relative_path",
    "exception_type",
    "user_prompt",
    "expected_answer",
]

CLEAN_TASK_RECORD_FIELDS = [
    "task_id",
    "task_type",
    "topic",
    "subtopic",
    "relative_path",
    "user_prompt",
    "expected_answer",
    "final_answer",
    "success",
    "expected_match",
    "exception_type",
    "openai_trajectory",
]

PROMPT_TASK_FIELDS = [field for field in KEEP_TASK_FIELDS if field != "expected_answer"]
ANSWER_BEARING_KEYS = {
    "answer",
    "answer_text",
    "benchmark_answer",
    "correct_answer",
    "expected",
    "expected_answer",
    "expected_output",
    "expected_result",
    "final_answer",
    "final_value",
    "gold",
    "gold_answer",
    "ground_truth",
    "reference_answer",
    "target_answer",
}
REDACTED = "[REDACTED_FOR_TEST_TIME_SCALING]"


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _extract_no_hint_failures(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    mode_results = payload.get("mode_results")
    if not isinstance(mode_results, dict) or not isinstance(mode_results.get("no_hint"), dict):
        raise ValueError("Input must be a clean AB payload with mode_results.no_hint")
    results = mode_results["no_hint"].get("results", [])
    if not isinstance(results, list):
        raise ValueError("mode_results.no_hint.results must be a list")
    return [item for item in results if isinstance(item, dict) and not bool(item.get("success"))]


def _resolve_optional_path(value: Optional[Path | str], fallback: Any, label: str) -> Path:
    raw = value if value not in (None, "") else fallback
    if raw in (None, ""):
        raise ValueError(f"{label} is required because it was not provided and source meta does not contain it")
    return Path(str(raw))


def _build_task_lookup(tasks_dir: Path, pattern: str) -> Dict[tuple[str, str], Dict[str, Any]]:
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for task in _load_tasks_from_cli(tasks_dir, pattern):
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        relative_path = str(meta.get("relative_path") or "").strip()
        task_id = str(task.get("id") or "").strip()
        if relative_path and task_id:
            lookup[(relative_path, task_id)] = task
    return lookup


def _find_source_task(item: Dict[str, Any], task_lookup: Dict[tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    relative_path = str(item.get("relative_path") or "").strip()
    task_id = str(item.get("task_id") or "").strip()
    if not relative_path or not task_id:
        raise ValueError("Failed clean result is missing relative_path or task_id; cannot reload source task")
    task = task_lookup.get((relative_path, task_id))
    if task is None:
        raise ValueError(f"Cannot find source task for relative_path={relative_path!r}, task_id={task_id!r}")
    return task


def _prepare_exception_tool_environment() -> None:
    """Keep TTS reruns aligned with the original exception-tool profiles.

    ModelDrivenToolAgent has a fallback generic INJECTION_CONFIG_JSON for exception
    tools. That fallback is useful for ad-hoc tools, but it can override the
    task-specific strict/guided profiles embedded in batch550 tools_exception.
    TTS should replay the original no-hint exception-tool behavior, so disable
    the generic fallback and clear any stale generic config before each rerun.
    """
    os.environ["DISABLE_DEFAULT_EXCEPTION_INJECTION"] = "1"
    existing = os.environ.get("INJECTION_CONFIG_JSON")
    if not existing:
        return
    try:
        parsed = json.loads(existing)
    except Exception:
        os.environ.pop("INJECTION_CONFIG_JSON", None)
        return
    if isinstance(parsed, dict) and parsed.get("before_tool_logic"):
        os.environ.pop("INJECTION_CONFIG_JSON", None)
    for key in list(os.environ.keys()):
        if key.startswith("INJECT_"):
            os.environ.pop(key, None)


def _forbidden_answer_values(*values: Any) -> List[str]:
    forbidden: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in forbidden:
            forbidden.append(text)
    return forbidden


def _redact_forbidden_text(text: str, forbidden_values: List[str]) -> str:
    redacted = str(text)
    for forbidden in forbidden_values:
        if not forbidden:
            continue
        if len(forbidden) <= 3:
            if redacted.strip() == forbidden:
                redacted = REDACTED
        else:
            redacted = redacted.replace(forbidden, REDACTED)
    return redacted


def _scrub_sensitive(value: Any, forbidden_values: Optional[List[str]] = None) -> Any:
    if isinstance(value, dict):
        scrubbed: Dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() in ANSWER_BEARING_KEYS:
                continue
            else:
                scrubbed[key] = _scrub_sensitive(child, forbidden_values)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_sensitive(item, forbidden_values) for item in value]
    if isinstance(value, str) and forbidden_values:
        return _redact_forbidden_text(value, forbidden_values)
    return value


def _scrub_message_content(content: Any, forbidden_values: Optional[List[str]] = None) -> Any:
    if not isinstance(content, str):
        return _scrub_sensitive(content, forbidden_values)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _redact_forbidden_text(content, forbidden_values or [])
    return json.dumps(_scrub_sensitive(parsed, forbidden_values), ensure_ascii=False, default=str)


def _compact_messages(openai_trajectory: Any, forbidden_values: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not isinstance(openai_trajectory, dict):
        return []
    messages = openai_trajectory.get("messages", [])
    if not isinstance(messages, list):
        return []
    compact: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        entry: Dict[str, Any] = {"role": message.get("role")}
        if message.get("content") is not None:
            entry["content"] = _scrub_message_content(message.get("content"), forbidden_values)
        if message.get("tool_calls") is not None:
            entry["tool_calls"] = _scrub_sensitive(message.get("tool_calls"), forbidden_values)
        if message.get("name") is not None:
            entry["name"] = message.get("name")
        if message.get("tool_call_id") is not None:
            entry["tool_call_id"] = message.get("tool_call_id")
        compact.append(entry)
    return compact


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    candidates = [raw]
    start = raw.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(raw)):
            ch = raw[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(raw[start : index + 1])
                        break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _build_reflection_prompt(item: Dict[str, Any]) -> str:
    public_item = {field: item.get(field) for field in PROMPT_TASK_FIELDS}
    forbidden_values = _forbidden_answer_values(item.get("expected_answer"))
    prior = {
        "final_answer": REDACTED,
        "expected_match": item.get("expected_match"),
        "trajectory_messages": _compact_messages(item.get("openai_trajectory"), forbidden_values),
    }
    return f"""
You are performing test-time scaling for a failed benchmark task.

You are given the task metadata and the previous no-hint OpenAI-style trajectory. The previous attempt was wrong or failed. Do not use the hidden expected answer; infer the correct answer only from the user request, tool results, and prior trajectory evidence.

Return exactly one JSON object with these keys:
- error_summary: a concise explanation of what went wrong in the previous attempt.
- correction_strategy: concise steps used to recover.
- final_answer: the corrected final answer. It must be a single scalar/string when the user requested a single value.

Task metadata:
{json.dumps(public_item, ensure_ascii=False, indent=2, default=str)}

Previous no-hint attempt:
{json.dumps(prior, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _build_tool_retry_prompt(item: Dict[str, Any]) -> str:
    public_item = {field: item.get(field) for field in PROMPT_TASK_FIELDS}
    forbidden_values = _forbidden_answer_values(item.get("expected_answer"))
    prior = {
        "final_answer": REDACTED,
        "expected_match": item.get("expected_match"),
        "trajectory_messages": _compact_messages(item.get("openai_trajectory"), forbidden_values),
    }
    return f"""
{str(item.get("user_prompt") or "").strip()}

[TEST_TIME_SCALING_CONTEXT]
The previous no-hint attempt for this benchmark task failed. Re-run the task by calling the available tool(s); do not answer from memory alone. Use the prior failed trajectory only to understand what went wrong, then gather fresh evidence with tool calls in this run.

Do not use or ask for the hidden expected answer. Return only the exact final scalar/string requested by the original task.

Task metadata:
{json.dumps(public_item, ensure_ascii=False, indent=2, default=str)}

Previous no-hint trajectory, with answer-bearing fields redacted:
{json.dumps(prior, ensure_ascii=False, indent=2, default=str)}

[/TEST_TIME_SCALING_CONTEXT]
""".strip()


def _prepare_retry_task(source_task: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    retry_task = copy.deepcopy(source_task)
    retry_task["user_prompt"] = _build_tool_retry_prompt(item)
    retry_task["final_answer"] = item.get("expected_answer")
    if item.get("task_id") is not None:
        retry_task["id"] = item.get("task_id")
    meta = retry_task.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta["test_time_scaling_retry"] = True
        meta["original_user_prompt"] = item.get("user_prompt")
    return retry_task


def _clean_task_record(
    item: Dict[str, Any],
    *,
    final_answer: Any,
    success: bool,
    expected_match: Optional[bool],
    openai_trajectory: Any,
) -> Dict[str, Any]:
    record = {
        "task_id": item.get("task_id"),
        "task_type": item.get("task_type"),
        "topic": item.get("topic"),
        "subtopic": item.get("subtopic"),
        "relative_path": item.get("relative_path"),
        "user_prompt": item.get("user_prompt"),
        "expected_answer": item.get("expected_answer"),
        "final_answer": final_answer,
        "success": success,
        "expected_match": expected_match,
        "exception_type": item.get("exception_type"),
        "openai_trajectory": openai_trajectory,
    }
    return {field: record.get(field) for field in CLEAN_TASK_RECORD_FIELDS}


def _run_one(
    item: Dict[str, Any],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: Optional[float],
    max_tokens: int,
    request_timeout: float,
    extra_body: Optional[Dict[str, Any]],
    tools_dir: Path,
    max_rounds: int,
    task_lookup: Dict[tuple[str, str], Dict[str, Any]],
    include_trace: bool,
    include_answers: bool,
) -> Dict[str, Any]:
    started = time.time()
    llm: Optional[LLMClient] = None
    expected = item.get("expected_answer")
    try:
        if expected is None:
            raise ValueError("Task is missing expected_answer; cannot score test-time scaling retry")
        source_task = _find_source_task(item, task_lookup)
        retry_task = _prepare_retry_task(source_task, item)
        _prepare_exception_tool_environment()
        llm = LLMClient(
            model=model,
            base_url=api_url or None,
            api_key=api_key or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            extra_body=extra_body,
        )
        agent = ModelDrivenToolAgent(tools_dir=tools_dir, llm=llm, max_rounds=max_rounds)
        execution = agent.execute_task(retry_task)
        final_answer = str(execution.get("final_answer") or "").strip()
        expected_match = str(final_answer).strip() == str(expected).strip()
        result = _clean_task_record(
            item,
            final_answer=final_answer,
            success=bool(expected_match),
            expected_match=expected_match,
            openai_trajectory=execution.get("openai_trajectory") or {},
        )
        if include_trace:
            result["duration_sec"] = round(time.time() - started, 3)
            result["error"] = _redact_forbidden_text(str(execution.get("error") or "").strip(), _forbidden_answer_values(expected)) or None
            result["tool_execution_rounds"] = len(execution.get("execution_rounds") or [])
            result["tts_openai_trace"] = llm.get_trace() if llm is not None else []
        return result
    except Exception as exc:
        result = _clean_task_record(
            item,
            final_answer=f"Error: {exc}",
            success=False,
            expected_match=False if expected is not None else None,
            openai_trajectory={"format": "openai_chat_completions_messages", "messages": []},
        )
        if include_trace:
            result["duration_sec"] = round(time.time() - started, 3)
            result["error"] = str(exc)
            result["tts_openai_trace"] = llm.get_trace() if llm is not None else []
        return result


def _rate(success: int, total: int) -> str:
    return f"{success / total * 100:.1f}%" if total else "0.0%"


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    success = sum(1 for item in results if item.get("success"))
    return {
        "total": total,
        "success": success,
        "failed": total - success,
        "success_rate": _rate(success, total),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry failed no-hint clean AB tasks using prior OpenAI trajectory as test-time scaling context")
    parser.add_argument("--input", type=Path, required=True, help="Clean AB JSON containing mode_results.no_hint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, help="Task JSON root. Defaults to source meta tasks_dir from the clean AB input.")
    parser.add_argument("--tools-dir", type=Path, help="Tool root used for reruns. Defaults to source meta tools_dir from the clean AB input.")
    parser.add_argument("--pattern", type=str, default="**/*.json")
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    parser.add_argument("--api-url", type=str, default="", help="Optional override; defaults to OPENAI_BASE_URL loaded from config/.env")
    parser.add_argument("--api-key", type=str, default="", help="Optional override; defaults to OPENAI_API_KEY loaded from config/.env")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-rounds", type=int, default=None, help="Maximum LLM-tool execution rounds for each rerun. Defaults to source meta max_rounds, then 30.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional cap for debugging; 0 means all failed no_hint tasks")
    parser.add_argument("--fail-seed", type=str, help="Seed for deterministic exception injection; use the same value as the source AB run, e.g. 7")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--agent-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING", help="Logging level for agent internals; default suppresses INFO task progress logs")
    parser.add_argument("--include-answers", action="store_true", help="Include expected_answer in output artifacts. Default omits answer keys to avoid contamination.")
    parser.add_argument("--include-trace", action="store_true", help="Include raw retry response and OpenAI trace in output artifacts. Default omits them to avoid persisting prompt/tool data.")
    args = parser.parse_args()

    logging.getLogger("agent").setLevel(getattr(logging, args.agent_log_level))

    resolved_api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    resolved_api_url = args.api_url or os.getenv("OPENAI_BASE_URL", "")
    if resolved_api_key:
        os.environ["OPENAI_API_KEY"] = resolved_api_key
    if resolved_api_url:
        os.environ["OPENAI_BASE_URL"] = resolved_api_url

    payload = _load_json(args.input)
    source_meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    tasks_dir = _resolve_optional_path(args.tasks_dir, source_meta.get("tasks_dir"), "--tasks-dir")
    tools_dir = _resolve_optional_path(args.tools_dir, source_meta.get("tools_dir"), "--tools-dir")
    resolved_max_rounds = args.max_rounds if args.max_rounds is not None else int(source_meta.get("max_rounds") or 30)
    resolved_fail_seed = args.fail_seed if args.fail_seed is not None else source_meta.get("fail_seed")
    if resolved_fail_seed is not None:
        os.environ["FAIL_SEED"] = str(resolved_fail_seed)
    task_lookup = _build_task_lookup(tasks_dir, args.pattern)
    failures = _extract_no_hint_failures(payload)
    if args.max_tasks and args.max_tasks > 0:
        failures = failures[: args.max_tasks]
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None

    results: List[Optional[Dict[str, Any]]] = [None] * len(failures)
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor, tqdm(total=len(failures), desc="tts", unit="task") as pbar:
        future_to_index = {
            executor.submit(
                _run_one,
                item,
                model=args.model,
                api_url=resolved_api_url,
                api_key=resolved_api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                extra_body=extra_body,
                tools_dir=tools_dir,
                max_rounds=resolved_max_rounds,
                task_lookup=task_lookup,
                include_trace=args.include_trace,
                include_answers=args.include_answers,
            ): index
            for index, item in enumerate(failures)
        }
        ok = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            if result.get("success"):
                ok += 1
            pbar.set_postfix({"ok": ok, "fail": pbar.n + 1 - ok})
            pbar.update(1)

    final_results = [item for item in results if isinstance(item, dict)]
    summary = _summarize(final_results)
    output_payload = {
        "format": "test_time_scaling_from_no_hint_failures",
        "source_file": str(args.input),
        "source_meta": source_meta,
        "meta": {
            "tasks_dir": str(tasks_dir),
            "tools_dir": str(tools_dir),
            "pattern": args.pattern,
            "llm_model": args.model,
            "api_url": resolved_api_url,
            "api_key_source": "cli" if args.api_key else "config/.env or environment",
            "api_url_source": "cli" if args.api_url else "config/.env or environment",
            "llm_temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "request_timeout": args.request_timeout,
            "max_rounds": resolved_max_rounds,
            "max_workers": args.max_workers,
            "fail_seed": os.getenv("FAIL_SEED"),
            "disable_thinking": args.disable_thinking,
            "include_answers": args.include_answers,
            "include_trace": args.include_trace,
            "source_no_hint_total": payload.get("mode_results", {}).get("no_hint", {}).get("total"),
            "source_no_hint_success": payload.get("mode_results", {}).get("no_hint", {}).get("success"),
            "selected_failures": len(failures),
            "duration_sec_total": round(time.time() - started, 3),
        },
        "results": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
