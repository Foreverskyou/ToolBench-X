import ast
import fnmatch
import json
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from gpt import GPT54
from prompt_genexception import prompt_genexception

load_dotenv(Path(__file__).resolve().parent / "config" / ".env")

_thread_local = threading.local()
checkpoint = None

VALID_EXCEPTION_CATEGORIES = [
    "Specification Uncertainty",
    "Invocation Uncertainty",
    "Execution Uncertainty",
    "Output Uncertainty",
    "Cross-Source Uncertainty",
]

SPECIFICATION_DRIFT_MODES = {"field_rename_drift", "type_drift", "shape_drift", "semantic_unit_drift"}
INVOCATION_DRIFT_MODES = {"drop_fields", "rename_fields", "coerce_types", "implicit_defaults"}
OUTPUT_DRIFT_MODES = {"unit_or_currency_wrapper", "label_plus_value_format", "alias_not_canonical", "explanation_appended", "truncate_or_reorder_payload"}


class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.lock = threading.Lock()
        self.completed_tasks: Set[str] = set()
        self.results: List[Dict] = []
        self.api_quota_exhausted = False
        self.paused = False
        self.load()

    def load(self) -> None:
        if self.checkpoint_path.exists():
            try:
                with self.checkpoint_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.results = data.get("results", [])
                self.completed_tasks = {
                    str(item.get("task_key", "")).strip()
                    for item in self.results
                    if isinstance(item, dict) and item.get("success") and str(item.get("task_key", "")).strip()
                }
                legacy_completed = set(data.get("completed_tasks", []))
                if legacy_completed and legacy_completed != self.completed_tasks:
                    print("ℹ️ 已按 success 结果重建 checkpoint completed_tasks，忽略旧的失败 completed 记录")
                self.api_quota_exhausted = data.get("api_quota_exhausted", False)
                self.paused = data.get("paused", False)
                print(f"📂 已加载进度：{len(self.completed_tasks)} 个已完成任务")
            except Exception as e:
                print(f"⚠️ 加载 checkpoint 失败：{e}，从头开始")
        else:
            print("🆕 未找到 checkpoint，从头开始")

    def _save_unlocked(self) -> None:
        data = {
            "completed_tasks": list(self.completed_tasks),
            "results": self.results,
            "api_quota_exhausted": self.api_quota_exhausted,
            "paused": self.paused,
            "timestamp": time.time(),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, self.checkpoint_path)

    def save(self) -> None:
        with self.lock:
            self._save_unlocked()

    def is_completed(self, task_key: str) -> bool:
        return task_key in self.completed_tasks

    def record_result(self, task_key: str, result: Dict) -> None:
        with self.lock:
            replaced = False
            for idx, existing in enumerate(self.results):
                if isinstance(existing, dict) and str(existing.get("task_key", "")).strip() == task_key:
                    self.results[idx] = result
                    replaced = True
                    break
            if not replaced:
                self.results.append(result)
            if result.get("success"):
                self.completed_tasks.add(task_key)
            else:
                self.completed_tasks.discard(task_key)
            self._save_unlocked()

    def set_quota_exhausted(self, exhausted: bool) -> None:
        with self.lock:
            self.api_quota_exhausted = exhausted
            self._save_unlocked()

    def reset(self) -> None:
        with self.lock:
            self.completed_tasks.clear()
            self.results.clear()
            self.api_quota_exhausted = False
            self.paused = False
            self._save_unlocked()
            print("🔄 已重置所有进度")

    def get_stats(self) -> Dict:
        return {
            "completed": len(self.completed_tasks),
            "api_quota_exhausted": self.api_quota_exhausted,
            "paused": self.paused,
        }


def is_quota_error(error_msg: str) -> bool:
    quota_keywords = [
        "429",
        "rate limit",
        "quota",
        "credit",
        "balance",
        "insufficient",
        "exceeded",
        "limit",
        "too many requests",
    ]
    text = str(error_msg).lower()
    return any(keyword in text for keyword in quota_keywords)


def signal_handler(signum, frame):
    if checkpoint is not None:
        print("\n\n⚠️ 检测到中断信号，正在保存进度...")
        checkpoint.paused = True
        checkpoint.save()
        print("💾 进度已保存，下次运行将自动恢复")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_thread_llm() -> GPT54:
    if not hasattr(_thread_local, "llm"):
        _thread_local.llm = GPT54()
    return _thread_local.llm


def worker_initializer() -> None:
    _thread_local.llm = GPT54()


@dataclass
class GenerationResult:
    task_key: str
    task_id: str
    source_filepath: str
    filepath: str
    success: bool
    exception_category: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    exception_hints: Optional[Dict] = None
    hint_error: Optional[str] = None
    hint_filepath: Optional[str] = None


def clean_code_block(code_content: str) -> str:
    return re.sub(r"^```(?:python)?\s*|\s*```$", "", code_content.strip(), flags=re.MULTILINE)


def safe_path_segment(value: str) -> str:
    return re.sub(r"[^\w\s-]", "", str(value)).strip().replace(" ", "_") or "unknown"


def build_task_key(task_type: str, main_topic: str, subtopic: str, task_id: str) -> str:
    return f"{task_type}/{main_topic}/{subtopic}/{task_id}"


def build_tool_path(
    base_dir: str,
    task_type: str,
    main_topic: str,
    subtopic: str,
    task_id: str,
    exception_category: Optional[str] = None,
) -> str:
    parts = [base_dir, safe_path_segment(task_type)]
    if exception_category:
        parts.append(safe_path_segment(exception_category))
    parts.extend(
        [
            safe_path_segment(main_topic),
            safe_path_segment(subtopic),
            f"{safe_path_segment(task_id)}.py",
        ]
    )
    return os.path.join(*parts)


def save_generated_code(
    code_content: str,
    output_dir: str,
    task_type: str,
    exception_category: str,
    main_topic: str,
    subtopic: str,
    task_id: str,
) -> Tuple[bool, str]:
    try:
        cleaned = clean_code_block(code_content)
        filepath = build_tool_path(output_dir, task_type, main_topic, subtopic, task_id, exception_category=exception_category)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return True, filepath
    except Exception as e:
        return False, str(e)


def save_hint_json(hints: Dict, output_python_filepath: str) -> Tuple[bool, str]:
    try:
        hint_path = str(Path(output_python_filepath).with_suffix(".hints.json"))
        Path(hint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(hint_path, "w", encoding="utf-8") as f:
            json.dump(hints, f, ensure_ascii=False, indent=2)
        return True, hint_path
    except Exception as e:
        return False, str(e)


def rebuild_hint_catalog_from_output_dir(output_dir: str) -> Dict[str, Dict[str, Any]]:
    base = Path(output_dir)
    catalog: Dict[str, Dict[str, Any]] = {}
    if not base.exists():
        return catalog
    for hint_path in sorted(base.rglob("*.hints.json")):
        try:
            rel = hint_path.relative_to(base)
        except Exception:
            continue
        parts = rel.parts
        task_key = None
        if len(parts) >= 5:
            task_type = parts[0]
            main_topic = parts[2]
            subtopic = parts[3]
            task_id = Path(parts[4]).stem.replace(".hints", "")
            task_key = build_task_key(task_type, main_topic, subtopic, task_id)
        elif len(parts) >= 4:
            task_type = parts[0]
            main_topic = parts[1]
            subtopic = parts[2]
            task_id = Path(parts[3]).stem.replace(".hints", "")
            task_key = build_task_key(task_type, main_topic, subtopic, task_id)
        if not task_key:
            continue
        try:
            hints = json.loads(hint_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(hints, dict) and hints:
            catalog[task_key] = hints
    return catalog


def load_baseline_success_task_keys(results_path: Optional[str]) -> Optional[Set[str]]:
    if not results_path:
        return None
    path = Path(results_path)
    if not path.exists():
        print(f"⚠️ baseline results not found: {path}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ failed to parse baseline results {path}: {e}")
        return None

    result_blocks = []
    if isinstance(data.get("results"), dict):
        result_blocks = data.get("results", {}).get("results", [])
    elif isinstance(data.get("results"), list):
        result_blocks = data.get("results", [])

    success_keys: Set[str] = set()
    for item in result_blocks:
        if not isinstance(item, dict) or not item.get("success"):
            continue
        task_id = str(item.get("task_id") or "").strip()
        rel = str(item.get("relative_path") or item.get("task_relative_path") or item.get("_meta", {}).get("relative_path", "")).strip()
        if not task_id or not rel:
            continue
        rel_path = Path(rel)
        if len(rel_path.parts) < 3:
            continue
        task_type = rel_path.parts[0]
        main_topic = rel_path.parts[1]
        subtopic = rel_path.stem
        success_keys.add(build_task_key(task_type, main_topic, subtopic, task_id))
    print(f"📘 Loaded {len(success_keys)} baseline-success task keys from {path}")
    return success_keys


def extract_exception_hints(code_content: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        cleaned = clean_code_block(code_content)
        tree = ast.parse(cleaned)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "EXCEPTION_HINTS":
                        try:
                            value = ast.literal_eval(node.value)
                        except Exception as e:
                            return None, f"EXCEPTION_HINTS 解析失败: {e}"
                        if isinstance(value, dict):
                            return value, None
                        return None, "EXCEPTION_HINTS 不是 dict"
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name) and target.id == "EXCEPTION_HINTS":
                    try:
                        value = ast.literal_eval(node.value) if node.value is not None else None
                    except Exception as e:
                        return None, f"EXCEPTION_HINTS 解析失败: {e}"
                    if isinstance(value, dict):
                        return value, None
                    return None, "EXCEPTION_HINTS 不是 dict"
        return None, "未找到 EXCEPTION_HINTS"
    except Exception as e:
        return None, f"代码解析失败: {e}"


def _normalize_tools_used(raw_tools: Any) -> List[str]:
    normalized: List[str] = []
    if not isinstance(raw_tools, list):
        return normalized
    for item in raw_tools:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
            continue
        if isinstance(item, dict):
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name:
                normalized.append(tool_name)
    return normalized


def _infer_canonical_answer_rules(final_answer: str) -> List[str]:
    answer = str(final_answer or "").strip()
    rules = [
        "Final answer must exactly match the benchmark expected_answer after trimming outer whitespace only.",
        "Do not add explanation text, labels, prefixes, suffixes, or narrative when benchmark expects a scalar/string answer.",
        "Reject wrapped answer forms like 'Answer: X', 'Result: X', or any prose-appended variant even when the core value looks plausible.",
        "Never echo input location/site/session/cart identifiers as the final answer unless the benchmark expected_answer is exactly that identifier.",
    ]
    if not answer:
        return rules
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
        rules.extend(
            [
                "Return numeric value only; no currency code, currency symbol, unit, or surrounding prose.",
                "Preserve decimal digits exactly as benchmark answer; do not round, pad, or reformat.",
            ]
        )
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer):
        rules.append("Return date exactly in YYYY-MM-DD format with no extra text.")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", answer):
        rules.append("Return timestamp exactly in ISO-8601 UTC form with trailing Z and no extra text.")
    if answer in {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }:
        rules.append("Return a single weekday token only; no sentence, label, or explanation.")
    if re.fullmatch(r"[A-Z]{2,}(?:[_-][A-Z0-9]+)+", answer):
        rules.append("Return the exact uppercase code token only; do not expand or explain the code.")
    if re.fullmatch(r"[A-Za-z]{1,4}-\d{2,}", answer):
        rules.append("Return the exact identifier token only; do not prepend status text or explanatory wording.")
    if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)*", answer):
        rules.append("Return a single lowercase enum token only; do not return input location text, labels like Answer:, or explanatory prose.")
    return rules


def _infer_hint_category_signature(item: Dict[str, Any]) -> Optional[str]:
    failpoint = str(item.get("failpoint") or "").strip()
    exception_type = str(item.get("exception_type") or "").strip()
    drift_mode = str(item.get("drift_mode") or "").strip()
    guardrail_text = " ".join(
        [
            str(item.get("likely_root_cause") or ""),
            str(item.get("symptom") or ""),
            " ".join(str(v) for v in (item.get("verification_checks") or [])),
            " ".join(str(v) for v in (item.get("recovery_steps") or [])),
        ]
    ).lower()
    if drift_mode in SPECIFICATION_DRIFT_MODES:
        return "Specification Uncertainty"
    if drift_mode in INVOCATION_DRIFT_MODES or (failpoint == "before_tool_logic" and exception_type in {"ValueError", "KeyError"}):
        return "Invocation Uncertainty"
    if exception_type in {"TimeoutError", "ConnectionError", "OSError"} or failpoint in {"before_external_call", "before_checkpoint_write"}:
        return "Execution Uncertainty"
    if drift_mode in OUTPUT_DRIFT_MODES or failpoint in {"before_return", "after_wrapper_transform"}:
        return "Output Uncertainty"
    if any(token in guardrail_text for token in ["cross-check", "contradict", "paired", "sibling", "cross source", "cross-source"]):
        return "Cross-Source Uncertainty"
    return None


def _extract_task_exception_category(hints: Dict[str, Any]) -> Optional[str]:
    categories: List[str] = []
    for item in (hints or {}).values():
        if not isinstance(item, dict):
            continue
        category = str(item.get("exception_category") or "").strip()
        if category and category not in categories:
            categories.append(category)
    return categories[0] if len(categories) == 1 else None


def _force_assigned_exception_category(hints: Dict[str, Any], target_exception_category: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for hint_key, item in (hints or {}).items():
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["exception_category"] = target_exception_category
        normalized[str(hint_key)] = updated
    return normalized


def _task_has_cross_source_semantics(hints: Dict[str, Any], task_meta: Dict[str, Any]) -> bool:
    task_type = str(task_meta.get("task_type") or "").strip()
    task_tools = set(_normalize_tools_used(task_meta.get("task_item", {}).get("tools_used", [])))
    mentioned_tools = set()
    text_chunks: List[str] = []

    for hint_key, item in (hints or {}).items():
        if not isinstance(item, dict):
            continue
        if "::" in str(hint_key):
            mentioned_tools.add(str(hint_key).split("::", 1)[0].strip())
        for tool_name in item.get("mandatory_tool_sequence", []) or []:
            text = str(tool_name).strip()
            if text:
                mentioned_tools.add(text)
        for field in [
            "likely_root_cause",
            "symptom",
            "alternate_correct_path",
            "path_switch_signal",
            "minimal_prompt_hint",
            "detailed_prompt_hint",
        ]:
            text = str(item.get(field) or "").strip()
            if text:
                text_chunks.append(text)
        for list_field in [
            "verification_checks",
            "recovery_steps",
            "guardrail_checks",
            "evidence_requirements",
            "finish_criteria",
            "forbidden_early_finish_when",
        ]:
            for value in item.get(list_field, []) or []:
                text = str(value).strip()
                if text:
                    text_chunks.append(text)

    if task_type in {"parallel", "mixture"} and len(task_tools) >= 2:
        return True
    if len(mentioned_tools) >= 2:
        return True

    aggregate_text = " ".join(text_chunks).lower()
    cross_source_markers = [
        "cross-check",
        "cross check",
        "compare with",
        "compare against",
        "reconcile",
        "contradict",
        "another source",
        "secondary source",
        "single source",
        "do not trust one source",
        "verify against",
        "paired tuple",
        "sibling field",
        "upstream verified",
        "cross-source",
        "cross source",
    ]
    return any(marker in aggregate_text for marker in cross_source_markers)


def _compatible_exception_categories(task_meta: Dict[str, Any]) -> List[str]:
    task_item = task_meta.get("task_item", {}) if isinstance(task_meta.get("task_item"), dict) else {}
    task_type = str(task_meta.get("task_type") or "").strip()
    main_topic = str(task_meta.get("main_topic") or "").strip().lower()
    subtopic = str(task_meta.get("subtopic") or "").strip().lower()
    user_prompt = str(task_item.get("user_prompt") or "").lower()
    expected_answer = str(task_item.get("expected_answer") or task_item.get("final_answer") or "").strip()
    tool_names = [tool.lower() for tool in _normalize_tools_used(task_item.get("tools_used", []))]
    joined_tools = " ".join(tool_names)

    allowed: List[str] = ["Execution Uncertainty", "Output Uncertainty"]

    transaction_markers = [
        "checkout", "payment", "cart", "order", "session", "customer",
        "invoice", "refund", "return", "shipment", "tracking", "sku",
        "account", "subscription", "auth", "login",
    ]
    explicit_id_markers = bool(re.search(r"\b[A-Z]{2,}-\d{2,}\b", user_prompt)) or any(token in user_prompt for token in [" cart ", " checkout ", " order ", " session ", " payment "])
    if any(marker in joined_tools for marker in transaction_markers) or explicit_id_markers:
        allowed.append("Invocation Uncertainty")

    if task_type in {"parallel", "mixture"} or len(tool_names) >= 3:
        allowed.append("Cross-Source Uncertainty")

    location_environment = "location__environment" in main_topic or "environment" in main_topic or "weather" in subtopic or "outage" in subtopic or "river" in subtopic
    if location_environment:
        allowed = [category for category in allowed if category != "Invocation Uncertainty"]
        if "Cross-Source Uncertainty" not in allowed and (task_type in {"parallel", "mixture"} or len(tool_names) >= 2):
            allowed.append("Cross-Source Uncertainty")

    if re.fullmatch(r"[A-Z]{2,}(?:[_-][A-Z0-9]+)+", expected_answer) or re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)*", expected_answer):
        if "Specification Uncertainty" not in allowed:
            allowed.append("Specification Uncertainty")

    if task_type == "sequential" and any(name.startswith(("validate_", "parse_", "normalize_", "resolve_")) for name in tool_names):
        if "Specification Uncertainty" not in allowed:
            allowed.append("Specification Uncertainty")

    ordered_allowed = [category for category in VALID_EXCEPTION_CATEGORIES if category in allowed]
    return ordered_allowed or ["Execution Uncertainty"]


def select_tasks_by_category_pools(tasks: List[Dict[str, Any]], max_tasks: Optional[int]) -> List[Dict[str, Any]]:
    if not tasks:
        return tasks
    ordered = sorted(list(tasks), key=lambda item: str(item.get("task_key", "")))
    for task in ordered:
        task["compatible_exception_categories"] = _compatible_exception_categories(task)

    target_total = len(ordered) if not max_tasks or max_tasks <= 0 else min(max_tasks, len(ordered))
    category_count = len(VALID_EXCEPTION_CATEGORIES)
    base_quota = target_total // category_count
    remainder = target_total % category_count
    target_quota: Dict[str, int] = {
        category: base_quota + (1 if index < remainder else 0)
        for index, category in enumerate(VALID_EXCEPTION_CATEGORIES)
    }

    pools: Dict[str, List[Dict[str, Any]]] = {category: [] for category in VALID_EXCEPTION_CATEGORIES}
    for task in ordered:
        for category in task.get("compatible_exception_categories", []):
            if category in pools:
                pools[category].append(task)
    for category in VALID_EXCEPTION_CATEGORIES:
        pools[category].sort(key=lambda item: (len(item.get("compatible_exception_categories", [])), str(item.get("task_key", ""))))

    selected_by_key: Dict[str, Dict[str, Any]] = {}
    selected_counts: Dict[str, int] = {category: 0 for category in VALID_EXCEPTION_CATEGORIES}

    def try_select_for_category(category: str, candidate_list: List[Dict[str, Any]], limit: int) -> None:
        if limit <= 0:
            return
        for task in candidate_list:
            if selected_counts[category] >= limit:
                break
            task_key = str(task.get("task_key", ""))
            if task_key in selected_by_key:
                continue
            chosen = dict(task)
            chosen["target_exception_category"] = category
            selected_by_key[task_key] = chosen
            selected_counts[category] += 1

    # Phase 1: category-first quota sampling.
    for category in VALID_EXCEPTION_CATEGORIES:
        try_select_for_category(category, pools[category], target_quota[category])

    # Phase 2: refill underfilled categories from remaining compatible tasks.
    deficits = True
    while deficits and len(selected_by_key) < target_total:
        deficits = False
        for category in sorted(VALID_EXCEPTION_CATEGORIES, key=lambda c: (target_quota[c] - selected_counts[c], -VALID_EXCEPTION_CATEGORIES.index(c)), reverse=True):
            deficit = target_quota[category] - selected_counts[category]
            if deficit <= 0:
                continue
            deficits = True
            remaining_pool = [task for task in pools[category] if str(task.get("task_key", "")) not in selected_by_key]
            before = selected_counts[category]
            try_select_for_category(category, remaining_pool, target_quota[category])
            if len(selected_by_key) >= target_total:
                break
            if selected_counts[category] == before:
                continue

    # Phase 3: global refill to hit target_total while keeping balance as even as possible.
    if len(selected_by_key) < target_total:
        remaining = [task for task in ordered if str(task.get("task_key", "")) not in selected_by_key]
        for task in remaining:
            if len(selected_by_key) >= target_total:
                break
            compatible = list(task.get("compatible_exception_categories", []))
            if not compatible:
                continue
            category = min(
                compatible,
                key=lambda c: (selected_counts.get(c, 0), VALID_EXCEPTION_CATEGORIES.index(c)),
            )
            chosen = dict(task)
            chosen["target_exception_category"] = category
            selected_by_key[str(task.get("task_key", ""))] = chosen
            selected_counts[category] += 1

    selected_tasks = list(selected_by_key.values())
    selected_tasks.sort(key=lambda item: str(item.get("task_key", "")))
    return selected_tasks


def assign_balanced_exception_categories(tasks: List[Dict[str, Any]], max_tasks: Optional[int]) -> List[Dict[str, Any]]:
    if not tasks:
        return tasks
    ordered = sorted(list(tasks), key=lambda item: str(item.get("task_key", "")))
    if max_tasks is not None and max_tasks > 0:
        ordered = ordered[:max_tasks]
    category_counts: Dict[str, int] = {category: 0 for category in VALID_EXCEPTION_CATEGORIES}
    assigned: List[Dict[str, Any]] = []
    for task in ordered:
        compatible = _compatible_exception_categories(task)
        chosen = dict(task)
        chosen["compatible_exception_categories"] = compatible
        target_category = min(
            compatible,
            key=lambda category: (category_counts.get(category, 0), VALID_EXCEPTION_CATEGORIES.index(category)),
        )
        chosen["target_exception_category"] = target_category
        category_counts[target_category] += 1
        assigned.append(chosen)
    return assigned


def _build_generation_requirements(task_meta: Dict[str, Any]) -> Dict[str, Any]:
    task_item = task_meta.get("task_item", {}) if isinstance(task_meta.get("task_item"), dict) else {}
    expected_answer = str(task_item.get("expected_answer", "")).strip()
    final_answer = str(task_item.get("final_answer", "")).strip()
    benchmark_answer = expected_answer or final_answer
    tools_used = _normalize_tools_used(task_item.get("tools_used", []))
    target_exception_category = str(task_meta.get("target_exception_category") or "").strip()
    return {
        "task_key": task_meta.get("task_key"),
        "expected_answer": benchmark_answer,
        "expected_answer_source": "expected_answer" if expected_answer else "final_answer",
        "exact_match_required": True,
        "tools_used_normalized": tools_used,
        "target_failure_families": [
            "tool_exception_and_drift",
            "premature_finish_or_wrong_intermediate_trust",
            "final_answer_canonicalization_failure",
        ],
        "allowed_exception_categories": VALID_EXCEPTION_CATEGORIES,
        "compatible_exception_categories": task_meta.get("compatible_exception_categories") or VALID_EXCEPTION_CATEGORIES,
        "single_exception_category_required": True,
        "target_exception_category": target_exception_category,
        "required_recovery_properties": [
            "When injection is disabled, the patched tool code must preserve the original benchmark-correct behavior and return schema.",
            "When strict_no_hint_profile is active, at least one realistic no-hint execution path should be able to fail or drift enough to reduce average success rate.",
            "When guided_with_hint_profile is active with valid hint context, the same underlying fault schedule should become more recoverable on average than strict_no_hint_profile.",
            "from_start hints must not weaken baseline correctness; they should improve or preserve average success relative to no_hint under the same injected failures.",
            "Preserve at least one deterministic path to the exact expected_answer.",
            "When the primary path is broken, leave one alternate correct path that remains reachable and verifiable.",
            "Hints must explicitly name the broken path, the alternate correct path, and the switch signal.",
            "Hints must tell the model how to verify a suspicious tool result is wrong or incomplete before trusting it.",
            "Hints must explicitly block premature finish until evidence completeness and contradiction checks pass.",
            "Hints must require final-answer canonicalization to the exact benchmark surface form before FINISH.",
            "When an upstream tool verifies a paired business tuple like best_promo_code plus discount_usd, downstream recovery steps must preserve that verified tuple instead of inventing a new combination.",
            "All failpoints for a task must share exactly one exception_category from the allowed taxonomy.",
            "The exception_category must be chosen from compatible_exception_categories for this specific task.",
        ],
        "canonical_answer_rules": _infer_canonical_answer_rules(benchmark_answer),
        "generation_acceptance_checks": [
            "Baseline mode must preserve exact expected-answer reachability and original return schema when injection is disabled.",
            "strict_no_hint_profile must be materially harsher than guided_with_hint_profile on at least one control-critical failpoint.",
            "Hints must improve recoverability through explicit retry/fallback/verification guidance rather than by removing failure contrast.",
            "Every EXCEPTION_HINTS entry contains verification_checks and canonical_answer_rules.",
            "Every EXCEPTION_HINTS entry contains alternate_correct_path and path_switch_signal.",
            "Every detailed_prompt_hint contains STEP 1 CHECK / STEP 2 ACTION / STEP 3 STOP/FINISH GATE.",
            "Every detailed_prompt_hint embeds HINT_GUARDS_JSON with verification_checks and canonical_answer_rules.",
            "At least one instruction explicitly blocks premature finish and one explicitly canonicalizes the final answer.",
            "For enum/code/day style answers, canonical rules must explicitly reject wrapped labels/prose and context echoes like input location or identifiers.",
            "For promotion/price tasks, hints must require best_promo_code and discount_usd to remain consistent with the verified upstream promotion payload before shipping or final-total steps.",
            "Every EXCEPTION_HINTS entry must include the same valid exception_category for the task.",
            "The generated exception_category must exactly match target_exception_category.",
        ],
    }


def _extract_hint_guards_json(detailed_hint: str) -> Optional[Dict[str, Any]]:
    text = str(detailed_hint or "")
    marker = "HINT_GUARDS_JSON"
    idx = text.find(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker) :]
    start = tail.find("{")
    if start < 0:
        return None
    brace_depth = 0
    end_index = -1
    for i, ch in enumerate(tail[start:]):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end_index = start + i + 1
                break
    if end_index < 0:
        return None
    json_block = tail[start:end_index]
    try:
        parsed = json.loads(json_block)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def validate_exception_hints(hints: Dict[str, Any], task_meta: Dict[str, Any]) -> Optional[str]:
    if not isinstance(hints, dict) or not hints:
        return "EXCEPTION_HINTS 为空或不是 dict"
    required_fields = {
        "exception_category",
        "failpoint",
        "exception_type",
        "drift_mode",
        "symptom",
        "likely_root_cause",
        "recovery_steps",
        "retry_strategy",
        "guardrail_checks",
        "required_inputs",
        "finish_criteria",
        "forbidden_early_finish_when",
        "mandatory_tool_sequence",
        "fallback_order",
        "evidence_requirements",
        "answer_fields_required",
        "verification_checks",
        "canonical_answer_rules",
        "alternate_correct_path",
        "path_switch_signal",
        "minimal_prompt_hint",
        "detailed_prompt_hint",
    }
    required_guard_keys = {
        "required_inputs",
        "mandatory_tool_sequence",
        "fallback_order",
        "finish_criteria",
        "forbidden_early_finish_when",
        "answer_fields_required",
        "verification_checks",
        "canonical_answer_rules",
    }
    task_tools = set(_normalize_tools_used(task_meta.get("task_item", {}).get("tools_used", [])))
    target_exception_category = str(task_meta.get("target_exception_category") or "").strip()
    validation_errors: List[str] = []
    has_finish_block = False
    has_canonicalization = False
    has_retry_or_fallback = False
    has_required_input_guard = False
    seen_categories: List[str] = []
    for hint_key, item in sorted(hints.items()):
        if not isinstance(item, dict):
            validation_errors.append(f"{hint_key}: hint 条目不是 dict")
            continue
        missing_fields = sorted(required_fields - set(item.keys()))
        if missing_fields:
            validation_errors.append(f"{hint_key}: 缺少字段 {missing_fields}")
        verification_checks = item.get("verification_checks")
        canonical_rules = item.get("canonical_answer_rules")
        exception_category = str(item.get("exception_category", "")).strip()
        alternate_correct_path = str(item.get("alternate_correct_path", "")).strip()
        path_switch_signal = str(item.get("path_switch_signal", "")).strip()
        if exception_category not in VALID_EXCEPTION_CATEGORIES:
            validation_errors.append(f"{hint_key}: exception_category 非法或缺失")
        elif exception_category not in seen_categories:
            seen_categories.append(exception_category)
        if not isinstance(verification_checks, list) or not verification_checks:
            validation_errors.append(f"{hint_key}: verification_checks 为空")
        if not isinstance(canonical_rules, list) or not canonical_rules:
            validation_errors.append(f"{hint_key}: canonical_answer_rules 为空")
        if not alternate_correct_path:
            validation_errors.append(f"{hint_key}: alternate_correct_path 为空")
        if not path_switch_signal:
            validation_errors.append(f"{hint_key}: path_switch_signal 为空")
        minimal_hint = str(item.get("minimal_prompt_hint", "")).strip()
        detailed_hint = str(item.get("detailed_prompt_hint", "")).strip()
        if "DO NOT FINISH UNTIL" in minimal_hint or "DO NOT FINISH UNTIL" in detailed_hint:
            has_finish_block = True
        detailed_hint_upper = detailed_hint.upper()
        if "CANONICAL" in detailed_hint_upper or "FINAL ANSWER" in detailed_hint_upper:
            has_canonicalization = True
        if any(token in detailed_hint_upper for token in ["RETRY", "FALLBACK", "ALTERNATE_CORRECT_PATH"]):
            has_retry_or_fallback = True
        required_inputs = item.get("required_inputs")
        if isinstance(required_inputs, dict) and any((values or []) for values in required_inputs.values()):
            has_required_input_guard = True
        if "STEP 1 CHECK" not in detailed_hint or "STEP 2 ACTION" not in detailed_hint or "STEP 3 STOP/FINISH GATE" not in detailed_hint:
            validation_errors.append(f"{hint_key}: detailed_prompt_hint 缺少标准 STEP 结构")
        detailed_hint_lower = detailed_hint.lower()
        has_alt_label = "alternate_correct_path:" in detailed_hint_lower
        has_switch_label = "path_switch_signal:" in detailed_hint_lower
        if alternate_correct_path and alternate_correct_path.lower() not in detailed_hint_lower and not has_alt_label:
            validation_errors.append(f"{hint_key}: detailed_prompt_hint 未显式包含 alternate_correct_path")
        if path_switch_signal and path_switch_signal.lower() not in detailed_hint_lower and not has_switch_label:
            validation_errors.append(f"{hint_key}: detailed_prompt_hint 未显式包含 path_switch_signal")
        guard_json = _extract_hint_guards_json(detailed_hint)
        if not isinstance(guard_json, dict):
            validation_errors.append(f"{hint_key}: 缺少可解析的 HINT_GUARDS_JSON")
        else:
            missing_guard_keys = sorted(required_guard_keys - set(guard_json.keys()))
            if missing_guard_keys:
                validation_errors.append(f"{hint_key}: HINT_GUARDS_JSON 缺少字段 {missing_guard_keys}")
        mandatory_tool_sequence = item.get("mandatory_tool_sequence")
        if isinstance(mandatory_tool_sequence, list) and task_tools:
            if not any(str(tool).strip() in task_tools for tool in mandatory_tool_sequence):
                validation_errors.append(f"{hint_key}: mandatory_tool_sequence 未覆盖任务工具")
        inferred_category = _infer_hint_category_signature(item)
        if exception_category != "Cross-Source Uncertainty" and inferred_category and exception_category and inferred_category != exception_category:
            validation_errors.append(f"{hint_key}: exception_category={exception_category} 与 failpoint/drift 签名 {inferred_category} 不一致")
    if len(seen_categories) > 1:
        validation_errors.append(f"单个任务存在多个 exception_category: {seen_categories}")
    if seen_categories == ["Cross-Source Uncertainty"] and not _task_has_cross_source_semantics(hints, task_meta):
        validation_errors.append("Cross-Source Uncertainty 缺少任务级跨来源语义信号")
    if target_exception_category and seen_categories and seen_categories != [target_exception_category]:
        validation_errors.append(f"生成的 exception_category={seen_categories} 与目标 target_exception_category={target_exception_category} 不一致")
    if not has_finish_block:
        validation_errors.append("所有 hints 均未显式阻止 premature finish")
    if not has_canonicalization:
        validation_errors.append("所有 hints 均未显式要求 final-answer canonicalization")
    if not has_retry_or_fallback:
        validation_errors.append("所有 hints 均未显式提供 retry/fallback 恢复路径")
    if not has_required_input_guard:
        validation_errors.append("所有 hints 均未提供 required_inputs 参数约束")
    if validation_errors:
        return "; ".join(validation_errors[:12])
    return None


def validate_generated_exception_code(code_content: str) -> Optional[str]:
    text = clean_code_block(code_content)
    required_snippets = {
        "strict_no_hint_profile": "缺少 strict_no_hint_profile",
        "guided_with_hint_profile": "缺少 guided_with_hint_profile",
        "INJECTION_CONFIG_JSON": "缺少 INJECTION_CONFIG_JSON 全局激活入口",
        "EXCEPTION_HINTS": "缺少 EXCEPTION_HINTS",
        "def get_exception_hints": "缺少 get_exception_hints()",
    }
    missing = [message for snippet, message in required_snippets.items() if snippet not in text]
    if missing:
        return "; ".join(missing)
    if "profile_name" not in text:
        return "缺少 profile_name 观测字段"
    if "alternate_correct_path" not in text or "path_switch_signal" not in text:
        return "缺少 alternate_correct_path/path_switch_signal 恢复语义字段"
    if "exception_category" not in text:
        return "缺少 exception_category 分类字段"
    baseline_markers = [
        "Keep original behavior unchanged when injection is disabled",
        "default to disabled mode",
        "enabled=False",
        '"enabled": False',
        "'enabled': False",
        '"probability": 0.0',
        "'probability': 0.0",
        '"max_times": 0',
        "'max_times': 0",
    ]
    if not any(marker in text for marker in baseline_markers):
        return "缺少 baseline mode preservation / disabled-mode safety contract"
    return None


def build_exception_prompt(task_meta: Dict, existing_tool_code: str) -> str:
    generation_requirements = _build_generation_requirements(task_meta)
    payload = {
        "task_type": task_meta["task_type"],
        "main_topic": task_meta["main_topic"],
        "subtopic": task_meta["subtopic"],
        "id": task_meta["task_item"].get("id", "unknown"),
        "user_prompt": task_meta["task_item"].get("user_prompt", ""),
        "tools_used": task_meta["task_item"].get("tools_used", []),
        "final_answer": task_meta["task_item"].get("final_answer", ""),
        "generation_requirements": generation_requirements,
        "existing_tool_code": existing_tool_code,
    }
    extra_instruction = {
        "must_cover": [
            "baseline correctness preservation when injection is disabled",
            "strict-no-hint failure contrast that can reduce average success",
            "with-hint recoverability that can improve average success over strict-no-hint",
            "tool-level exception/drift recovery",
            "premature finish blocking",
            "verification of suspicious/wrong intermediate tool results",
            "final-answer canonicalization to exact expected_answer form",
        ],
        "exact_expected_answer": generation_requirements["expected_answer"],
        "canonical_answer_rules": generation_requirements["canonical_answer_rules"],
        "evaluation_objective": {
            "baseline_tools_should_match_expected_answer": True,
            "strict_no_hint_should_lower_average_success": True,
            "deferred_and_from_start_hints_should_raise_average_success": True,
        },
    }
    return (
        f"{prompt_genexception}\n\n"
        f"# Additional Generation Emphasis\n{json.dumps(extra_instruction, ensure_ascii=False, indent=2)}\n\n"
        f"# Runtime Input\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def collect_existing_success_tasks(
    input_dir: str,
    tools_dir: str,
    task_types: List[str],
    include_patterns: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    baseline_success_task_keys: Optional[Set[str]] = None,
) -> List[Dict]:
    all_tasks: List[Dict] = []
    normalized_patterns = [p.strip() for p in (include_patterns or []) if str(p).strip()]
    for tt in task_types:
        tt_dir = os.path.join(input_dir, tt)
        if not os.path.exists(tt_dir):
            continue
        for mt_dir in os.listdir(tt_dir):
            mt_path = os.path.join(tt_dir, mt_dir)
            if not os.path.isdir(mt_path):
                continue
            for filename in os.listdir(mt_path):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(mt_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "tasks" not in data or not isinstance(data["tasks"], list):
                        continue
                    subtopic = filename.replace(".json", "")
                    relative_task_file = f"{tt}/{mt_dir}/{filename}"
                    for task_item in data["tasks"]:
                        task_id = task_item.get("id", "unknown")
                        source_tool_path = build_tool_path(tools_dir, tt, mt_dir, subtopic, task_id, exception_category=None)
                        if not os.path.exists(source_tool_path):
                            continue
                        task_key = build_task_key(tt, mt_dir, subtopic, task_id)
                        if baseline_success_task_keys is not None and task_key not in baseline_success_task_keys:
                            continue
                        if normalized_patterns and not any(
                            fnmatch.fnmatch(relative_task_file, pattern)
                            or fnmatch.fnmatch(task_key, pattern)
                            for pattern in normalized_patterns
                        ):
                            continue
                        if checkpoint is not None and checkpoint.is_completed(task_key):
                            continue
                        all_tasks.append(
                            {
                                "task_key": task_key,
                                "task_item": task_item,
                                "task_type": tt,
                                "main_topic": mt_dir,
                                "subtopic": subtopic,
                                "source_filepath": source_tool_path,
                            }
                        )
                except Exception:
                    continue
    all_tasks.sort(key=lambda item: str(item.get("task_key", "")))
    if max_tasks is not None and max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]
    return all_tasks


def generate_single_exception_tool(
    task_meta: Dict,
    output_dir: str,
    retry_count: int,
    rate_limit_delay: float,
) -> GenerationResult:
    start_time = time.time()
    task_id = task_meta["task_item"].get("id", "unknown")
    task_key = task_meta["task_key"]
    source_filepath = task_meta["source_filepath"]
    try:
        if checkpoint is not None:
            if checkpoint.paused:
                return GenerationResult(
                    task_key=task_key,
                    task_id=task_id,
                    source_filepath=source_filepath,
                    filepath="",
                    success=False,
                    error="任务已暂停",
                    duration=0.0,
                )
            if checkpoint.api_quota_exhausted:
                return GenerationResult(
                    task_key=task_key,
                    task_id=task_id,
                    source_filepath=source_filepath,
                    filepath="",
                    success=False,
                    error="API 配额已用尽",
                    duration=0.0,
                )
        with open(source_filepath, "r", encoding="utf-8") as f:
            existing_tool_code = f.read()
        llm = get_thread_llm()
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        base_prompt = build_exception_prompt(task_meta, existing_tool_code)
        code_response = None
        last_error: Optional[str] = None
        for attempt in range(retry_count + 1):
            try:
                if checkpoint is not None and (checkpoint.paused or checkpoint.api_quota_exhausted):
                    return GenerationResult(
                        task_key=task_key,
                        task_id=task_id,
                        source_filepath=source_filepath,
                        filepath="",
                        success=False,
                        error="任务中断",
                        duration=time.time() - start_time,
                    )
                code_response = llm.get_completion(base_prompt)
                if code_response and not str(code_response).startswith("Error:"):
                    hints, hint_error = extract_exception_hints(code_response)
                    if isinstance(hints, dict):
                        break
                    last_error = hint_error or "EXCEPTION_HINTS 缺失"
                    code_response = None
                    if attempt < retry_count:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    break
                last_error = str(code_response)[:200]
                if is_quota_error(last_error):
                    if checkpoint is not None:
                        checkpoint.set_quota_exhausted(True)
                    return GenerationResult(
                        task_key=task_key,
                        task_id=task_id,
                        source_filepath=source_filepath,
                        filepath="",
                        success=False,
                        error=f"API 配额已用尽: {last_error}",
                        duration=time.time() - start_time,
                    )
            except Exception as e:
                last_error = str(e)[:200]
                if is_quota_error(last_error):
                    if checkpoint is not None:
                        checkpoint.set_quota_exhausted(True)
                    return GenerationResult(
                        task_key=task_key,
                        task_id=task_id,
                        source_filepath=source_filepath,
                        filepath="",
                        success=False,
                        error=f"API 配额已用尽: {last_error}",
                        duration=time.time() - start_time,
                    )
                if attempt < retry_count:
                    time.sleep(1.0 * (attempt + 1))
        if not code_response or str(code_response).startswith("Error:"):
            return GenerationResult(
                task_key=task_key,
                task_id=task_id,
                source_filepath=source_filepath,
                filepath="",
                success=False,
                error=f"LLM 生成失败: {last_error}",
                duration=time.time() - start_time,
            )
        hints, hint_error = extract_exception_hints(code_response)
        target_exception_category = str(task_meta.get("target_exception_category") or "").strip()
        if not isinstance(hints, dict):
            return GenerationResult(
                task_key=task_key,
                task_id=task_id,
                source_filepath=source_filepath,
                filepath="",
                success=False,
                exception_category=None,
                error=hint_error or "无法解析 EXCEPTION_HINTS",
                duration=time.time() - start_time,
                exception_hints=hints,
                hint_error=hint_error,
            )
        exception_category = target_exception_category or _extract_task_exception_category(hints) or "Execution Uncertainty"
        hints = _force_assigned_exception_category(hints, exception_category) if exception_category else hints

        success, output_filepath = save_generated_code(
            code_response,
            output_dir,
            task_meta["task_type"],
            exception_category,
            task_meta["main_topic"],
            task_meta["subtopic"],
            task_id,
        )
        if not success:
            return GenerationResult(
                task_key=task_key,
                task_id=task_id,
                source_filepath=source_filepath,
                filepath="",
                success=False,
                exception_category=exception_category,
                error=f"保存失败: {output_filepath}",
                duration=time.time() - start_time,
            )
        hint_filepath = None
        if isinstance(hints, dict):
            saved, hint_result = save_hint_json(hints, output_filepath)
            if saved:
                hint_filepath = hint_result
            else:
                hint_error = f"{hint_error}; 保存 hint 失败: {hint_result}" if hint_error else f"保存 hint 失败: {hint_result}"

        return GenerationResult(
            task_key=task_key,
            task_id=task_id,
            source_filepath=source_filepath,
            filepath=output_filepath,
            success=True,
            exception_category=exception_category,
            duration=time.time() - start_time,
            exception_hints=hints,
            hint_error=hint_error,
            hint_filepath=hint_filepath,
        )
    except Exception as e:
        return GenerationResult(
            task_key=task_key,
            task_id=task_id,
            source_filepath=source_filepath,
            filepath="",
            success=False,
            error=str(e),
            duration=time.time() - start_time,
        )


def genexception_parallel(
    input_dir: str = "tasks",
    tools_dir: str = "tools",
    output_dir: str = "tools_exception",
    task_types: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    max_workers: int = 5,
    retry_count: int = 2,
    rate_limit_delay: float = 0.5,
    checkpoint_path: Optional[str] = None,
    reset_checkpoint: bool = False,
    baseline_results: Optional[str] = None,
) -> Dict:
    global checkpoint
    if task_types is None:
        task_types = ["sequential", "parallel", "mixture"]

    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    checkpoint = CheckpointManager(checkpoint_path)

    if reset_checkpoint:
        checkpoint.reset()

    stats = checkpoint.get_stats()
    print(f"📂 当前进度：{stats['completed']} 个任务已完成")
    if stats["api_quota_exhausted"]:
        print("⚠️ API 配额标记为已用尽，将跳过所有任务")
    if stats["paused"]:
        checkpoint.paused = False
        checkpoint.save()

    baseline_success_task_keys = load_baseline_success_task_keys(baseline_results)
    baseline_success_candidate_count = len(baseline_success_task_keys) if baseline_success_task_keys is not None else None

    phase_started = time.time()
    print("🔍 阶段 1/3: 开始收集可用 tasks...")
    all_tasks = collect_existing_success_tasks(
        input_dir,
        tools_dir,
        task_types,
        include_patterns=include_patterns,
        max_tasks=None,
        baseline_success_task_keys=baseline_success_task_keys,
    )
    print(f"✅ 阶段 1/3 完成: 收集到 {len(all_tasks)} 个 runnable tasks, 用时 {time.time() - phase_started:.2f}s")

    phase_started = time.time()
    print("🗂️ 阶段 2/3: 开始为每个 task 分配兼容的 exception_category...")
    all_tasks = assign_balanced_exception_categories(all_tasks, max_tasks=max_tasks)
    print(f"✅ 阶段 2/3 完成: 选出 {len(all_tasks)} 个 tasks, 用时 {time.time() - phase_started:.2f}s")

    phase_started = time.time()
    print("📚 阶段 3/3: 开始重建已有 hints catalog...")
    existing_hints_catalog = rebuild_hint_catalog_from_output_dir(output_dir)
    print(f"✅ 阶段 3/3 完成: catalog 中已有 {len(existing_hints_catalog)} 个 hints, 用时 {time.time() - phase_started:.2f}s")
    if not all_tasks:
        summary = {
            "total": stats["completed"],
            "success": stats["completed"],
            "failed": 0,
            "success_rate": "100.0%" if stats["completed"] > 0 else "0%",
            "results": [],
            "checkpoint": stats,
            "hints_catalog_count": len(existing_hints_catalog),
            "hint_parse_issues": [],
            "baseline_success_candidate_count": baseline_success_candidate_count,
            "baseline_success_selected_count": 0,
        }
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "genexception_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        hints_path = os.path.join(output_dir, "exception_hints_catalog.json")
        with open(hints_path, "w", encoding="utf-8") as f:
            json.dump(existing_hints_catalog, f, ensure_ascii=False, indent=2)
        return summary
    results: List[GenerationResult] = []
    success_count = 0
    failed_count = 0
    completed_count = 0
    recent_task = "-"
    with ThreadPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        future_to_task = {
            executor.submit(
                generate_single_exception_tool,
                task,
                output_dir,
                retry_count,
                rate_limit_delay,
            ): task
            for task in all_tasks
        }
        with tqdm(
            total=len(all_tasks),
            desc="exception",
            unit="task",
            dynamic_ncols=True,
            leave=True,
            mininterval=1.0,
        ) as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed_count += 1
                if result.success:
                    success_count += 1
                else:
                    failed_count += 1
                category_label = (result.exception_category or "NA").replace(" Uncertainty", "")
                recent_task = f"{result.task_id}:{category_label}"[:24]
                checkpoint.record_result(
                    result.task_key,
                    {
                        "task_key": result.task_key,
                        "task_id": result.task_id,
                        "source_filepath": result.source_filepath,
                        "filepath": result.filepath,
                        "exception_category": result.exception_category,
                        "success": result.success,
                        "error": result.error,
                        "duration": round(result.duration, 2),
                        "exception_hints": result.exception_hints,
                        "hint_error": result.hint_error,
                        "hint_filepath": result.hint_filepath,
                    },
                )
                running_count = max(0, len(all_tasks) - completed_count)
                pbar.set_postfix(
                    {
                        "ok": success_count,
                        "fail": failed_count,
                        "last": recent_task,
                    },
                    refresh=False,
                )
                pbar.update(1)
                if checkpoint.api_quota_exhausted:
                    print("\n⚠️ API 配额已用尽，停止后续任务")
                    break
    summary = {
        "total": len(all_tasks) + stats["completed"],
        "success": success_count + stats["completed"],
        "failed": failed_count,
        "success_rate": f"{(success_count + stats['completed']) / max(1, len(all_tasks) + stats['completed']) * 100:.1f}%",
        "results": [
            {
                "task_key": r.task_key,
                "task_id": r.task_id,
                "source_filepath": r.source_filepath,
                "filepath": r.filepath,
                "exception_category": r.exception_category,
                "success": r.success,
                "error": r.error,
                "duration": round(r.duration, 2),
                "hint_error": r.hint_error,
                "hint_filepath": r.hint_filepath,
            }
            for r in results
        ],
        "checkpoint": checkpoint.get_stats(),
        "baseline_success_candidate_count": baseline_success_candidate_count,
        "baseline_success_selected_count": len(all_tasks),
    }

    planned_category_counts: Dict[str, int] = {category: 0 for category in VALID_EXCEPTION_CATEGORIES}
    compatible_pool_counts: Dict[str, int] = {category: 0 for category in VALID_EXCEPTION_CATEGORIES}
    for task in all_tasks:
        category = str(task.get("target_exception_category") or "").strip()
        if category in planned_category_counts:
            planned_category_counts[category] += 1
    for task in collect_existing_success_tasks(
        input_dir,
        tools_dir,
        task_types,
        include_patterns=include_patterns,
        max_tasks=None,
        baseline_success_task_keys=baseline_success_task_keys,
    ):
        for category in _compatible_exception_categories(task):
            if category in compatible_pool_counts:
                compatible_pool_counts[category] += 1

    generated_category_counts: Dict[str, int] = {category: 0 for category in VALID_EXCEPTION_CATEGORIES}
    for r in results:
        category = str(r.exception_category or "").strip()
        if category in generated_category_counts:
            generated_category_counts[category] += 1

    summary["planned_category_counts"] = planned_category_counts
    summary["compatible_pool_counts"] = compatible_pool_counts
    summary["generated_category_counts"] = generated_category_counts

    hints_catalog = rebuild_hint_catalog_from_output_dir(output_dir)

    hint_parse_issues = [
        {
            "task_key": r.task_key,
            "task_id": r.task_id,
            "hint_error": r.hint_error,
            "filepath": r.filepath,
        }
        for r in results
        if r.success and r.hint_error
    ]

    summary["hints_catalog_count"] = len(hints_catalog)
    summary["hint_parse_issues"] = hint_parse_issues
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "genexception_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    hints_path = os.path.join(output_dir, "exception_hints_catalog.json")
    with open(hints_path, "w", encoding="utf-8") as f:
        json.dump(hints_catalog, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="并行生成带异常注入的工具代码")
    parser.add_argument("--input-dir", type=str, default="tasks")
    parser.add_argument("--tools-dir", type=str, default="tools")
    parser.add_argument("--output-dir", type=str, default="tools_exception")
    parser.add_argument("--include-pattern", action="append", default=[], help="Glob on relative task json path or task_key; repeatable")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of selected task items after filtering")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--rate-limit-delay", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--baseline-results", type=str, default=None, help="Only inject exceptions for task keys that succeeded in this baseline results JSON")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    result = genexception_parallel(
        input_dir=args.input_dir,
        tools_dir=args.tools_dir,
        output_dir=args.output_dir,
        task_types=["sequential", "parallel", "mixture"],
        include_patterns=args.include_pattern,
        max_tasks=args.max_tasks,
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        rate_limit_delay=args.rate_limit_delay,
        checkpoint_path=args.checkpoint_path,
        reset_checkpoint=args.reset,
        baseline_results=args.baseline_results,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
