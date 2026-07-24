import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .loader import discover_tool_path, load_tool_module
from .llm_client import LLMClient
from .parser import build_openai_tool_schema, validate_tool_args, format_tool_signature
from .utils import logger


class ModelDrivenToolAgent:
    ORACLE_HAZARD_DEFINITIONS = {
        "Specification_Uncertainty": (
            "The expected tool contract, schema, field meaning, or units may not match runtime behavior."
        ),
        "Invocation_Uncertainty": (
            "The selected tool or its arguments may be invalid, incomplete, malformed, or inappropriate."
        ),
        "Execution_Uncertainty": (
            "The tool or provider may have failed during execution, such as through timeout, connection, I/O, or service failure."
        ),
        "Output_Uncertainty": (
            "The returned payload may be malformed, incomplete, corrupted, wrongly typed, or unsafe to consume."
        ),
        "Cross-Source_Uncertainty": (
            "Evidence from multiple tools, sources, or branches may conflict and require reconciliation."
        ),
    }
    MODEL_VISIBLE_ANSWER_KEYS = {
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
    MODEL_VISIBLE_REDACTION = "[REDACTED_FOR_TEST_TIME_SCALING]"

    def __init__(
        self,
        tools_dir: Path,
        llm: Optional[LLMClient] = None,
        max_rounds: int = 30,
        enforce_expected_match: bool = True,
    ):
        self.tools_dir = tools_dir
        self.llm = llm or LLMClient()
        self.max_rounds = max(1, max_rounds)
        self.enforce_expected_match = enforce_expected_match
        self._maybe_enable_default_exception_injection()

    @staticmethod
    def _normalize_tool_specs(task: Dict[str, Any]) -> List[Dict[str, str]]:
        raw_tools = task.get("tools_used", [])
        normalized: List[Dict[str, str]] = []
        for item in raw_tools:
            if isinstance(item, str):
                normalized.append({"tool_name": item, "type": "sequential"})
            elif isinstance(item, dict) and isinstance(item.get("tool_name"), str):
                normalized.append({"tool_name": item["tool_name"], "type": str(item.get("type") or "sequential")})
            else:
                raise ValueError(f"Invalid tool spec: {item}")
        return normalized

    @staticmethod
    def _infer_task_type(task: Dict[str, Any], tool_specs: List[Dict[str, str]]) -> str:
        relative_path = str(task.get("_meta", {}).get("relative_path", ""))
        if relative_path:
            first_part = Path(relative_path).parts[0]
            if first_part in {"sequential", "parallel", "mixture"}:
                return first_part
        tool_types = {spec["type"] for spec in tool_specs}
        if "parallel" in tool_types:
            return "mixture" if "sequential" in tool_types else "parallel"
        return "sequential"

    @staticmethod
    def _is_expected_match(expected: Any, actual: Any) -> bool:
        if expected is None:
            return True
        return str(actual).strip() == str(expected).strip()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}

    @classmethod
    def _scrub_model_visible_answers(cls, value: Any) -> Any:
        if isinstance(value, dict):
            scrubbed: Dict[str, Any] = {}
            for key, child in value.items():
                if str(key).lower() in cls.MODEL_VISIBLE_ANSWER_KEYS:
                    continue
                else:
                    scrubbed[key] = cls._scrub_model_visible_answers(child)
            return scrubbed
        if isinstance(value, list):
            return [cls._scrub_model_visible_answers(item) for item in value]
        return value

    @staticmethod
    def _build_runtime_hint_block(deferred_hint: Optional[Dict[str, str]]) -> str:
        if not deferred_hint:
            return ""
        payload = {
            "hint_key": str(deferred_hint.get("hint_key", "")).strip(),
            "hint_strategy": str(deferred_hint.get("retry_strategy", "")).strip(),
            "minimal_prompt_hint": str(deferred_hint.get("minimal_prompt_hint", "")).strip(),
            "detailed_prompt_hint": str(deferred_hint.get("detailed_prompt_hint", "")).strip(),
            "mandatory_tool_sequence": deferred_hint.get("mandatory_tool_sequence") or [],
            "answer_fields_required": deferred_hint.get("answer_fields_required") or [],
            "forbidden_early_finish_when": deferred_hint.get("forbidden_early_finish_when") or [],
            "canonical_answer_rules": deferred_hint.get("canonical_answer_rules") or [],
            "required_inputs": deferred_hint.get("required_inputs") or {},
            "verification_checks": deferred_hint.get("verification_checks") or [],
        }
        return "\n\n[RECOVERY_HINT_CONTEXT]\n" + json.dumps(payload, ensure_ascii=False) + "\n[/RECOVERY_HINT_CONTEXT]"

    @staticmethod
    def _build_oracle_hazard_block(oracle_hazard_label: Optional[str]) -> str:
        label = str(oracle_hazard_label or "").strip()
        if not label:
            return ""
        definition = ModelDrivenToolAgent.ORACLE_HAZARD_DEFINITIONS.get(
            label,
            "Use the category name as task-level diagnostic context.",
        )
        return (
            "\n\n[ORACLE_HAZARD_LABEL]\n"
            f"The observed anomaly belongs to the following task-level hazard category: {label}.\n"
            f"Category definition: {definition}\n"
            "No recovery procedure or answer information is provided.\n"
            "[/ORACLE_HAZARD_LABEL]"
        )

    def _build_runtime_user_prompt(
        self,
        user_prompt: str,
        deferred_hint: Optional[Dict[str, str]],
        oracle_hazard_label: Optional[str] = None,
    ) -> str:
        return (
            str(user_prompt)
            + self._build_runtime_hint_block(deferred_hint)
            + self._build_oracle_hazard_block(oracle_hazard_label)
        )

    @staticmethod
    def _merge_hint_payloads(*hint_candidates: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        list_fields = {
            "mandatory_tool_sequence",
            "answer_fields_required",
            "forbidden_early_finish_when",
            "canonical_answer_rules",
            "verification_checks",
        }
        dict_fields = {"required_inputs"}
        text_fields = {"minimal_prompt_hint", "detailed_prompt_hint", "retry_strategy", "hint_key"}
        for hint in hint_candidates:
            if not isinstance(hint, dict):
                continue
            for field in text_fields:
                value = str(hint.get(field) or "").strip()
                if value:
                    if field in {"minimal_prompt_hint", "detailed_prompt_hint"} and merged.get(field):
                        existing = str(merged.get(field) or "").strip()
                        if value not in existing:
                            merged[field] = existing + "\n" + value
                    else:
                        merged[field] = value
            for field in list_fields:
                existing = merged.setdefault(field, [])
                for item in hint.get(field, []) or []:
                    text = str(item).strip()
                    if text and text not in existing:
                        existing.append(text)
            for field in dict_fields:
                existing_map = merged.setdefault(field, {})
                raw_map = hint.get(field, {}) or {}
                if not isinstance(raw_map, dict):
                    continue
                for key, values in raw_map.items():
                    key_text = str(key).strip()
                    if not key_text:
                        continue
                    dest = existing_map.setdefault(key_text, [])
                    for value in values or []:
                        item_text = str(value).strip()
                        if item_text and item_text not in dest:
                            dest.append(item_text)
        return merged or None

    def _merge_failure_hint_into_context(
        self,
        task: Dict[str, Any],
        current_hint: Optional[Dict[str, Any]],
        failed_tool: str,
        failure_text: str,
    ) -> Optional[Dict[str, Any]]:
        selected_hint = self._select_hint_for_failure(task, failed_tool, failure_text)
        if not selected_hint:
            return current_hint
        base_hint = current_hint or self._build_start_hint(task)
        return self._merge_hint_payloads(base_hint, selected_hint)

    def _apply_tool_runtime_context(self, module: Any, user_prompt: str, deferred_hint: Optional[Dict[str, str]]) -> None:
        runtime_user_prompt = self._build_runtime_user_prompt(user_prompt, deferred_hint)
        os.environ["USER_PROMPT"] = runtime_user_prompt
        os.environ["INJECTION_USER_PROMPT"] = runtime_user_prompt
        if hasattr(module, "_USER_PROMPT"):
            try:
                setattr(module, "_USER_PROMPT", runtime_user_prompt)
            except Exception:
                pass
        if callable(getattr(module, "_parse_hint_context", None)) and hasattr(module, "_HINT_CONTEXT"):
            try:
                setattr(module, "_HINT_CONTEXT", module._parse_hint_context(runtime_user_prompt))
            except Exception:
                pass
        if "tools_exception" not in str(self.tools_dir):
            return
        profile_name = "guided_with_hint_profile" if deferred_hint else "strict_no_hint_profile"
        os.environ["INJECTION_PROFILE"] = profile_name
        strict_profile = getattr(module, "strict_no_hint_profile", None)
        guided_profile = getattr(module, "guided_with_hint_profile", None)
        if callable(getattr(module, "set_injection_config", None)) and isinstance(strict_profile, dict) and isinstance(guided_profile, dict):
            profile_config = {
                "profile": profile_name,
                "profile_name": profile_name,
                "profiles": {
                    "strict_no_hint_profile": strict_profile,
                    "guided_with_hint_profile": guided_profile,
                },
            }
            module.set_injection_config(profile_config)
            os.environ["INJECTION_CONFIG_JSON"] = json.dumps(profile_config, ensure_ascii=False)

    def _maybe_enable_default_exception_injection(self) -> None:
        tools_path = str(self.tools_dir)
        if "tools_exception" not in tools_path:
            return
        if os.getenv("DISABLE_DEFAULT_EXCEPTION_INJECTION", "0").lower() in {"1", "true", "yes", "on"}:
            return
        has_global_config = bool(os.getenv("INJECTION_CONFIG_JSON"))
        has_failpoint_env = any(k.startswith("INJECT_") for k in os.environ.keys())
        if has_global_config or has_failpoint_env:
            return
        default_config = {
            "before_tool_logic": {
                "enabled": True,
                "probability": 1.0,
                "max_times": 1,
                "exception_type": "RuntimeError",
                "drift_mode": None,
            },
            "before_external_call": {
                "enabled": True,
                "probability": 0.5,
                "max_times": 1,
                "exception_type": "TimeoutError",
                "drift_mode": None,
            },
            "after_external_call_before_parse": {
                "enabled": True,
                "probability": 0.3,
                "max_times": 1,
                "exception_type": "ValueError",
                "drift_mode": "shape_drift",
            },
            "before_return": {
                "enabled": True,
                "probability": 0.25,
                "max_times": 1,
                "exception_type": "RuntimeError",
                "drift_mode": "rename_fields",
            },
        }
        os.environ["INJECTION_CONFIG_JSON"] = json.dumps(default_config, ensure_ascii=False)

    def _load_task_tools(self, task: Dict[str, Any], tool_names: List[str]) -> Any:
        meta = task.get("_meta", {})
        source_file = meta.get("source_file")
        relative_path = meta.get("relative_path")
        task_id = task.get("id")
        if not source_file:
            for tool_name in tool_names:
                for path in self.tools_dir.rglob(f"{tool_name}.py"):
                    return load_tool_module(path)
            raise ValueError(f"Cannot find tools: {tool_names}")
        effective_relative_path = str(relative_path) if relative_path else Path(source_file).name
        tool_path = discover_tool_path(self.tools_dir, effective_relative_path, str(task_id) if task_id else None)
        if tool_path and tool_path.exists():
            return load_tool_module(tool_path)
        if effective_relative_path:
            raise ValueError(f"Cannot find tool module for task path '{effective_relative_path}' and task_id '{task_id}'")
        for tool_name in tool_names:
            for tool_file in self.tools_dir.rglob(f"{tool_name}.py"):
                return load_tool_module(tool_file)
        raise ValueError(f"Cannot find tool module for task: {source_file}")

    def _build_tool_runtime_catalog(self, module: Any, tool_names: List[str]) -> Dict[str, Dict[str, Any]]:
        catalog: Dict[str, Dict[str, Any]] = {}
        for tool_name in tool_names:
            fn = getattr(module, tool_name, None)
            if fn is None:
                continue
            signature = inspect.signature(fn)
            openai_tool = build_openai_tool_schema(fn, tool_name)
            function_def = openai_tool.get("function", {}) if isinstance(openai_tool, dict) else {}
            parameters_schema = function_def.get("parameters", {"type": "object", "properties": {}, "additionalProperties": False})
            description = str(function_def.get("description") or f"Execute {tool_name}")
            catalog[tool_name] = {
                "fn": fn,
                "signature": signature,
                "signature_str": format_tool_signature(signature),
                "description": description,
                "parameters_schema": parameters_schema,
            }
        return catalog

    def _select_hint_for_failure(self, task: Dict[str, Any], failed_tool: str, failed_error: str) -> Optional[Dict[str, Any]]:
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        catalog_entry = meta.get("ab_hint_catalog_entry")
        if not isinstance(catalog_entry, dict) or not catalog_entry:
            return None
        fp_match = re.search(r"(?:at\s+)([a-zA-Z0-9_]+)(?:::(?:|\s*)([a-zA-Z0-9_]+))?", failed_error)
        failpoint = ""
        if fp_match:
            if fp_match.group(2):
                failpoint = fp_match.group(2)
            elif fp_match.group(1) and fp_match.group(1) != failed_tool:
                failpoint = fp_match.group(1)
        preferred_key = f"{failed_tool}::{failpoint}" if failpoint else ""
        chosen_key = ""
        chosen_item: Optional[Dict[str, Any]] = None
        if preferred_key and isinstance(catalog_entry.get(preferred_key), dict):
            chosen_key = preferred_key
            chosen_item = catalog_entry.get(preferred_key)
        if chosen_item is None:
            tool_prefix = f"{failed_tool}::"
            for key in sorted(catalog_entry.keys()):
                item = catalog_entry.get(key)
                if not isinstance(item, dict):
                    continue
                if str(key).startswith(tool_prefix):
                    chosen_key = str(key)
                    chosen_item = item
                    break
        if chosen_item is None:
            return None
        minimal_hint = str(chosen_item.get("minimal_prompt_hint", "")).strip()
        detailed_hint = str(chosen_item.get("detailed_prompt_hint", "")).strip()
        retry_strategy = str(chosen_item.get("retry_strategy", "")).strip()
        return {
            "hint_key": chosen_key,
            "minimal_prompt_hint": minimal_hint,
            "detailed_prompt_hint": detailed_hint,
            "retry_strategy": retry_strategy,
            "mandatory_tool_sequence": chosen_item.get("mandatory_tool_sequence") or [],
            "answer_fields_required": chosen_item.get("answer_fields_required") or [],
            "forbidden_early_finish_when": chosen_item.get("forbidden_early_finish_when") or [],
            "canonical_answer_rules": chosen_item.get("canonical_answer_rules") or [],
            "required_inputs": chosen_item.get("required_inputs") or {},
            "verification_checks": chosen_item.get("verification_checks") or [],
        }

    @staticmethod
    def _tool_execution_issue_text(tool_execution: Dict[str, Any]) -> Optional[str]:
        if not isinstance(tool_execution, dict):
            return None
        if not tool_execution.get("success"):
            text = str(tool_execution.get("error") or "tool execution failed").strip()
            return text or "tool execution failed"
        result = tool_execution.get("result")
        if not isinstance(result, dict):
            return None
        result_error = str(result.get("error") or "").strip()
        if result.get("ok") is False and result_error:
            return result_error
        if result.get("ok") is False:
            return "tool result marked ok=false"
        if result_error:
            return result_error
        return None

    @staticmethod
    def _has_exception_tooling(tools_dir: Path) -> bool:
        return "tools_exception" in str(tools_dir)

    @staticmethod
    def _unguided_retry_consumed(
        tool_executions: List[Dict[str, Any]], tool_name: str
    ) -> bool:
        same_tool = [
            item
            for item in tool_executions
            if isinstance(item, dict) and item.get("tool_name") == tool_name
        ]
        first_issue = next(
            (
                index
                for index, item in enumerate(same_tool)
                if ModelDrivenToolAgent._tool_execution_issue_text(item)
            ),
            None,
        )
        if first_issue is None:
            return False
        return len(same_tool) > first_issue + 1

    @staticmethod
    def _successful_result_payloads(tool_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for item in tool_results.values():
            if not isinstance(item, dict) or not item.get("success"):
                continue
            result = item.get("result")
            if isinstance(result, dict):
                payloads.append(result)
        return payloads

    def _build_start_hint(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        catalog_entry = meta.get("ab_hint_catalog_entry")
        if not isinstance(catalog_entry, dict) or not catalog_entry:
            return None
        verification_checks: List[str] = []
        canonical_rules: List[str] = []
        finish_blocks: List[str] = []
        required_inputs: List[str] = []
        required_inputs_map: Dict[str, List[str]] = {}
        mandatory_tools: List[str] = []
        answer_fields_required: List[str] = []
        for key in sorted(catalog_entry.keys()):
            item = catalog_entry.get(key)
            if not isinstance(item, dict):
                continue
            for value in item.get("verification_checks", []) or []:
                text = str(value).strip()
                if text and text not in verification_checks:
                    verification_checks.append(text)
            for value in item.get("canonical_answer_rules", []) or []:
                text = str(value).strip()
                if text and text not in canonical_rules:
                    canonical_rules.append(text)
            for value in item.get("forbidden_early_finish_when", []) or []:
                text = str(value).strip()
                if text and text not in finish_blocks:
                    finish_blocks.append(text)
            for tool_name, args in (item.get("required_inputs", {}) or {}).items():
                arg_list = ", ".join(str(arg).strip() for arg in args if str(arg).strip())
                text = f"{tool_name}: {arg_list}" if arg_list else str(tool_name).strip()
                if text and text not in required_inputs:
                    required_inputs.append(text)
                tool_key = str(tool_name).strip()
                if tool_key:
                    dest = required_inputs_map.setdefault(tool_key, [])
                    for arg in args or []:
                        arg_text = str(arg).strip()
                        if arg_text and arg_text not in dest:
                            dest.append(arg_text)
            for tool_name in item.get("mandatory_tool_sequence", []) or []:
                text = str(tool_name).strip()
                if text and text not in mandatory_tools:
                    mandatory_tools.append(text)
            for field in item.get("answer_fields_required", []) or []:
                field_text = str(field).strip()
                if field_text and field_text not in answer_fields_required:
                    answer_fields_required.append(field_text)
        if not (verification_checks or canonical_rules or finish_blocks or mandatory_tools or answer_fields_required):
            return None
        minimal_lines = [
            "- DO NOT FINISH UNTIL required evidence fields are present and suspicious tool outputs are verified.",
            "- If a tool result conflicts with other evidence or required fields are missing, keep calling mandatory tools instead of finishing.",
            "- Before FINISH, prefer exact tool final_value and canonicalize the answer to benchmark form only.",
        ]
        details: List[str] = [
            "STEP 1 CHECK",
            "CHECK mandatory tool coverage, required inputs, and contradiction signals before trusting any candidate answer.",
        ]
        if mandatory_tools:
            details.append(f"Mandatory tool order: {', '.join(mandatory_tools)}.")
        if required_inputs:
            details.append(f"Required-input reminders: {'; '.join(required_inputs[:6])}.")
        details.extend([
            "STEP 2 ACTION",
            "If evidence is incomplete or a tool output looks suspicious, continue the mandatory tool path or re-validate with downstream evidence instead of finishing early.",
        ])
        if verification_checks:
            details.append(f"Verification checks: {'; '.join(verification_checks[:6])}.")
        details.extend([
            "STEP 3 STOP/FINISH GATE",
            "BLOCK FINISH when answer fields are missing, contradictory, or only available in non-canonical wording.",
        ])
        if finish_blocks:
            details.append(f"Early-finish blockers: {'; '.join(finish_blocks[:6])}.")
        if canonical_rules:
            details.append(f"Canonical answer rules: {'; '.join(canonical_rules[:6])}.")
        return {
            "hint_key": str(meta.get("ab_hint_key") or "task-wide-start-hint"),
            "minimal_prompt_hint": "\n".join(minimal_lines),
            "detailed_prompt_hint": "\n".join(details),
            "retry_strategy": "guard_finish_then_continue",
            "mandatory_tool_sequence": mandatory_tools,
            "answer_fields_required": answer_fields_required,
            "forbidden_early_finish_when": finish_blocks,
            "canonical_answer_rules": canonical_rules,
            "required_inputs": required_inputs_map,
            "verification_checks": verification_checks,
        }

    @staticmethod
    def _next_unresolved_mandatory_tool(
        allowed_tools: List[str],
        tool_results: Dict[str, Any],
        deferred_hint: Optional[Dict[str, Any]],
    ) -> str:
        preferred_tools: List[str] = []
        if isinstance(deferred_hint, dict):
            for tool_name in deferred_hint.get("mandatory_tool_sequence", []) or []:
                name = str(tool_name).strip()
                if name and name in allowed_tools and name not in preferred_tools:
                    preferred_tools.append(name)
        for tool_name in allowed_tools:
            if tool_name not in preferred_tools:
                preferred_tools.append(tool_name)
        for tool_name in preferred_tools:
            execution = tool_results.get(tool_name)
            if not isinstance(execution, dict) or not execution.get("success"):
                return tool_name
        return preferred_tools[0] if preferred_tools else ""

    def _collect_hint_guardrails(self, task: Dict[str, Any]) -> Dict[str, List[str]]:
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        catalog_entry = meta.get("ab_hint_catalog_entry")
        guardrails = {
            "mandatory_tool_sequence": [],
            "answer_fields_required": [],
            "forbidden_early_finish_when": [],
            "canonical_answer_rules": [],
        }
        if not isinstance(catalog_entry, dict):
            return guardrails
        for key in sorted(catalog_entry.keys()):
            item = catalog_entry.get(key)
            if not isinstance(item, dict):
                continue
            for field in guardrails.keys():
                for value in item.get(field, []) or []:
                    text = str(value).strip()
                    if text and text not in guardrails[field]:
                        guardrails[field].append(text)
        return guardrails

    def _find_present_answer_fields(self, tool_results: Dict[str, Any]) -> List[str]:
        present: List[str] = []
        for result in self._successful_result_payloads(tool_results):
            for key, value in result.items():
                if value in (None, "", [], {}, ()):
                    continue
                key_text = str(key).strip()
                if key_text and key_text not in present:
                    present.append(key_text)
        return present

    @staticmethod
    def _tool_order_from_runtime(runtime: Dict[str, Dict[str, Any]]) -> List[str]:
        return [name for name in runtime.keys() if str(name).strip()]

    def _default_blocks_finish(
        self,
        task: Dict[str, Any],
        runtime: Dict[str, Dict[str, Any]],
        tool_results: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        required_tools = self._tool_order_from_runtime(runtime)
        if not required_tools:
            return False, None
        preferred_scalar = self.llm.extract_final_scalar(tool_results)
        for tool_name in required_tools:
            execution = tool_results.get(tool_name)
            if not isinstance(execution, dict):
                if preferred_scalar is None:
                    return True, tool_name
                continue
            issue = self._tool_execution_issue_text(execution)
            if issue:
                return True, tool_name
        if preferred_scalar is not None:
            return False, None
        task_type = str(task.get("_meta", {}).get("relative_path", "")).split("/", 1)[0] if isinstance(task.get("_meta"), dict) else ""
        if task_type in {"sequential", "parallel", "mixture"}:
            for tool_name in required_tools:
                if tool_name not in tool_results:
                    return True, tool_name
        return False, None

    def _hint_blocks_finish(self, task: Dict[str, Any], tool_results: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        if not meta.get("ab_hint_catalog_entry"):
            return False, None
        guardrails = self._collect_hint_guardrails(task)
        mandatory_tools = guardrails.get("mandatory_tool_sequence", [])
        for tool_name in mandatory_tools:
            execution = tool_results.get(tool_name)
            if not isinstance(execution, dict) or not execution.get("success"):
                return True, str(tool_name)
            if self._tool_execution_issue_text(execution):
                return True, str(tool_name)
        present_fields = set(self._find_present_answer_fields(tool_results))
        for field in guardrails.get("answer_fields_required", []):
            if field not in present_fields:
                return True, None
        if not self.llm.extract_final_scalar(tool_results):
            if mandatory_tools or guardrails.get("answer_fields_required"):
                return True, mandatory_tools[0] if mandatory_tools else None
        return False, None

    @staticmethod
    def _canonical_rules_expect_atomic_token(canonical_rules: List[str]) -> bool:
        lowered_rules = "\n".join(str(rule or "").lower() for rule in canonical_rules)
        token_markers = [
            "uppercase code token only",
            "identifier token only",
            "weekday token only",
            "lowercase enum token only",
            "single lowercase enum token only",
            "single weekday token only",
            "single token only",
        ]
        return any(marker in lowered_rules for marker in token_markers)

    @staticmethod
    def _is_wrapped_answer_text(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in ["answer:", "result:", "because", "therefore", "final answer"]) 

    @staticmethod
    def _looks_like_location_phrase(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        return bool(re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", value))

    @staticmethod
    def _extract_prompt_option_set(user_prompt: str) -> List[str]:
        prompt = str(user_prompt or "")
        match = re.search(r"\bamong\s+(.+?)\s+(?:that|with|currently|which|who|where|having)\b", prompt, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\bamong\s+(.+?)[\.,]", prompt, flags=re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1)
        normalized = re.sub(r"\s+(?:and|or)\s+", ",", raw, flags=re.IGNORECASE)
        options: List[str] = []
        for part in normalized.split(","):
            text = str(part).strip().strip(".")
            if not text:
                continue
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", text):
                continue
            lowered = text.lower()
            if lowered not in options:
                options.append(lowered)
        return options if 2 <= len(options) <= 8 else []

    @staticmethod
    def _extract_prompt_option_preference(user_prompt: str) -> Optional[str]:
        prompt = str(user_prompt or "").lower()
        if "alphabetically earliest" in prompt or "alphabetical earliest" in prompt:
            return "alphabetically_earliest"
        if "alphabetically latest" in prompt or "alphabetical latest" in prompt:
            return "alphabetically_latest"
        return None

    def _extract_option_candidates_from_results(self, option_set: List[str], tool_results: Dict[str, Any]) -> List[str]:
        options = {str(item).strip().lower() for item in option_set if str(item).strip()}
        matched: List[str] = []
        for result in self._successful_result_payloads(tool_results):
            for value in result.values():
                normalized = self.llm._normalize_candidate_scalar_text(value)
                text = str(normalized or "").strip().lower()
                if text and text in options and text not in matched:
                    matched.append(text)
        return matched

    def _resolve_prompt_option_choice(self, user_prompt: str, tool_results: Dict[str, Any]) -> Optional[str]:
        option_set = self._extract_prompt_option_set(user_prompt)
        preference = self._extract_prompt_option_preference(user_prompt)
        if not option_set or not preference:
            return None
        candidates = self._extract_option_candidates_from_results(option_set, tool_results)
        if not candidates:
            return None
        return min(candidates) if preference == "alphabetically_earliest" else max(candidates)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(str(value).strip())
        except Exception:
            return None

    def _resolve_numeric_total_from_results(self, tool_results: Dict[str, Any]) -> Optional[str]:
        ordered_results = list(self._successful_result_payloads(tool_results))
        reversed_results = list(reversed(ordered_results))
        subtotal = discount = shipping = None
        observed_final = None
        for result in reversed_results:
            if observed_final is None:
                observed_final = self._coerce_float(result.get("final_total_usd"))
                if observed_final is None:
                    observed_final = self._coerce_float(result.get("final_total"))
                if observed_final is None:
                    observed_final = self._coerce_float(result.get("final_value"))
            if subtotal is None:
                subtotal = self._coerce_float(result.get("subtotal_usd"))
            if discount is None:
                discount = self._coerce_float(result.get("discount_usd"))
            if shipping is None:
                shipping = self._coerce_float(result.get("shipping_usd"))
            if subtotal is not None and discount is not None and shipping is not None:
                break
        if subtotal is None or discount is None or shipping is None:
            return None
        recomputed = round(subtotal - discount + shipping, 2)
        if observed_final is not None and abs(observed_final - recomputed) <= 0.01:
            return None
        return f"{recomputed:.2f}"

    def _resolve_prompt_numeric_choice(self, user_prompt: str, tool_results: Dict[str, Any]) -> Optional[str]:
        prompt = str(user_prompt or "").lower()
        if not any(marker in prompt for marker in ["payable amount", "final amount", "final payable", "amount in usd", "total in usd"]):
            return None
        return self._resolve_numeric_total_from_results(tool_results)

    @classmethod
    def _candidate_is_context_echo(cls, candidate: str, tool_results: Dict[str, Any]) -> bool:
        normalized = str(candidate or "").strip().lower()
        if not normalized:
            return False
        fields = {"location", "site", "city", "region", "town", "locality", "service_area", "checkout_session_id", "payment_attempt_id", "gateway_response_id", "cart_id", "order_id"}
        for result in cls._successful_result_payloads(tool_results):
            for field in fields:
                value = str(result.get(field) or "").strip().lower()
                if value and value == normalized:
                    return True
        return False

    def _candidate_rejected_by_common_guardrails(
        self,
        candidate: str,
        canonical_rules: List[str],
        tool_results: Dict[str, Any],
        user_prompt: str,
    ) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return True
        if self._is_wrapped_answer_text(text):
            return True
        option_set = self._extract_prompt_option_set(user_prompt)
        if option_set and text.lower() not in option_set:
            return True
        if self._canonical_rules_expect_atomic_token(canonical_rules):
            if self._looks_like_location_phrase(text):
                return True
            if self._candidate_is_context_echo(text, tool_results):
                return True
        return False

    @staticmethod
    def _candidate_matches_canonical_rules(candidate: str, canonical_rules: List[str]) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return False
        lowered_rules = "\n".join(str(rule or "").lower() for rule in canonical_rules)
        if not lowered_rules:
            return True
        if "numeric value only" in lowered_rules or "currency" in lowered_rules:
            if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
                return False
        if "yyyy-mm-dd" in lowered_rules:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                return False
        if "iso-8601" in lowered_rules or "trailing z" in lowered_rules:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", text.lower()):
                return False
        if "uppercase code token only" in lowered_rules:
            if not re.fullmatch(r"[A-Z]{2,}(?:[_-][A-Z0-9]+)+", text):
                return False
        if "identifier token only" in lowered_rules:
            if not re.fullmatch(r"[A-Za-z]{1,4}-\d{2,}", text):
                return False
        if "weekday token only" in lowered_rules:
            if text not in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}:
                return False
        if "no extra text" in lowered_rules or "do not add explanation" in lowered_rules:
            if len(text.split()) > 3 and not re.fullmatch(r"[A-Z]{2,}(?:[_-][A-Z0-9]+)+", text):
                return False
        return True

    def _select_best_final_candidate(
        self,
        proposed_answer: str,
        preferred_scalar: Optional[str],
        synthesized_answer: str,
        canonical_rules: List[str],
        tool_results: Dict[str, Any],
        user_prompt: str,
    ) -> str:
        candidates: List[str] = []
        for item in [preferred_scalar, synthesized_answer, proposed_answer]:
            text = str(item or "").strip()
            if text and text not in candidates:
                candidates.append(text)
        if not candidates:
            return ""
        filtered_candidates = [
            candidate
            for candidate in candidates
            if not self._candidate_rejected_by_common_guardrails(candidate, canonical_rules, tool_results, user_prompt)
        ]
        if filtered_candidates:
            candidates = filtered_candidates
        if not canonical_rules:
            return candidates[0]
        for candidate in candidates:
            if self._candidate_matches_canonical_rules(candidate, canonical_rules):
                return candidate
        return candidates[0]

    def _finalize_answer_with_hint_guardrails(
        self,
        task: Dict[str, Any],
        user_prompt: str,
        tool_results: Dict[str, Any],
        proposed_answer: str,
    ) -> str:
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        hint_active = bool(meta.get("ab_hint_catalog_entry"))
        canonical_rules = []
        if hint_active:
            canonical_rules = self._collect_hint_guardrails(task).get("canonical_answer_rules", [])
        preferred_scalar = self.llm.extract_final_scalar(tool_results)
        prompt_resolved_scalar = self._resolve_prompt_option_choice(user_prompt, tool_results)
        if prompt_resolved_scalar:
            preferred_scalar = prompt_resolved_scalar
        numeric_prompt_scalar = self._resolve_prompt_numeric_choice(user_prompt, tool_results)
        if numeric_prompt_scalar:
            preferred_scalar = numeric_prompt_scalar
        synthesized = self.llm.synthesize_final_answer(user_prompt, tool_results)
        synthesized = str(synthesized or "").strip()
        if not hint_active:
            return self._select_best_final_candidate(
                proposed_answer=str(proposed_answer or "").strip(),
                preferred_scalar=preferred_scalar,
                synthesized_answer=synthesized,
                canonical_rules=[],
                tool_results=tool_results,
                user_prompt=user_prompt,
            )
        return self._select_best_final_candidate(
            proposed_answer=str(proposed_answer or "").strip(),
            preferred_scalar=preferred_scalar,
            synthesized_answer=synthesized,
            canonical_rules=canonical_rules,
            tool_results=tool_results,
            user_prompt=user_prompt,
        )

    @staticmethod
    def _build_openai_trajectory(user_prompt: str, tool_executions: List[Dict[str, Any]], final_answer: Any) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        for index, item in enumerate(tool_executions, 1):
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name") or "").strip()
            if not tool_name:
                continue
            tool_call_id = f"call_{index}_{tool_name}"
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False, default=str),
                            },
                        }
                    ],
                }
            )
            tool_payload: Any
            if item.get("success"):
                tool_payload = item.get("result")
            else:
                tool_payload = {
                    "success": False,
                    "error": item.get("error"),
                }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(tool_payload, ensure_ascii=False, default=str),
                }
            )
        messages.append({"role": "assistant", "content": "" if final_answer is None else str(final_answer)})
        return {"format": "openai_chat_completions_messages", "messages": messages}

    @staticmethod
    def _build_complete_trajectory(
        user_prompt: str,
        tool_executions: List[Dict[str, Any]],
        final_answer: Any,
        llm_trace: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        clean_trajectory = ModelDrivenToolAgent._build_openai_trajectory(user_prompt, tool_executions, final_answer)
        return {
            "format": "complete_execution_trace",
            "summary": {
                "user_prompt": user_prompt,
                "final_answer": "" if final_answer is None else str(final_answer),
            },
            "reconstructed_openai_trajectory": clean_trajectory,
            "events": events,
            "llm_trace": llm_trace,
            "tool_executions": tool_executions,
        }

    @staticmethod
    def _build_task_level_events(execution_log: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        task_id = execution_log.get("task_id")

        def _round_value(item: Dict[str, Any]) -> int:
            meta_raw = item.get("meta")
            meta: Dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            raw = item.get("round") or meta.get("round")
            raw_text = str(raw or "")
            return int(raw_text) if raw_text.isdigit() else 0

        events.append(
            {
                "event_type": "task_start",
                "task_id": task_id,
                "task_type": execution_log.get("task_type"),
                "time": execution_log.get("start_time"),
                "user_prompt": execution_log.get("user_prompt"),
            }
        )

        llm_events = list(execution_log.get("llm_trace") or [])
        tool_events = list(execution_log.get("tool_executions") or [])
        policy_actions = list(execution_log.get("policy_actions") or [])
        round_numbers = sorted(
            {
                _round_value(item)
                for item in [*llm_events, *tool_events, *policy_actions]
                if isinstance(item, dict) and _round_value(item) > 0
            }
        )

        for round_number in round_numbers:
            events.append({"event_type": "round_start", "task_id": task_id, "round": round_number})
            for policy in policy_actions:
                if not isinstance(policy, dict) or int(policy.get("round") or 0) != round_number:
                    continue
                events.append(
                    {
                        "event_type": "policy_action",
                        "task_id": task_id,
                        "round": round_number,
                        "action": policy.get("action"),
                        "tool_name": policy.get("tool_name"),
                        "final_answer": policy.get("final_answer"),
                        "reason": policy.get("reason"),
                    }
                )
            for trace in llm_events:
                if not isinstance(trace, dict) or _round_value(trace) != round_number:
                    continue
                event = {"event_type": "llm_trace", "task_id": task_id, "round": round_number}
                event.update(trace)
                events.append(event)
            for tool in tool_events:
                if not isinstance(tool, dict) or int(tool.get("round") or 0) != round_number:
                    continue
                events.append(
                    {
                        "event_type": "tool_execution",
                        "task_id": task_id,
                        "round": round_number,
                        "tool_name": tool.get("tool_name"),
                        "success": tool.get("success"),
                        "arguments": tool.get("arguments"),
                        "error": tool.get("error"),
                        "result": tool.get("result"),
                        "duration_sec": tool.get("duration_sec"),
                    }
                )
            events.append({"event_type": "round_finish", "task_id": task_id, "round": round_number})

        events.append(
            {
                "event_type": "task_finish",
                "task_id": task_id,
                "time": execution_log.get("end_time"),
                "success": execution_log.get("success"),
                "final_answer": execution_log.get("final_answer"),
                "error": execution_log.get("error"),
            }
        )
        return events

    def _decide_next_action(
        self,
        user_prompt: str,
        round_index: int,
        allowed_tools: List[str],
        tool_results: Dict[str, Any],
        action_history: List[Dict[str, Any]],
        last_error: Optional[str],
        last_tool_name: Optional[str],
        deferred_hint: Optional[Dict[str, str]],
        oracle_hazard_label: Optional[str] = None,
        redact_answer_fields: bool = False,
    ) -> Dict[str, Any]:
        hint_block = ""
        if deferred_hint:
            hint_block = (
                "\n\n[RECOVERY_HINT_CONTEXT]\n"
                f"hint_key: {deferred_hint.get('hint_key', '')}\n"
                f"retry_strategy: {deferred_hint.get('retry_strategy', '')}\n"
                f"minimal_prompt_hint: {deferred_hint.get('minimal_prompt_hint', '')}\n"
                f"detailed_prompt_hint: {deferred_hint.get('detailed_prompt_hint', '')}\n"
                "[/RECOVERY_HINT_CONTEXT]"
            )
        oracle_label_block = self._build_oracle_hazard_block(oracle_hazard_label)
        model_visible_tool_results = self._scrub_model_visible_answers(tool_results) if redact_answer_fields else tool_results
        finish_guidance = (
            "- When action=finish, final_answer must be supported by successful tool outputs; do not rely on redacted answer fields."
            if redact_answer_fields
            else "- When action=finish, final_answer must be the exact raw scalar from tool outputs when available.\n- Prefer tool fields like final_value/final_answer/value/result over rephrasing."
        )
        prompt = f"""
You are a tool-orchestration policy model.

Return ONLY JSON with this schema:
{{
  "action": "call_tool" | "retry" | "fallback" | "finish",
  "tool_name": "<tool name or empty>",
  "final_answer": "<non-empty only when action=finish>",
  "reason": "<short reason>"
}}

Rules:
- Use only allowed tools.
- Prefer finish only when available tool outputs are sufficient.
- If a tool failed, you may retry same tool or fallback to another tool.
{finish_guidance}
- Never add currency symbols, currency codes, labels, or explanation text to final_answer.
- Treat any tool result with success=false, ok=false, missing required fields, or explicit error text as unresolved evidence, not completion.
- If RECOVERY_HINT_CONTEXT is present, obey its retry strategy, mandatory tool sequence, required inputs, verification checks, and finish blockers.
- If ORACLE_HAZARD_LABEL is present, use the category only as diagnostic context; it does not provide a recovery procedure or answer.
- Do not finish while any allowed tool that is needed for the answer has not been executed successfully.
- Without RECOVERY_HINT_CONTEXT, do not keep doing blind retries or speculative schema repair on the same failing tool. At most one unguided retry per failing tool; after that, prefer another evidence-producing tool or finish as unresolved.
- No markdown.

## User Request
{user_prompt}

## Allowed Tools
{json.dumps(allowed_tools, ensure_ascii=False)}

## Round
{round_index}

## Previous Action History
{json.dumps(action_history, ensure_ascii=False, indent=2, default=str)}

## Current Tool Results
{json.dumps(model_visible_tool_results, ensure_ascii=False, indent=2, default=str)}

## Last Error
{last_error or "None"}
{hint_block}
{oracle_label_block}
""".strip()
        response = self.llm.chat_completion(
            [{"role": "user", "content": prompt}],
            trace_label="policy_decision",
            trace_meta={"round": round_index, "phase": "policy_decision"},
        )
        content = str(response.get("content") or "").strip()
        decision = self._extract_json(content)
        action = str(decision.get("action", "")).strip().lower()
        tool_name = str(decision.get("tool_name", "")).strip()
        final_answer = str(decision.get("final_answer", "")).strip()
        reason = str(decision.get("reason", "")).strip()
        fallback_tool = self._next_unresolved_mandatory_tool(allowed_tools, tool_results, deferred_hint)
        if not fallback_tool:
            fallback_tool = str(last_tool_name or "").strip() or (allowed_tools[0] if allowed_tools else "")
        if action not in {"call_tool", "retry", "fallback", "finish"}:
            if fallback_tool:
                return {
                    "action": "call_tool",
                    "tool_name": fallback_tool,
                    "final_answer": "",
                    "reason": "invalid_decision_json_call_fallback_tool",
                }
            return {"action": "finish", "tool_name": "", "final_answer": "", "reason": "invalid_decision_json"}
        if action in {"call_tool", "fallback"} and tool_name not in allowed_tools:
            if fallback_tool:
                return {
                    "action": "call_tool",
                    "tool_name": fallback_tool,
                    "final_answer": "",
                    "reason": "invalid_tool_name_call_fallback_tool",
                }
            return {"action": "finish", "tool_name": "", "final_answer": "", "reason": "invalid_tool_name"}
        return {"action": action, "tool_name": tool_name, "final_answer": final_answer, "reason": reason}

    @staticmethod
    def _extract_prompt_preface_value(user_prompt: str) -> Optional[str]:
        prompt = str(user_prompt or "").strip()
        if not prompt:
            return None
        match = re.match(r"\s*For\s+(.+?)(?:,|\s+return\b|\s+provide\b|\s+give\b)", prompt, flags=re.IGNORECASE)
        if match:
            value = str(match.group(1) or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _extract_id_like_value(user_prompt: str) -> Optional[str]:
        prompt = str(user_prompt or "")
        match = re.search(r"\b([A-Z]{2,}(?:-[A-Z0-9]+)+)\b", prompt)
        return str(match.group(1)).strip() if match else None

    @staticmethod
    def _extract_zip_like_value(user_prompt: str) -> Optional[str]:
        prompt = str(user_prompt or "")
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b", prompt)
        return str(match.group(1)).strip() if match else None

    def _backfill_missing_required_args(
        self,
        user_prompt: str,
        raw_args: Dict[str, Any],
        signature: inspect.Signature,
    ) -> Dict[str, Any]:
        args = dict(raw_args or {})
        preface_value = self._extract_prompt_preface_value(user_prompt)
        id_like_value = self._extract_id_like_value(user_prompt)
        zip_like_value = self._extract_zip_like_value(user_prompt)
        location_names = {
            "location", "city", "locality", "neighborhood", "area", "destination", "service_area",
            "facility", "site", "region", "town",
        }
        id_names = {
            "cart_id", "checkout_id", "checkout_session", "session_id", "draft_id", "order_id", "order_draft",
            "abandoned_cart_id", "return_case_id", "case_id", "customer_id", "sku", "product_id", "tracking_id",
        }
        zip_names = {"zip", "zipcode", "zip_code", "destination_zip", "postal_code"}
        for param_name, param in signature.parameters.items():
            if param_name in args or param.default is not inspect.Parameter.empty:
                continue
            lowered = str(param_name).strip().lower()
            if lowered in location_names and preface_value:
                args[param_name] = preface_value
                continue
            if lowered in zip_names and zip_like_value:
                args[param_name] = zip_like_value
                continue
            if lowered in id_names and id_like_value:
                args[param_name] = id_like_value
                continue
        return args

    @staticmethod
    def _extract_verified_promo_tuple(tool_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ordered_results = list(ModelDrivenToolAgent._successful_result_payloads(tool_results))
        for result in reversed(ordered_results):
            promo_code = str(result.get("best_promo_code") or result.get("promo_code") or "").strip()
            discount_value = result.get("discount_usd")
            if not promo_code or discount_value in (None, "", [], {}, ()):
                continue
            try:
                discount_usd = round(float(str(discount_value).strip()), 2)
            except Exception:
                continue
            return {"best_promo_code": promo_code, "discount_usd": discount_usd}
        return None

    def _enforce_verified_combo_args(
        self,
        raw_args: Dict[str, Any],
        signature: inspect.Signature,
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        args = dict(raw_args or {})
        verified_promo = self._extract_verified_promo_tuple(tool_results)
        param_names = set(signature.parameters.keys())
        if verified_promo and "discount_usd" in param_names:
            args["discount_usd"] = verified_promo["discount_usd"]
            if "best_promo_code" in param_names:
                args["best_promo_code"] = verified_promo["best_promo_code"]
        return args

    def _execute_tool_once(
        self,
        tool_name: str,
        runtime: Dict[str, Any],
        user_prompt: str,
        previous_results: Dict[str, Any],
        policy_trace: List[Dict[str, Any]],
        round_index: int,
        redact_answer_fields: bool = False,
    ) -> Dict[str, Any]:
        start = time.time()
        runtime_item = runtime.get(tool_name)
        if not runtime_item:
            return {
                "tool_name": tool_name,
                "success": False,
                "error": f"Tool '{tool_name}' unavailable",
                "duration_sec": time.time() - start,
                "round": round_index,
                "attempt": 1,
            }
        fn = runtime_item["fn"]
        signature = runtime_item["signature"]
        signature_str = runtime_item["signature_str"]
        description = runtime_item["description"]
        parameters_schema = runtime_item["parameters_schema"]
        try:
            model_visible_previous_results = self._scrub_model_visible_answers(previous_results) if redact_answer_fields else previous_results
            previous_payload = {
                "prior_tool_results": model_visible_previous_results,
                "policy_trace": policy_trace,
            }
            args = self.llm.generate_tool_args_with_schema(
                tool_name=tool_name,
                tool_description=description,
                parameters_schema=parameters_schema,
                user_prompt=user_prompt,
                previous_results=json.dumps(previous_payload, ensure_ascii=False, default=str),
                trace_meta={"round": round_index, "phase": "tool_argument_generation"},
            )
            args = self._backfill_missing_required_args(user_prompt, args, signature)
            args = self._enforce_verified_combo_args(args, signature, previous_results)
            validated_args = validate_tool_args(args, signature)
            result = fn(**validated_args)
            return {
                "tool_name": tool_name,
                "success": True,
                "arguments": validated_args,
                "result": result,
                "signature": signature_str,
                "duration_sec": time.time() - start,
                "round": round_index,
                "attempt": 1,
            }
        except Exception as e:
            return {
                "tool_name": tool_name,
                "success": False,
                "error": str(e),
                "signature": signature_str,
                "duration_sec": time.time() - start,
                "round": round_index,
                "attempt": 1,
            }

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task.get("id", "unknown")
        user_prompt = str(task.get("user_prompt", ""))
        tool_specs = self._normalize_tool_specs(task)
        expected = task.get("final_answer")
        task_type = self._infer_task_type(task, tool_specs)
        logger.info(f"🚀 Executing task(model-driven): {task_id} ({task_type})")
        execution_log: Dict[str, Any] = {
            "task_id": task_id,
            "task_type": task_type,
            "user_prompt": user_prompt,
            "expected_answer": expected,
            "expected_match": None,
            "tool_executions": [],
            "policy_actions": [],
            "final_answer": None,
            "success": False,
            "error": None,
            "start_time": time.time(),
        }
        tool_results: Dict[str, Any] = {}
        action_history: List[Dict[str, Any]] = []
        allowed_tools = [spec["tool_name"] for spec in tool_specs]
        last_tool_name: Optional[str] = None
        last_error: Optional[str] = None
        meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
        tool_runtime_user_prompt = str(meta.get("original_user_prompt") or user_prompt)
        redact_model_visible_answer_fields = bool(meta.get("test_time_scaling_retry"))
        hint_mode = str(meta.get("ab_hint_injection_mode") or "").strip()
        deferred_hint: Optional[Dict[str, str]] = self._build_start_hint(task) if hint_mode == "from_start" else None
        hint_injected = deferred_hint is not None
        oracle_hazard_label = str(meta.get("oracle_hazard_label") or "").strip()
        oracle_label_visible = False
        execution_log["oracle_hazard_label"] = oracle_hazard_label or None
        execution_log["oracle_label_injected"] = False
        self.llm.reset_trace()
        try:
            self._apply_tool_runtime_context(object(), tool_runtime_user_prompt, deferred_hint)
            module = self._load_task_tools(task, allowed_tools)
            self._apply_tool_runtime_context(module, tool_runtime_user_prompt, deferred_hint)
            runtime = self._build_tool_runtime_catalog(module, allowed_tools)
            for round_idx in range(1, self.max_rounds + 1):
                decision = self._decide_next_action(
                    user_prompt=user_prompt,
                    round_index=round_idx,
                    allowed_tools=allowed_tools,
                    tool_results=tool_results,
                    action_history=action_history,
                    last_error=last_error,
                    last_tool_name=last_tool_name,
                    deferred_hint=deferred_hint,
                    oracle_hazard_label=oracle_hazard_label if oracle_label_visible else None,
                    redact_answer_fields=redact_model_visible_answer_fields,
                )
                execution_log["policy_actions"].append({"round": round_idx, **decision})
                action = decision.get("action", "finish")
                if action == "finish":
                    blocked_finish, next_tool = self._default_blocks_finish(task, runtime, tool_results)
                    if blocked_finish and next_tool and next_tool in runtime:
                        action_history.append(
                            {
                                "round": round_idx,
                                "action": "block_finish",
                                "tool_name": next_tool,
                                "reason": "default_guardrails_blocked_early_finish",
                            }
                        )
                        last_error = "default_guardrails_blocked_early_finish"
                        last_tool_name = next_tool
                        continue
                    blocked_finish, next_tool = self._hint_blocks_finish(task, tool_results)
                    if blocked_finish and next_tool and next_tool in runtime:
                        action_history.append(
                            {
                                "round": round_idx,
                                "action": "block_finish",
                                "tool_name": next_tool,
                                "reason": "hint_guardrails_blocked_early_finish",
                            }
                        )
                        last_error = "hint_guardrails_blocked_early_finish"
                        last_tool_name = next_tool
                        continue
                    proposed = str(decision.get("final_answer") or "").strip()
                    final_answer = self._finalize_answer_with_hint_guardrails(task, user_prompt, tool_results, proposed)
                    execution_log["final_answer"] = final_answer
                    matched = self._is_expected_match(expected, final_answer)
                    execution_log["expected_match"] = matched if expected is not None else None
                    if self.enforce_expected_match and expected is not None and not matched:
                        execution_log["success"] = False
                        execution_log["error"] = (
                            f"Final answer does not match expected_answer. expected={str(expected)!r}, actual={str(final_answer)!r}"
                        )
                    else:
                        execution_log["success"] = True
                    break

                target_tool = str(decision.get("tool_name") or "").strip()
                if action == "retry":
                    target_tool = target_tool or str(last_tool_name or "").strip()
                if not target_tool or target_tool not in runtime:
                    last_error = f"Invalid target tool: {target_tool}"
                    action_history.append({"round": round_idx, "action": action, "error": last_error})
                    continue
                if self._has_exception_tooling(self.tools_dir) and not deferred_hint and not oracle_label_visible:
                    if self._unguided_retry_consumed(
                        execution_log["tool_executions"], target_tool
                    ):
                        last_error = f"unguided_retry_blocked_for_{target_tool}"
                        action_history.append(
                            {
                                "round": round_idx,
                                "action": "block_unguided_retry",
                                "tool_name": target_tool,
                                "reason": last_error,
                            }
                        )
                        continue

                self._apply_tool_runtime_context(module, tool_runtime_user_prompt, deferred_hint)
                tool_execution = self._execute_tool_once(
                    tool_name=target_tool,
                    runtime=runtime,
                    user_prompt=self._build_runtime_user_prompt(
                        tool_runtime_user_prompt,
                        deferred_hint,
                        oracle_hazard_label if oracle_label_visible else None,
                    ),
                    previous_results=tool_results,
                    policy_trace=execution_log["policy_actions"],
                    round_index=round_idx,
                    redact_answer_fields=redact_model_visible_answer_fields,
                )
                execution_log["tool_executions"].append(tool_execution)
                tool_results[target_tool] = tool_execution
                last_tool_name = target_tool

                if tool_execution.get("success"):
                    semantic_issue = self._tool_execution_issue_text(tool_execution)
                    last_error = semantic_issue or None
                    action_history.append(
                        {
                            "round": round_idx,
                            "action": action,
                            "tool_name": target_tool,
                            "success": semantic_issue is None,
                            "semantic_issue": semantic_issue,
                        }
                    )
                    if semantic_issue:
                        if oracle_hazard_label:
                            oracle_label_visible = True
                            execution_log["oracle_label_injected"] = True
                        merged_hint = self._merge_failure_hint_into_context(task, deferred_hint, target_tool, semantic_issue)
                        if merged_hint:
                            deferred_hint = merged_hint
                            hint_injected = True
                    continue

                last_error = str(tool_execution.get("error") or "tool execution failed")
                if oracle_hazard_label:
                    oracle_label_visible = True
                    execution_log["oracle_label_injected"] = True
                action_history.append(
                    {
                        "round": round_idx,
                        "action": action,
                        "tool_name": target_tool,
                        "success": False,
                        "error": last_error,
                    }
                )
                merged_hint = self._merge_failure_hint_into_context(task, deferred_hint, target_tool, last_error)
                if merged_hint:
                    deferred_hint = merged_hint
                    hint_injected = True

            if execution_log.get("final_answer") is None:
                execution_log["final_answer"] = self._finalize_answer_with_hint_guardrails(task, user_prompt, tool_results, "")
                matched = self._is_expected_match(expected, execution_log["final_answer"])
                execution_log["expected_match"] = matched if expected is not None else None
                if self.enforce_expected_match and expected is not None and not matched:
                    execution_log["success"] = False
                    execution_log["error"] = (
                        f"Final answer does not match expected_answer. expected={str(expected)!r}, actual={str(execution_log['final_answer'])!r}"
                    )
                else:
                    execution_log["success"] = True
        except Exception as e:
            execution_log["error"] = str(e)
            execution_log["final_answer"] = f"Error: {e}"
            execution_log["success"] = False

        execution_log["end_time"] = time.time()
        execution_log["duration_sec"] = execution_log["end_time"] - execution_log["start_time"]
        execution_log["llm_trace"] = self.llm.get_trace()
        execution_log["execution_rounds"] = [{"round": idx} for idx in sorted({int(action.get("round") or 0) for action in execution_log.get("policy_actions", []) if isinstance(action, dict) and int(action.get("round") or 0) > 0})]
        task_events = self._build_task_level_events(execution_log)
        execution_log["openai_trajectory"] = self._build_openai_trajectory(
            user_prompt=user_prompt,
            tool_executions=execution_log.get("tool_executions", []),
            final_answer=execution_log.get("final_answer"),
        )
        execution_log["complete_trajectory"] = self._build_complete_trajectory(
            user_prompt=user_prompt,
            tool_executions=execution_log.get("tool_executions", []),
            final_answer=execution_log.get("final_answer"),
            llm_trace=execution_log.get("llm_trace", []),
            events=task_events,
        )
        return execution_log

    def run_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"📦 Running batch(model-driven) of {len(tasks)} tasks")
        results = []
        success_count = 0
        for i, task in enumerate(tasks, 1):
            logger.info(f"[{i}/{len(tasks)}] Processing...")
            result = self.execute_task(task)
            results.append(result)
            if result.get("success"):
                success_count += 1
        summary = {
            "total": len(tasks),
            "success": success_count,
            "failed": len(tasks) - success_count,
            "success_rate": f"{success_count/len(tasks)*100:.1f}%" if tasks else "0%",
            "results": results,
        }
        logger.info(f"📊 Batch complete(model-driven): {summary['success_rate']} success rate")
        return summary
