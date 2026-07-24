"""LLM 客户端封装，支持重试、缓存和标准化接口."""
import ast
import copy
import json
import os
import importlib
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
try:
    _tenacity = importlib.import_module("tenacity")
    retry = _tenacity.retry
    stop_after_attempt = _tenacity.stop_after_attempt
    wait_exponential = _tenacity.wait_exponential
except ImportError:
    def retry(*_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator

    def stop_after_attempt(*_args, **_kwargs):
        return None

    def wait_exponential(*_args, **_kwargs):
        return None

from .utils import logger


class LLMClient:
    """统一的 LLM 客户端，支持多种模型和配置."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 8192,
        timeout: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        self.extra_body = copy.deepcopy(extra_body) if extra_body is not None else None

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=self.timeout,
        )
        self._trace_events: List[Dict[str, Any]] = []
        self._trace_seq = 0

    def reset_trace(self) -> None:
        self._trace_events = []
        self._trace_seq = 0

    def get_trace(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._trace_events)

    def _append_trace_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._trace_seq += 1
        event: Dict[str, Any] = {
            "seq": self._trace_seq,
            "time": time.time(),
            "type": event_type,
        }
        event.update(payload)
        self._trace_events.append(event)

    @staticmethod
    def _normalize_tool_calls_for_trace(tool_calls: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for call in tool_calls or []:
            function_obj = getattr(call, "function", None)
            if function_obj is None and isinstance(call, dict):
                function_obj = call.get("function")
            normalized.append(
                {
                    "id": getattr(call, "id", None) if not isinstance(call, dict) else call.get("id"),
                    "type": getattr(call, "type", None) if not isinstance(call, dict) else call.get("type"),
                    "function": {
                        "name": getattr(function_obj, "name", None)
                        if not isinstance(function_obj, dict)
                        else function_obj.get("name"),
                        "arguments": getattr(function_obj, "arguments", None)
                        if not isinstance(function_obj, dict)
                        else function_obj.get("arguments"),
                    },
                }
            )
        return normalized
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        trace_label: str = "chat_completion",
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送聊天请求，支持工具调用."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body is not None:
            kwargs["extra_body"] = copy.deepcopy(self.extra_body)
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        self._append_trace_event(
            "llm_request",
            {
                "label": trace_label,
                "meta": copy.deepcopy(trace_meta or {}),
                "request": {
                    "model": self.model,
                    "messages": copy.deepcopy(messages),
                    "tools": copy.deepcopy(tools),
                    "tool_choice": copy.deepcopy(tool_choice),
                    "max_tokens": self.max_tokens,
                },
            },
        )
        if self.temperature is not None:
            self._trace_events[-1]["request"]["temperature"] = self.temperature
        if self.extra_body is not None:
            self._trace_events[-1]["request"]["extra_body"] = copy.deepcopy(self.extra_body)
        
        logger.debug(f"LLM request: model={self.model}, tools={len(tools) if tools else 0}")
        
        response = self.client.chat.completions.create(**kwargs)
        
        result = {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        self._append_trace_event(
            "llm_response",
            {
                "label": trace_label,
                "meta": copy.deepcopy(trace_meta or {}),
                "response": {
                    "content": result["content"],
                    "tool_calls": self._normalize_tool_calls_for_trace(result["tool_calls"]),
                    "finish_reason": result["finish_reason"],
                    "usage": copy.deepcopy(result["usage"]),
                },
            },
        )
        return result

    def generate_tool_args_with_schema(
        self,
        tool_name: str,
        tool_description: str,
        parameters_schema: Dict[str, Any],
        user_prompt: str,
        previous_results: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hint_context = self._extract_recovery_hint_context(user_prompt)
        required_inputs = []
        if isinstance(hint_context.get("required_inputs"), dict):
            required_inputs = [
                str(item).strip()
                for item in (hint_context.get("required_inputs", {}).get(tool_name) or [])
                if str(item).strip()
            ]
        schema_required = [
            str(item).strip()
            for item in (parameters_schema.get("required") or [])
            if str(item).strip()
        ]
        required_note = ""
        if required_inputs or schema_required:
            merged_required: List[str] = []
            for item in required_inputs + schema_required:
                if item and item not in merged_required:
                    merged_required.append(item)
            required_note = f"\n- For {tool_name}, required inputs that must already be evidenced before the call: {', '.join(merged_required)}."
        prompt = f"""
You must call the provided function tool using valid arguments.

## User Request
{user_prompt}

## Previous Tool Results
{previous_results if previous_results else "None yet"}

## Operational Rules
- Use only evidence already present in the user request or previous tool results.
- Never invent missing arguments, IDs, paths, amounts, dates, ZIP codes, or enum values.
- If [RECOVERY_HINT_CONTEXT] appears in the user request, treat it as high-priority guidance for retries, required inputs, normalization, and finish blocking.
- Reuse verified values exactly; do not paraphrase or transform argument values unless the hint explicitly says to normalize aliases.
- When previous tool results contain errors or ok=false payloads, avoid copying incomplete fields from those failed results.
- Return arguments that maximize the chance of a correct downstream final scalar, not a plausible guess.
{required_note}
""".strip()

        combined_trace_meta = {"tool_name": tool_name, "strategy": "tool_call"}
        if trace_meta:
            combined_trace_meta.update(copy.deepcopy(trace_meta))
        response = self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": parameters_schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            trace_label="generate_tool_args_with_schema",
            trace_meta=combined_trace_meta,
        )

        args = self._extract_tool_call_arguments(response.get("tool_calls"), tool_name)
        if args is not None:
            self._append_trace_event(
                "tool_argument_resolution",
                {
                    "tool_name": tool_name,
                    "resolution": "tool_call_arguments",
                    "arguments": copy.deepcopy(args),
                },
            )
            return args

        content = response.get("content") or ""
        extracted = self._extract_json(content, silent=True)
        if extracted:
            self._append_trace_event(
                "tool_argument_resolution",
                {
                    "tool_name": tool_name,
                    "resolution": "assistant_content_json",
                    "arguments": copy.deepcopy(extracted),
                },
            )
            return extracted

        logger.warning(
            "Falling back to plain JSON argument generation for tool '%s' after invalid or empty tool call arguments",
            tool_name,
        )
        return self._generate_tool_args_text_fallback(
            tool_name=tool_name,
            parameters_schema=parameters_schema,
            user_prompt=user_prompt,
            previous_results=previous_results,
            trace_meta=trace_meta,
        )

    def _generate_tool_args_text_fallback(
        self,
        tool_name: str,
        parameters_schema: Dict[str, Any],
        user_prompt: str,
        previous_results: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        schema_text = json.dumps(parameters_schema, ensure_ascii=False, sort_keys=True)
        prompt = f"""
Return ONLY one valid JSON object for the tool arguments.

## Tool Name
{tool_name}

## JSON Schema
{schema_text}

## User Request
{user_prompt}

## Previous Tool Results
{previous_results if previous_results else "None yet"}

## Rules
- Output exactly one JSON object and nothing else.
- Use only keys allowed by the schema.
- Do not wrap the JSON in markdown fences.
- Do not invent missing values.
""".strip()

        combined_trace_meta = {"tool_name": tool_name, "strategy": "plain_json_fallback"}
        if trace_meta:
            combined_trace_meta.update(copy.deepcopy(trace_meta))
        response = self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            trace_label="generate_tool_args_text_fallback",
            trace_meta=combined_trace_meta,
        )
        content = response.get("content") or ""
        extracted = self._extract_json(content)
        self._append_trace_event(
            "tool_argument_resolution",
            {
                "tool_name": tool_name,
                "resolution": "plain_json_fallback",
                "arguments": copy.deepcopy(extracted),
            },
        )
        return extracted

    def _parse_possible_json_object(self, raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None

        candidate = raw.strip()
        if not candidate:
            return None

        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate, flags=re.IGNORECASE)
        if code_block_match:
            candidate = code_block_match.group(1).strip()

        parse_candidates = [candidate]
        extracted = self._extract_json(candidate, silent=True)
        if extracted:
            return extracted

        fenced_object = self._extract_first_balanced_object(candidate)
        if fenced_object and fenced_object not in parse_candidates:
            parse_candidates.append(fenced_object)

        for item in parse_candidates:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass

        return None

    @staticmethod
    def _extract_first_balanced_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return None

    def _extract_tool_call_arguments(
        self,
        tool_calls: Any,
        expected_tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        if not tool_calls:
            return None

        for call in tool_calls:
            function_obj = getattr(call, "function", None)
            if function_obj is None and isinstance(call, dict):
                function_obj = call.get("function")

            name = getattr(function_obj, "name", None)
            if name is None and isinstance(function_obj, dict):
                name = function_obj.get("name")
            if name != expected_tool_name:
                continue

            arguments = getattr(function_obj, "arguments", None)
            if arguments is None and isinstance(function_obj, dict):
                arguments = function_obj.get("arguments")

            if isinstance(arguments, dict):
                return arguments
            if isinstance(arguments, str) and arguments.strip():
                parsed = self._parse_possible_json_object(arguments)
                if parsed is not None:
                    self._append_trace_event(
                        "tool_argument_parse",
                        {
                            "tool_name": expected_tool_name,
                            "status": "recovered",
                            "raw_arguments": arguments,
                            "parsed_arguments": copy.deepcopy(parsed),
                        },
                    )
                    return parsed
                self._append_trace_event(
                    "tool_argument_parse",
                    {
                        "tool_name": expected_tool_name,
                        "status": "invalid_json",
                        "raw_arguments": arguments,
                    },
                )
                logger.warning(f"Tool call arguments are not valid JSON: {arguments[:200]}")

        return None
    
    @lru_cache(maxsize=100)
    def generate_tool_args(
        self,
        tool_name: str,
        tool_signature: str,
        user_prompt: str,
        previous_results: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """让 LLM 生成工具调用参数（带缓存）."""
        prompt = f"""
You are an expert at calling Python functions. Generate arguments for the tool below.

## User Request
{user_prompt}

## Tool to Call
Name: {tool_name}
Signature: {tool_signature}

## Previous Tool Results
{previous_results if previous_results else "None yet"}

## Instructions
- Return ONLY a valid JSON object with the arguments
- Use the exact parameter names from the signature
- Do not include explanations, markdown, or extra text
- If a parameter is optional and not needed, omit it

Example output: {{"param1": "value1", "param2": 123}}
""".strip()
        
        combined_trace_meta = {"tool_name": tool_name, "strategy": "text_only_json"}
        if trace_meta:
            combined_trace_meta.update(copy.deepcopy(trace_meta))
        response = self.chat_completion(
            [{"role": "user", "content": prompt}],
            trace_label="generate_tool_args_text_only",
            trace_meta=combined_trace_meta,
        )
        
        # 解析 JSON（容错处理）
        content = response["content"] or ""
        return self._extract_json(content)
    
    def _extract_json(self, text: str, silent: bool = False) -> Dict[str, Any]:
        """从文本中提取 JSON 对象."""
        if not text:
            return {}
        
        # 尝试提取代码块
        import re
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        candidate = match.group(1) if match else text
        
        # 尝试直接解析
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        
        # 尝试查找第一个 { ... }
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        
        if not silent:
            logger.warning(f"Failed to extract JSON from: {text[:200]}")
        return {}

    def _extract_recovery_hint_context(self, text: str) -> Dict[str, Any]:
        raw = str(text or "")
        marker_start = "[RECOVERY_HINT_CONTEXT]"
        marker_end = "[/RECOVERY_HINT_CONTEXT]"
        start = raw.find(marker_start)
        end = raw.find(marker_end)
        if start < 0 or end < 0 or end <= start:
            return {}
        block = raw[start + len(marker_start):end].strip()
        if not block:
            return {}
        try:
            data = json.loads(block)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    
    def _normalize_scalar(self, value: Any) -> Optional[str]:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (str, int, float)):
            return str(value).strip()
        return None

    def _normalize_candidate_scalar_text(self, text: Any) -> Optional[str]:
        if isinstance(text, bool) or text is None:
            return None
        value = str(text).strip()
        if not value:
            return None
        value = self._unwrap_common_scalar_wrappers(value)
        value = value.replace(",", "")
        currency_match = None
        currency_match = re.fullmatch(r"(?:USD|usd)?\s*\$?\s*(-?\d+(?:\.\d+)?)", value)
        if currency_match:
            return currency_match.group(1)
        scalar_match = re.fullmatch(r"-?\d+(?:\.\d+)?", value)
        if scalar_match:
            return scalar_match.group(0)
        return value

    def _unwrap_common_scalar_wrappers(self, text: str) -> str:
        value = str(text or "").strip().strip('"').strip("'")
        wrapper_pattern = re.compile(r"^(?:answer|result|final answer|value)\s*[:|-]\s*(.+)$", flags=re.IGNORECASE)
        for _ in range(3):
            match = wrapper_pattern.fullmatch(value)
            if not match:
                break
            candidate = str(match.group(1) or "").strip().strip('"').strip("'")
            if not candidate:
                break
            value = candidate
        return value

    def _iter_successful_result_payloads(self, tool_results: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        payloads: List[Tuple[str, Dict[str, Any]]] = []
        for tool_name, execution in (tool_results or {}).items():
            if not isinstance(execution, dict) or not execution.get("success"):
                continue
            result = execution.get("result")
            if isinstance(result, dict):
                payloads.append((str(tool_name), result))
        return payloads

    @staticmethod
    def _is_single_token_enum_like(text: str) -> bool:
        value = str(text or "").strip()
        if not value or " " in value:
            return False
        if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)*", value):
            return True
        if re.fullmatch(r"[A-Z]{2,}(?:[_-][A-Z0-9]+)+", value):
            return True
        return False

    @staticmethod
    def _looks_like_location_phrase(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", value):
            return True
        return False

    @staticmethod
    def _is_wrapped_answer_text(text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value:
            return False
        return any(token in value for token in ["answer:", "result:", "because", "therefore", "final answer"]) 

    @staticmethod
    def _context_field_values(result: Dict[str, Any]) -> List[str]:
        fields = [
            "location",
            "site",
            "city",
            "region",
            "town",
            "locality",
            "service_area",
            "checkout_session_id",
            "payment_attempt_id",
            "gateway_response_id",
            "cart_id",
            "order_id",
        ]
        values: List[str] = []
        for field in fields:
            raw = result.get(field)
            text = str(raw or "").strip()
            if text:
                values.append(text)
        return values

    def _score_scalar_candidate(self, key: str, original_value: Any, normalized_value: str, result: Dict[str, Any]) -> int:
        key_priority = {
            "final_value": 140,
            "final_answer": 135,
            "final_total": 130,
            "final_total_usd": 125,
            "decline_reason_code": 120,
            "accepted_recycling_code": 118,
            "pickup_day": 116,
            "net_refund_amount": 114,
            "net_refund": 112,
            "answer": 105,
            "value": 100,
            "result": 95,
        }
        score = key_priority.get(str(key).strip(), 60)
        original_text = str(original_value or "").strip()
        normalized_text = str(normalized_value or "").strip()
        if not normalized_text:
            return -10**9
        if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_text):
            score += 18
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized_text):
            score += 18
        if re.fullmatch(r"[A-Z]{2,}(?:[_-][A-Z0-9]+)+", normalized_text):
            score += 20
        if re.fullmatch(r"[A-Za-z]{1,4}-\d{2,}", normalized_text):
            score += 16
        if self._is_single_token_enum_like(normalized_text):
            score += 14
        if len(normalized_text.split()) == 1:
            score += 8
        if original_text and normalized_text != original_text:
            score += 10
        if result.get("ok") is True:
            score += 8
        if str(result.get("error") or "").strip():
            score -= 25
        if self._is_wrapped_answer_text(original_text):
            score -= 25
        if self._looks_like_location_phrase(normalized_text):
            score -= 18
        context_values = {value.lower() for value in self._context_field_values(result)}
        if normalized_text.lower() in context_values:
            score -= 35
        if str(key).strip().lower() in {"location", "site", "city", "region", "town", "locality", "service_area"}:
            score -= 30
        categorical_context_keys = {
            "utility_type",
            "decline_reason_code",
            "accepted_recycling_code",
            "pickup_day",
            "accepted_disposal_method",
            "disposal_method",
            "final_value",
            "final_answer",
        }
        if str(key).strip() in categorical_context_keys and self._is_single_token_enum_like(normalized_text):
            score += 18
        if str(key).strip() == "utility_type" and normalized_text in {"electricity", "water", "gas", "internet"}:
            score += 22
        if len(normalized_text) > 48:
            score -= 20
        return score

    def _candidate_consensus_bonus(self, scalar: str, tool_results: Dict[str, Any]) -> int:
        normalized = str(scalar or "").strip().lower()
        if not normalized:
            return 0
        occurrences = 0
        for _tool_name, result in self._iter_successful_result_payloads(tool_results):
            for value in result.values():
                candidate = self._normalize_candidate_scalar_text(value)
                if candidate and str(candidate).strip().lower() == normalized:
                    occurrences += 1
        if occurrences <= 1:
            return 0
        return min(18, (occurrences - 1) * 6)

    def _collect_scalar_candidates(self, tool_results: Dict[str, Any]) -> List[Tuple[int, str]]:
        preferred_keys = [
            "final_value",
            "final_answer",
            "final_total",
            "final_total_usd",
            "net_refund_amount",
            "net_refund",
            "decline_reason_code",
            "accepted_recycling_code",
            "pickup_day",
            "max_allowed",
            "organics_cart_size_gallons",
            "accepted_disposal_method",
            "disposal_method",
            "answer",
            "value",
            "result",
        ]
        candidates: List[Tuple[int, str]] = []
        ordered_results = self._iter_successful_result_payloads(tool_results)
        for reverse_index, (_tool_name, result) in enumerate(reversed(ordered_results)):
            recency_bonus = max(0, 10 - reverse_index)
            for key in preferred_keys:
                if key not in result:
                    continue
                scalar = self._normalize_candidate_scalar_text(result.get(key))
                if scalar is None:
                    continue
                score = self._score_scalar_candidate(key, result.get(key), scalar, result) + recency_bonus
                score += self._candidate_consensus_bonus(scalar, tool_results)
                candidates.append((score, scalar))
            if len(result) == 1:
                only_key, only_value = next(iter(result.items()))
                scalar = self._normalize_candidate_scalar_text(only_value)
                if scalar is not None:
                    score = self._score_scalar_candidate(str(only_key), only_value, scalar, result) + recency_bonus - 5
                    score += self._candidate_consensus_bonus(scalar, tool_results)
                    candidates.append((score, scalar))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _extract_final_scalar(self, tool_results: Dict[str, Any]) -> Optional[str]:
        candidates = self._collect_scalar_candidates(tool_results)
        return candidates[0][1] if candidates else None

    def extract_final_scalar(self, tool_results: Dict[str, Any]) -> Optional[str]:
        return self._extract_final_scalar(tool_results)

    def synthesize_final_answer(
        self,
        user_prompt: str,
        tool_results: Dict[str, Any],
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        scalar = self._extract_final_scalar(tool_results)
        if scalar is not None:
            return scalar

        prompt = f"""
You are a helpful assistant. Based on the tool execution results below,
return the exact final scalar answer to the user's request.

## User Request
{user_prompt}

## Tool Execution Results
{json.dumps(tool_results, ensure_ascii=False, indent=2, default=str)}

## Instructions
- Return ONLY the raw final scalar value
- Do not include markdown, prose, labels, or explanation
- Use tool execution outputs as the sole source of truth
- Do not fabricate values from expected answer when tools do not support it
- Prefer canonical scalar fields like final_value, final_answer, final_total, final_total_usd, net_refund_amount, answer, value, or result
- If the tool output contains a currency label like "$42.18" or "USD 42.18", strip the label and return only the canonical numeric scalar
- If tool results are incomplete or contradictory, return the best supported scalar from successful tool outputs only; never guess beyond the evidence

Final Answer:
""".strip()
        
        combined_trace_meta = {"strategy": "final_answer_synthesis"}
        if trace_meta:
            combined_trace_meta.update(copy.deepcopy(trace_meta))
        response = self.chat_completion(
            [{"role": "user", "content": prompt}],
            trace_label="synthesize_final_answer",
            trace_meta=combined_trace_meta,
        )
        content = (response["content"] or "Unable to generate answer.").strip()
        return content
