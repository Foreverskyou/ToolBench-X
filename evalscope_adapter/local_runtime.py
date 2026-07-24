from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ab_main_model_driven import _build_ab_comparison, _prepare_tasks_for_mode
from agent.evaluation import compute_exact_match_metrics, compute_tool_coverage
from agent.llm_client import LLMClient
from agent.loader import discover_tool_path
from agent.model_driven_core import ModelDrivenToolAgent
from config.settings import LLM_MODEL, LLM_TEMPERATURE
from main_model_driven import _attach_hints, _load_hint_catalog, _load_tasks_from_cli
from export_openai_trajectory_clean import clean_eval_payload


RAW_RESULT_EXCLUDED_FIELDS = {"llm_trace", "execution_rounds", "complete_trajectory"}


def _sanitize_raw_result(result: Dict[str, Any]) -> Dict[str, Any]:
    for field in RAW_RESULT_EXCLUDED_FIELDS:
        result.pop(field, None)
    return result


def _write_json(path: Optional[Path | str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _task_cfg_get(task_cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(task_cfg, dict):
        return task_cfg.get(key, default)
    return getattr(task_cfg, key, default)


def _load_hazard_manifest(path: Path | str) -> Dict[tuple[str, str], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"Hazard manifest {path} has no list-valued 'items' field")
    labels: Dict[tuple[str, str], str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("relative_path") or "").strip()
        task_id = str(item.get("task_id") or "").strip()
        label = str(item.get("exception_type") or "").strip()
        if relative_path and task_id and label:
            labels[(relative_path, task_id)] = label
    return labels


def _attach_oracle_hazard_labels(
    tasks: List[Dict[str, Any]],
    hazard_manifest: Path | str,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    prepared = copy.deepcopy(tasks)
    labels = _load_hazard_manifest(hazard_manifest)
    stats = {"total": len(prepared), "label_attached": 0, "label_missing": 0}
    for task in prepared:
        meta = task.setdefault("_meta", {})
        key = (
            str(meta.get("relative_path") or "").strip(),
            str(task.get("id") or "").strip(),
        )
        label = labels.get(key)
        if label:
            meta["oracle_hazard_label"] = label
            stats["label_attached"] += 1
        else:
            stats["label_missing"] += 1
    return prepared, stats


def _execute_single_task(
    task: Dict[str, Any],
    tools_dir: Path,
    max_rounds: int,
    llm_model: str,
    llm_temperature: Optional[float],
    llm_api_url: Optional[str],
    llm_api_key: Optional[str],
    llm_request_timeout: float,
    llm_extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    llm = LLMClient(
        model=llm_model,
        base_url=llm_api_url or None,
        api_key=llm_api_key or None,
        temperature=llm_temperature,
        timeout=llm_request_timeout,
        extra_body=llm_extra_body,
    )
    agent = ModelDrivenToolAgent(tools_dir=tools_dir, llm=llm, max_rounds=max_rounds)
    return agent.execute_task(task)


def _run_batch_parallel(
    tasks: List[Dict[str, Any]],
    tools_dir: Path,
    max_rounds: int,
    max_workers: int,
    llm_model: str,
    llm_temperature: Optional[float],
    llm_api_url: Optional[str],
    llm_api_key: Optional[str],
    llm_request_timeout: float,
    llm_extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    started = time.time()
    indexed_tasks = list(enumerate(tasks))
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(indexed_tasks)
    success_count = 0
    failed_count = 0

    tqdm.write(f"📦 Running batch of {len(tasks)} tasks with max_workers={max_workers}")
    if indexed_tasks:
        preview = ", ".join(str(task.get("id") or f"#{idx + 1}") for idx, task in indexed_tasks[: min(5, len(indexed_tasks))])
        tqdm.write(f"🧾 First tasks in queue: {preview}")

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor, tqdm(
        total=len(indexed_tasks),
        desc="📦 batch",
        unit="task",
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        future_to_index = {
            executor.submit(
                _execute_single_task,
                task,
                tools_dir,
                max_rounds,
                llm_model,
                llm_temperature,
                llm_api_url,
                llm_api_key,
                llm_request_timeout,
                llm_extra_body,
            ): index
            for index, task in indexed_tasks
        }
        pbar.set_postfix({"ok": 0, "fail": 0, "last": "-"}, refresh=True)
        for completed, future in enumerate(as_completed(future_to_index), 1):
            index = future_to_index[future]
            result = _sanitize_raw_result(future.result())
            task = indexed_tasks[index][1]
            task_meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
            relative_path = str(task_meta.get("relative_path") or "").strip()
            if relative_path:
                result["relative_path"] = relative_path
            task_id = str(result.get("task_id") or task.get("id") or "unknown")
            result["task_id"] = task_id
            if relative_path:
                rel_path = Path(relative_path)
                if len(rel_path.parts) >= 3:
                    task_type = rel_path.parts[0]
                    main_topic = rel_path.parts[1]
                    subtopic = rel_path.stem
                    result["task_key"] = f"{task_type}/{main_topic}/{subtopic}/{task_id}"
            ordered_results[index] = result
            if result.get("success"):
                success_count += 1
            else:
                failed_count += 1
            final_answer = str(result.get("final_answer") or "")
            short_answer = final_answer if len(final_answer) <= 24 else final_answer[:21] + "..."
            pbar.update(1)
            pbar.set_postfix(
                {
                    "ok": success_count,
                    "fail": failed_count,
                    "last": f"{task_id}:{short_answer}",
                },
                refresh=True,
            )

    tqdm.write(
        f"✅ Batch complete: total={len(tasks)} ok={success_count} fail={failed_count} duration={round(time.time() - started, 3)}s"
    )

    results = [item for item in ordered_results if item is not None]
    return {
        "total": len(tasks),
        "success": success_count,
        "failed": len(tasks) - success_count,
        "success_rate": f"{success_count/len(tasks)*100:.1f}%" if tasks else "0%",
        "results": results,
        "duration_sec_total": round(time.time() - started, 3),
    }


class LocalEvalScopeAdapter:
    def __init__(
        self,
        tasks_dir: Path | str,
        tools_dir: Path | str,
        pattern: str = "**/*.json",
        max_rounds: int = 30,
        max_workers: int = 1,
        covered_only: bool = True,
        llm_model: str = LLM_MODEL,
        llm_temperature: Optional[float] = LLM_TEMPERATURE,
        llm_api_url: str = "",
        llm_api_key: str = "",
        llm_request_timeout: float = 120.0,
        llm_extra_body: Optional[Dict[str, Any]] = None,
        clean_output_path: Optional[Path | str] = None,
    ):
        self.tasks_dir = Path(tasks_dir)
        self.tools_dir = Path(tools_dir)
        self.pattern = pattern
        self.max_rounds = max_rounds
        self.max_workers = max_workers
        self.covered_only = covered_only
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key
        self.llm_request_timeout = llm_request_timeout
        self.llm_extra_body = llm_extra_body
        self.clean_output_path = Path(clean_output_path) if clean_output_path else None

    def _base_meta(self, mode: str, task_id: Optional[str] = None, hint_catalog: Optional[Path | str] = None, hint_injection_mode: Optional[str] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "tasks_dir": str(self.tasks_dir),
            "tools_dir": str(self.tools_dir),
            "pattern": self.pattern,
            "task_id": task_id,
            "max_rounds": self.max_rounds,
            "max_workers": self.max_workers,
            "mode": mode,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "fail_seed": os.getenv("FAIL_SEED"),
        }
        if self.llm_extra_body is not None:
            meta["llm_extra_body"] = self.llm_extra_body
        if hint_catalog is not None:
            meta["hint_catalog"] = str(hint_catalog)
        if hint_injection_mode is not None:
            meta["hint_injection_mode"] = hint_injection_mode
        return meta

    def _discover_tool_path_silent(self, relative_path: str, task_id: Optional[str] = None) -> Optional[Path]:
        return discover_tool_path(
            self.tools_dir,
            relative_path,
            str(task_id) if task_id is not None else None,
            silent=True,
        )

    def _filter_covered_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for task in tasks:
            meta = task.get("_meta", {}) if isinstance(task.get("_meta"), dict) else {}
            relative_path = meta.get("relative_path")
            task_id = task.get("id")
            if not relative_path:
                continue
            if self._discover_tool_path_silent(str(relative_path), str(task_id) if task_id is not None else None):
                filtered.append(task)
        return filtered

    def load_tasks(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        tasks = _load_tasks_from_cli(self.tasks_dir, self.pattern)
        if task_id:
            tasks = [t for t in tasks if str(t.get("id")) == str(task_id)]
        if self.covered_only:
            before = len(tasks)
            tasks = self._filter_covered_tasks(tasks)
            print(f"🧩 Covered-only filter: kept {len(tasks)}/{before} tasks matched to tools in {self.tools_dir}", flush=True)
        return tasks

    def run_baseline(self, task_id: Optional[str] = None, output_path: Optional[Path | str] = None) -> Dict[str, Any]:
        tasks = self.load_tasks(task_id=task_id)
        results = _run_batch_parallel(
            tasks,
            self.tools_dir,
            self.max_rounds,
            self.max_workers,
            self.llm_model,
            self.llm_temperature,
            self.llm_api_url,
            self.llm_api_key,
            self.llm_request_timeout,
            self.llm_extra_body,
        )
        payload = {
            "meta": self._base_meta("baseline", task_id=task_id),
            "coverage": compute_tool_coverage(tasks, self.tools_dir),
            "results": results,
            "exact_match": compute_exact_match_metrics(results),
        }
        self._maybe_write(output_path, payload)
        self._maybe_write_clean(payload, output_path)
        return payload

    def run_with_hints(
        self,
        hint_catalog: Path | str,
        hint_injection_mode: str = "deferred_on_first_error",
        task_id: Optional[str] = None,
        output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        tasks = self.load_tasks(task_id=task_id)
        hint_catalog_data = _load_hint_catalog(Path(hint_catalog))
        prepared_tasks, hint_stats = _attach_hints(tasks, hint_catalog_data, hint_injection_mode)
        results = _run_batch_parallel(
            prepared_tasks,
            self.tools_dir,
            self.max_rounds,
            self.max_workers,
            self.llm_model,
            self.llm_temperature,
            self.llm_api_url,
            self.llm_api_key,
            self.llm_request_timeout,
            self.llm_extra_body,
        )
        payload = {
            "meta": self._base_meta("with_hint", task_id=task_id, hint_catalog=hint_catalog, hint_injection_mode=hint_injection_mode),
            "hint_stats": hint_stats,
            "coverage": compute_tool_coverage(prepared_tasks, self.tools_dir),
            "results": results,
            "exact_match": compute_exact_match_metrics(results),
        }
        self._maybe_write(output_path, payload)
        self._maybe_write_clean(payload, output_path)
        return payload

    def run_oracle_label(
        self,
        hazard_manifest: Path | str,
        task_id: Optional[str] = None,
        output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        tasks = self.load_tasks(task_id=task_id)
        prepared_tasks, oracle_label_stats = _attach_oracle_hazard_labels(
            tasks,
            hazard_manifest,
        )
        results = _run_batch_parallel(
            prepared_tasks,
            self.tools_dir,
            self.max_rounds,
            self.max_workers,
            self.llm_model,
            self.llm_temperature,
            self.llm_api_url,
            self.llm_api_key,
            self.llm_request_timeout,
            self.llm_extra_body,
        )
        payload = {
            "meta": {
                **self._base_meta("oracle_label", task_id=task_id),
                "hazard_manifest": str(hazard_manifest),
                "oracle_label_injection_mode": "deferred_on_first_observable_anomaly",
            },
            "oracle_label_stats": oracle_label_stats,
            "coverage": compute_tool_coverage(prepared_tasks, self.tools_dir),
            "results": results,
            "exact_match": compute_exact_match_metrics(results),
        }
        self._maybe_write(output_path, payload)
        self._maybe_write_clean(payload, output_path)
        return payload

    def run_ab(
        self,
        hint_catalog: Path | str,
        hint_injection_mode: str = "deferred_on_first_error",
        task_id: Optional[str] = None,
        output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        base_tasks = self.load_tasks(task_id=task_id)
        hint_catalog_data = _load_hint_catalog(Path(hint_catalog))

        no_hint_tasks, no_hint_stats = _prepare_tasks_for_mode(base_tasks, "no_hint", hint_catalog_data, hint_injection_mode)
        with_hint_tasks, with_hint_stats = _prepare_tasks_for_mode(base_tasks, "with_hint", hint_catalog_data, hint_injection_mode)

        no_hint_results = _run_batch_parallel(
            no_hint_tasks,
            self.tools_dir,
            self.max_rounds,
            self.max_workers,
            self.llm_model,
            self.llm_temperature,
            self.llm_api_url,
            self.llm_api_key,
            self.llm_request_timeout,
            self.llm_extra_body,
        )
        with_hint_results = _run_batch_parallel(
            with_hint_tasks,
            self.tools_dir,
            self.max_rounds,
            self.max_workers,
            self.llm_model,
            self.llm_temperature,
            self.llm_api_url,
            self.llm_api_key,
            self.llm_request_timeout,
            self.llm_extra_body,
        )

        no_hint_results["exact_match"] = compute_exact_match_metrics(no_hint_results)
        no_hint_results["coverage"] = compute_tool_coverage(no_hint_tasks, self.tools_dir)
        no_hint_results["mode"] = "no_hint"

        with_hint_results["exact_match"] = compute_exact_match_metrics(with_hint_results)
        with_hint_results["coverage"] = compute_tool_coverage(with_hint_tasks, self.tools_dir)
        with_hint_results["mode"] = "with_hint"

        payload = {
            "meta": self._base_meta("ab", task_id=task_id, hint_catalog=hint_catalog, hint_injection_mode=hint_injection_mode),
            "prep_stats": {"no_hint": no_hint_stats, "with_hint": with_hint_stats},
            "mode_results": {"no_hint": no_hint_results, "with_hint": with_hint_results},
            "comparison": _build_ab_comparison(no_hint_results, with_hint_results),
        }
        self._maybe_write(output_path, payload)
        self._maybe_write_clean(payload, output_path)
        return payload

    @staticmethod
    def _maybe_write(output_path: Optional[Path | str], payload: Dict[str, Any]) -> None:
        _write_json(output_path, payload)

    def _maybe_write_clean(self, payload: Dict[str, Any], output_path: Optional[Path | str]) -> None:
        if not self.clean_output_path:
            return
        source_name = Path(output_path).name if output_path else "eval.json"
        clean_payload = clean_eval_payload(payload, source_name)
        _write_json(self.clean_output_path, clean_payload)


def run_from_task_config(
    task_cfg: Any,
    *,
    tasks_dir: Path | str,
    tools_dir: Path | str,
    mode: str = "baseline",
    pattern: str = "**/*.json",
    hint_catalog: Optional[Path | str] = None,
    hazard_manifest: Optional[Path | str] = None,
    hint_injection_mode: str = "deferred_on_first_error",
    task_id: Optional[str] = None,
    output_path: Optional[Path | str] = None,
    clean_output_path: Optional[Path | str] = None,
    max_rounds: int = 30,
    covered_only: bool = True,
    request_timeout: float = 120.0,
) -> Dict[str, Any]:
    max_workers = int(_task_cfg_get(task_cfg, "eval_batch_size", 1) or 1)
    adapter = LocalEvalScopeAdapter(
        tasks_dir=tasks_dir,
        tools_dir=tools_dir,
        pattern=pattern,
        max_rounds=max_rounds,
        max_workers=max_workers,
        covered_only=covered_only,
        llm_model=str(_task_cfg_get(task_cfg, "model", LLM_MODEL) or LLM_MODEL),
        llm_api_url=str(_task_cfg_get(task_cfg, "api_url", "") or ""),
        llm_api_key=str(_task_cfg_get(task_cfg, "api_key", "") or ""),
        llm_temperature=(
            float((_task_cfg_get(task_cfg, "generation_config", {}) or {})["temperature"])
            if isinstance(_task_cfg_get(task_cfg, "generation_config", {}), dict)
            and "temperature" in (_task_cfg_get(task_cfg, "generation_config", {}) or {})
            and (_task_cfg_get(task_cfg, "generation_config", {}) or {})["temperature"] is not None
            else None
        ),
        llm_request_timeout=request_timeout,
        llm_extra_body=(
            (_task_cfg_get(task_cfg, "generation_config", {}) or {}).get("extra_body")
            if isinstance(_task_cfg_get(task_cfg, "generation_config", {}), dict)
            else None
        ),
        clean_output_path=clean_output_path,
    )

    if mode == "baseline":
        payload = adapter.run_baseline(task_id=task_id, output_path=output_path)
    elif mode == "with_hint":
        if not hint_catalog:
            raise ValueError("hint_catalog is required for mode='with_hint'")
        payload = adapter.run_with_hints(
            hint_catalog=hint_catalog,
            hint_injection_mode=hint_injection_mode,
            task_id=task_id,
            output_path=output_path,
        )
    elif mode == "oracle_label":
        if not hazard_manifest:
            raise ValueError("hazard_manifest is required for mode='oracle_label'")
        payload = adapter.run_oracle_label(
            hazard_manifest=hazard_manifest,
            task_id=task_id,
            output_path=output_path,
        )
    elif mode == "ab":
        if not hint_catalog:
            raise ValueError("hint_catalog is required for mode='ab'")
        payload = adapter.run_ab(
            hint_catalog=hint_catalog,
            hint_injection_mode=hint_injection_mode,
            task_id=task_id,
            output_path=output_path,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return payload
