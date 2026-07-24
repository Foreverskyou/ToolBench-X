#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


KEEP_FIELDS = [
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
    "oracle_hazard_label",
    "oracle_label_injected",
    "openai_trajectory",
]


MetadataIndex = Dict[Tuple[str, str], Dict[str, str]]


def _topic_fields(relative_path: Any) -> Dict[str, str]:
    parts = str(relative_path or "").split("/")
    topic = parts[1] if len(parts) >= 2 else ""
    subtopic = Path(parts[2]).stem if len(parts) >= 3 else ""
    return {"topic": topic, "subtopic": subtopic}


def _metadata_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("task_id") or ""), str(item.get("user_prompt") or ""))


def _index_result_metadata(item: Dict[str, Any], metadata_index: MetadataIndex) -> None:
    relative_path = str(item.get("relative_path") or "").strip()
    if not relative_path:
        return
    relative_parts = Path(relative_path).parts
    key = _metadata_key(item)
    if not key[0] or not key[1] or key in metadata_index:
        return
        metadata_index[key] = {
            "task_type": str(item.get("task_type") or (relative_parts[0] if relative_parts else "")),
            "exception_type": str(item.get("exception_type") or ""),
            "relative_path": relative_path,
            **_topic_fields(relative_path),
        }


def _iter_results_from_payload(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    mode_results = payload.get("mode_results")
    if isinstance(mode_results, dict):
        for mode_payload in mode_results.values():
            if isinstance(mode_payload, dict):
                for item in mode_payload.get("results", []) or []:
                    if isinstance(item, dict):
                        yield item
        return
    results_payload = payload.get("results")
    if isinstance(results_payload, dict):
        for item in results_payload.get("results", []) or []:
            if isinstance(item, dict):
                yield item


def _build_metadata_index(metadata_sources: List[Path]) -> MetadataIndex:
    index: MetadataIndex = {}
    for source in metadata_sources:
        try:
            payload = _load_json(source) if source.is_file() else None
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            for manifest_item in payload.get("items", []):
                if not isinstance(manifest_item, dict):
                    continue
                task_id = str(manifest_item.get("task_id") or "")
                relative_path = str(manifest_item.get("relative_path") or "")
                if not task_id or not relative_path:
                    continue
                metadata = {
                    "task_type": str(manifest_item.get("task_type") or Path(relative_path).parts[0]),
                    "exception_type": str(manifest_item.get("exception_type") or ""),
                    "relative_path": relative_path,
                    **_topic_fields(relative_path),
                }
                index[(task_id, relative_path)] = metadata
                index[(task_id, str(manifest_item.get("user_prompt") or ""))] = metadata
            continue
        for path in _iter_input_files(source, skip_merged=False):
            try:
                payload = _load_json(path)
            except Exception:
                continue
            for item in _iter_results_from_payload(payload):
                _index_result_metadata(item, index)
    return index


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _clean_result(item: Dict[str, Any], metadata_index: MetadataIndex | None = None) -> Dict[str, Any]:
    metadata = (metadata_index or {}).get(
        (str(item.get("task_id") or ""), str(item.get("relative_path") or "")),
        (metadata_index or {}).get(_metadata_key(item), {}),
    )
    relative_path = item.get("relative_path") or metadata.get("relative_path")
    topic_fields = _topic_fields(relative_path)
    enriched = {**item, **metadata, **topic_fields, "relative_path": relative_path}
    return {field: enriched.get(field) for field in KEEP_FIELDS}


def _clean_mode(mode_payload: Dict[str, Any], metadata_index: MetadataIndex | None = None) -> Dict[str, Any]:
    results = mode_payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("mode_results entry must contain a list-valued results field")
    cleaned_results = [_clean_result(item, metadata_index) for item in results if isinstance(item, dict)]
    return {
        "total": len(cleaned_results),
        "success": sum(1 for item in cleaned_results if item.get("success")),
        "failed": sum(1 for item in cleaned_results if not item.get("success")),
        "success_rate": _format_rate(sum(1 for item in cleaned_results if item.get("success")), len(cleaned_results)),
        "results": cleaned_results,
    }


def _format_rate(success: int, total: int) -> str:
    return f"{success / total * 100:.1f}%" if total else "0.0%"


def clean_ab_payload(payload: Dict[str, Any], source_name: str, metadata_index: MetadataIndex | None = None) -> Dict[str, Any]:
    mode_results = payload.get("mode_results")
    if not isinstance(mode_results, dict):
        raise ValueError(f"{source_name} is not an AB payload with mode_results")

    clean_modes: Dict[str, Any] = {}
    for mode in ["no_hint", "with_hint"]:
        mode_payload = mode_results.get(mode)
        if isinstance(mode_payload, dict):
            clean_modes[mode] = _clean_mode(mode_payload, metadata_index)
    if not clean_modes:
        raise ValueError(f"{source_name} has no cleanable AB mode_results")

    return {
        "format": "openai_trajectory_clean_ab",
        "source_file": source_name,
        "kept_fields": KEEP_FIELDS,
        "meta": payload.get("meta", {}),
        "prep_stats": payload.get("prep_stats", {}),
        "mode_results": clean_modes,
    }


def clean_single_mode_payload(payload: Dict[str, Any], source_name: str, metadata_index: MetadataIndex | None = None) -> Dict[str, Any]:
    results_payload = payload.get("results")
    if not isinstance(results_payload, dict):
        raise ValueError(f"{source_name} is not a single-mode payload with results")
    mode = str(payload.get("meta", {}).get("mode") or "baseline")
    return {
        "format": f"openai_trajectory_clean_{mode}",
        "source_file": source_name,
        "kept_fields": KEEP_FIELDS,
        "meta": payload.get("meta", {}),
        "results": _clean_mode(results_payload, metadata_index),
    }


def clean_eval_payload(payload: Dict[str, Any], source_name: str, metadata_index: MetadataIndex | None = None) -> Dict[str, Any]:
    if isinstance(payload.get("mode_results"), dict):
        return clean_ab_payload(payload, source_name, metadata_index)
    return clean_single_mode_payload(payload, source_name, metadata_index)


def merge_clean_payloads(payloads: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    payloads = list(payloads)
    if not payloads:
        return {
            "format": "openai_trajectory_clean_merged",
            "source_files": [],
            "kept_fields": KEEP_FIELDS,
            "results": {"total": 0, "success": 0, "failed": 0, "success_rate": "0.0%", "results": []},
        }
    if all("mode_results" not in payload for payload in payloads):
        results: List[Dict[str, Any]] = []
        source_files: List[str] = []
        for payload in payloads:
            source_files.append(str(payload.get("source_file") or ""))
            results.extend(payload["results"]["results"])
        success = sum(1 for item in results if item.get("success"))
        total = len(results)
        return {
            "format": "openai_trajectory_clean_single_mode_merged",
            "source_files": source_files,
            "kept_fields": KEEP_FIELDS,
            "results": {
                "total": total,
                "success": success,
                "failed": total - success,
                "success_rate": _format_rate(success, total),
                "results": results,
            },
        }

    merged_results: Dict[str, List[Dict[str, Any]]] = {"no_hint": [], "with_hint": []}
    source_files: List[str] = []
    for payload in payloads:
        source_files.append(str(payload.get("source_file") or ""))
        for mode in ["no_hint", "with_hint"]:
            if mode in payload["mode_results"]:
                merged_results[mode].extend(payload["mode_results"][mode]["results"])

    mode_results: Dict[str, Any] = {}
    for mode, results in merged_results.items():
        success = sum(1 for item in results if item.get("success"))
        total = len(results)
        mode_results[mode] = {
            "total": total,
            "success": success,
            "failed": total - success,
            "success_rate": _format_rate(success, total),
            "results": results,
        }

    return {
        "format": "openai_trajectory_clean_ab_merged",
        "source_files": source_files,
        "kept_fields": KEEP_FIELDS,
        "mode_results": mode_results,
    }


def _summary_counts(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("mode_results"), dict):
        return {
            mode: {
                "total": mode_payload.get("total"),
                "success": mode_payload.get("success"),
                "failed": mode_payload.get("failed"),
                "success_rate": mode_payload.get("success_rate"),
            }
            for mode, mode_payload in payload["mode_results"].items()
            if isinstance(mode_payload, dict)
        }
    results = payload.get("results", {})
    return {
        "total": results.get("total"),
        "success": results.get("success"),
        "failed": results.get("failed"),
        "success_rate": results.get("success_rate"),
    }


def _iter_input_files(input_path: Path, skip_merged: bool = True) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.json") if not (skip_merged and path.name.startswith("merged_")))
    raise FileNotFoundError(input_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export clean eval results containing only OpenAI trajectories and core task fields")
    parser.add_argument("--input", type=Path, required=True, help="Eval result JSON file or directory of eval JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for clean exports")
    parser.add_argument("--merged-name", type=str, default="merged_openai_trajectory_clean.json")
    parser.add_argument("--manifest-name", type=str, default="openai_trajectory_clean_manifest.json")
    parser.add_argument("--metadata-source", type=Path, action="append", default=[], help="Optional eval JSON file/dir used to backfill relative_path/topic/subtopic for old results")
    args = parser.parse_args()

    input_files = _iter_input_files(args.input)
    metadata_index = _build_metadata_index(args.metadata_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_payloads: List[Dict[str, Any]] = []
    written_files: List[str] = []
    for input_file in input_files:
        clean_payload = clean_eval_payload(_load_json(input_file), input_file.name, metadata_index)
        clean_payloads.append(clean_payload)
        output_file = args.output_dir / f"{input_file.stem}_openai_trajectory_clean.json"
        output_file.write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        written_files.append(str(output_file))

    merged_payload = merge_clean_payloads(clean_payloads)
    merged_file = args.output_dir / args.merged_name
    merged_file.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    written_files.append(str(merged_file))

    manifest = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "source_files": [str(path) for path in input_files],
        "written_files": written_files,
        "kept_fields": KEEP_FIELDS,
        "metadata_sources": [str(path) for path in args.metadata_source],
        "metadata_index_count": len(metadata_index),
        "merged_counts": _summary_counts(merged_payload),
    }
    manifest_file = args.output_dir / args.manifest_name
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
