import argparse
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
from prompt_gentask import (
    prompt_markdown_mixture,
    prompt_markdown_parallel,
    prompt_markdown_sequential,
)

load_dotenv(Path(__file__).resolve().parent / "config" / ".env")

_thread_local = threading.local()
checkpoint = None


class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.lock = threading.Lock()
        self.completed_tasks: Set[str] = set()
        self.results: List[Dict[str, Any]] = []
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

    def record_result(self, task_key: str, result: Dict[str, Any]) -> None:
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

    def reset(self) -> None:
        with self.lock:
            self.completed_tasks.clear()
            self.results.clear()
            self.paused = False
            self._save_unlocked()
            print("🔄 已重置所有进度")

    def get_stats(self) -> Dict[str, Any]:
        return {"completed": len(self.completed_tasks), "paused": self.paused}


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
class TaskGenerationResult:
    task_key: str
    task_type: str
    main_topic: str
    subtopic: str
    filepath: str
    success: bool
    error: Optional[str] = None
    duration: float = 0.0
    generated_task_count: int = 0


def safe_path_segment(value: str) -> str:
    return re.sub(r"[^\w\s-]", "", str(value)).strip().replace(" ", "_") or "unknown"


def build_task_key(task_type: str, main_topic: str, subtopic: str) -> str:
    return f"{task_type}/{main_topic}/{subtopic}"


def parse_response_to_json(response: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.MULTILINE)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    json_str = match.group(0) if match else cleaned
    data = json.loads(json_str)
    if not isinstance(data, dict) or "tasks" not in data or not isinstance(data["tasks"], list):
        raise ValueError("Response missing valid 'tasks' list")
    return data


def normalize_task_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    tasks = data.get("tasks", [])
    normalized = []
    for idx, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {idx} is not an object")
        item = dict(task)
        item["id"] = f"task_{idx}"
        normalized.append(item)
    if len(normalized) != 5:
        raise ValueError(f"Expected exactly 5 tasks, got {len(normalized)}")
    return {"tasks": normalized}


def save_generated_tasks(
    data: Dict[str, Any],
    task_type: str,
    main_topic: str,
    subtopic: str,
    output_dir: str,
) -> Tuple[bool, str]:
    try:
        safe_tt = safe_path_segment(task_type)
        safe_mt = safe_path_segment(main_topic)
        safe_st = safe_path_segment(subtopic)
        dir_path = os.path.join(output_dir, safe_tt, safe_mt)
        filepath = os.path.join(dir_path, f"{safe_st}.json")
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return True, filepath
    except Exception as e:
        return False, str(e)


def build_task_prompt(task_type: str, main_topic: str, subtopic: str) -> str:
    template_map = {
        "sequential": prompt_markdown_sequential,
        "parallel": prompt_markdown_parallel,
        "mixture": prompt_markdown_mixture,
    }
    if task_type not in template_map:
        raise ValueError(f"未知 task_type: {task_type}")
    prompt = template_map[task_type]
    prompt = re.sub(r"## Main Topic\s*\*\*[^*]+\*\*", f"## Main Topic\n**{main_topic}**", prompt)
    prompt = re.sub(r"## Subtopics\s*-\s*[^\n]+", f"## Subtopics\n- {subtopic}", prompt)
    return prompt


def load_topic_pairs(topics_file: str, task_types: List[str]) -> List[Dict[str, str]]:
    with open(topics_file, "r", encoding="utf-8") as f:
        topics_config = json.load(f)
    pairs: List[Dict[str, str]] = []
    for task_type in task_types:
        for main_topic, subtopics in topics_config.items():
            if not isinstance(subtopics, list):
                continue
            for subtopic in subtopics:
                pairs.append(
                    {
                        "task_key": build_task_key(task_type, str(main_topic), str(subtopic)),
                        "task_type": task_type,
                        "main_topic": str(main_topic),
                        "subtopic": str(subtopic),
                    }
                )
    return pairs


def select_topic_pairs_balanced_by_type(
    all_pairs: List[Dict[str, str]],
    max_tasks: Optional[int],
    task_types: List[str],
) -> List[Dict[str, str]]:
    if max_tasks is None or max_tasks <= 0 or not all_pairs:
        return all_pairs

    ordered_types = [tt for tt in task_types if tt in {str(item.get("task_type")) for item in all_pairs}]
    buckets: Dict[str, List[Dict[str, str]]] = {tt: [] for tt in ordered_types}
    for item in all_pairs:
        tt = str(item.get("task_type", ""))
        if tt in buckets:
            buckets[tt].append(item)

    for tt in ordered_types:
        buckets[tt].sort(key=lambda item: str(item.get("task_key", "")))

    total_available = sum(len(bucket) for bucket in buckets.values())
    target_total = min(max_tasks, total_available)
    if target_total <= 0:
        return []

    base_quota = target_total // len(ordered_types)
    remainder = target_total % len(ordered_types)
    quotas = {tt: base_quota + (1 if idx < remainder else 0) for idx, tt in enumerate(ordered_types)}

    selected: List[Dict[str, str]] = []
    selected_keys = set()
    for tt in ordered_types:
        for item in buckets[tt][: quotas[tt]]:
            key = str(item.get("task_key", ""))
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)

    cursor = {tt: quotas[tt] for tt in ordered_types}
    while len(selected) < target_total:
        progressed = False
        for tt in ordered_types:
            bucket = buckets[tt]
            while cursor[tt] < len(bucket):
                item = bucket[cursor[tt]]
                cursor[tt] += 1
                key = str(item.get("task_key", ""))
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                progressed = True
                break
            if len(selected) >= target_total:
                break
        if not progressed:
            break

    return selected


def generate_single_task_file(
    topic_info: Dict[str, str],
    output_dir: str,
    retry_count: int = 2,
    rate_limit_delay: float = 0.5,
) -> TaskGenerationResult:
    start_time = time.time()
    task_key = str(topic_info.get("task_key", ""))
    task_type = str(topic_info.get("task_type", ""))
    main_topic = str(topic_info.get("main_topic", ""))
    subtopic = str(topic_info.get("subtopic", ""))

    try:
        llm = get_thread_llm()
        prompt = build_task_prompt(task_type, main_topic, subtopic)
        response = None
        last_error = None

        for attempt in range(retry_count + 1):
            try:
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
                response = llm.get_completion(prompt)
                if response and not str(response).startswith("Error:"):
                    break
                last_error = str(response)[:200]
            except Exception as e:
                last_error = str(e)[:200]
                if attempt < retry_count:
                    time.sleep(1.0 * (attempt + 1))

        if not response or str(response).startswith("Error:"):
            return TaskGenerationResult(
                task_key=task_key,
                task_type=task_type,
                main_topic=main_topic,
                subtopic=subtopic,
                filepath="",
                success=False,
                error=f"LLM 生成失败：{last_error}",
                duration=time.time() - start_time,
            )

        parsed = parse_response_to_json(str(response))
        normalized = normalize_task_payload(parsed)
        success, result = save_generated_tasks(normalized, task_type, main_topic, subtopic, output_dir)
        if success:
            return TaskGenerationResult(
                task_key=task_key,
                task_type=task_type,
                main_topic=main_topic,
                subtopic=subtopic,
                filepath=result,
                success=True,
                duration=time.time() - start_time,
                generated_task_count=len(normalized.get("tasks", [])),
            )
        return TaskGenerationResult(
            task_key=task_key,
            task_type=task_type,
            main_topic=main_topic,
            subtopic=subtopic,
            filepath="",
            success=False,
            error=f"保存失败：{result}",
            duration=time.time() - start_time,
        )
    except Exception as e:
        return TaskGenerationResult(
            task_key=task_key,
            task_type=task_type,
            main_topic=main_topic,
            subtopic=subtopic,
            filepath="",
            success=False,
            error=str(e),
            duration=time.time() - start_time,
        )


def gentask_parallel(
    topics_file: str = "topic.json",
    output_dir: str = "tasks",
    task_types: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    max_workers: int = 5,
    retry_count: int = 2,
    rate_limit_delay: float = 0.5,
    checkpoint_path: Optional[str] = None,
    reset_checkpoint: bool = False,
) -> Dict[str, Any]:
    if task_types is None:
        task_types = ["sequential", "parallel", "mixture"]

    global checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    checkpoint = CheckpointManager(checkpoint_path)
    if reset_checkpoint:
        checkpoint.reset()

    stats = checkpoint.get_stats()
    print(f"📂 当前进度：{stats['completed']} 个任务已完成")
    if stats["paused"]:
        checkpoint.paused = False
        checkpoint.save()

    print("收集 topics...")
    all_pairs = load_topic_pairs(topics_file, task_types)
    all_pairs = [item for item in all_pairs if not checkpoint.is_completed(str(item.get("task_key", "")))]
    all_pairs = select_topic_pairs_balanced_by_type(all_pairs, max_tasks, task_types)

    if not all_pairs:
        print("没有找到任何待生成 topics!")
        return {"total": 0, "success": 0, "failed": 0}

    print(f"共发现 {len(all_pairs)} 个 topic 生成任务")
    if max_tasks is not None and max_tasks > 0:
        print(f"任务上限：{max_tasks}")
    print(f"并发数：{max_workers}, 重试次数：{retry_count}")

    results: List[TaskGenerationResult] = []
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        future_to_task = {
            executor.submit(
                generate_single_task_file,
                item,
                output_dir,
                retry_count,
                rate_limit_delay,
            ): item
            for item in all_pairs
        }

        with tqdm(total=len(all_pairs), desc="🧠 生成 tasks", unit="topic") as pbar:
            for future in as_completed(future_to_task):
                topic_info = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        success_count += 1
                        pbar.set_postfix({"✅": success_count, "❌": failed_count})
                    else:
                        failed_count += 1
                        print(f"\n  {topic_info['task_key']}: {result.error}")
                    checkpoint.record_result(
                        str(topic_info.get("task_key", result.task_key)),
                        {
                            "task_key": str(topic_info.get("task_key", result.task_key)),
                            "task_type": result.task_type,
                            "main_topic": result.main_topic,
                            "subtopic": result.subtopic,
                            "filepath": result.filepath,
                            "success": result.success,
                            "error": result.error,
                            "duration": round(result.duration, 2),
                            "generated_task_count": result.generated_task_count,
                        },
                    )
                    pbar.update(1)
                except Exception as e:
                    failed_count += 1
                    print(f"\n  异常：{e}")
                    pbar.update(1)

    summary = {
        "total": len(all_pairs) + stats["completed"],
        "success": success_count + stats["completed"],
        "failed": failed_count,
        "success_rate": f"{(success_count + stats['completed'])/max(1, len(all_pairs) + stats['completed'])*100:.1f}%",
        "results": [
            {
                "task_key": r.task_key,
                "task_type": r.task_type,
                "main_topic": r.main_topic,
                "subtopic": r.subtopic,
                "filepath": r.filepath,
                "success": r.success,
                "error": r.error,
                "duration": round(r.duration, 2),
                "generated_task_count": r.generated_task_count,
            }
            for r in results
        ],
        "checkpoint": checkpoint.get_stats(),
    }

    print(f"\n{'=' * 60}")
    print("生成完成!")
    print(f"{'=' * 60}")
    print(f"总任务数：   {summary['total']}")
    print(f"成功：       {summary['success']} ✅")
    print(f"失败：       {summary['failed']}")
    print(f"成功率：     {summary['success_rate']}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "generation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"报告已保存：{report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行生成 tasks")
    parser.add_argument("--topics-file", type=str, default="topic.json")
    parser.add_argument("--output-dir", type=str, default="tasks")
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        default=[],
        help="Task type to include (repeatable). Defaults to sequential, parallel, mixture.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of selected topic-generation jobs")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--rate-limit-delay", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint state before running")
    args = parser.parse_args()

    gentask_parallel(
        topics_file=args.topics_file,
        output_dir=args.output_dir,
        task_types=args.task_types or ["sequential", "parallel", "mixture"],
        max_tasks=args.max_tasks,
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        rate_limit_delay=args.rate_limit_delay,
        checkpoint_path=args.checkpoint_path,
        reset_checkpoint=args.reset,
    )
