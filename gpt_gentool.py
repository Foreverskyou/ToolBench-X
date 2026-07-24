# gentool_parallel.py
import argparse
import json
import os
import re
import signal
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from tqdm import tqdm

from gpt import GPT54
from prompt_gentool import prompt_from_zh

load_dotenv(Path(__file__).resolve().parent / "config" / ".env")


# ==================== 线程安全核心 ====================
# 使用 threading.local() 为每个线程创建独立的 GPT54 实例
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
        return {
            "completed": len(self.completed_tasks),
            "paused": self.paused,
        }


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
    """获取当前线程的 GPT54 实例（不存在则创建）."""
    if not hasattr(_thread_local, 'llm'):
        _thread_local.llm = GPT54()
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] 创建新的 GPT54 实例")
    return _thread_local.llm


def worker_initializer():
    """线程池初始化钩子：为每个工作线程创建独立的 GPT54 实例."""
    thread_name = threading.current_thread().name
    # 预先创建实例，避免首次调用时的延迟
    _thread_local.llm = GPT54()
    print(f"[{thread_name}] 工作线程初始化完成")


# ==================== 数据类 ====================
@dataclass
class GenerationResult:
    """单个 task 的生成结果."""
    task_key: str
    task_id: str
    filepath: str
    success: bool
    error: Optional[str] = None
    duration: float = 0.0


# ==================== 工具函数 ====================
def save_mock_code(
    code_content: str,
    task_type: str,
    main_topic: str,
    subtopic: str,
    task_id: str,
    base_dir: str = "tools",
) -> Tuple[bool, str]:
    """
    保存 mock 函数代码为 Python 文件
    :return: (success, filepath_or_error)
    """
    try:
        # 1️⃣ 清理代码：移除 ```python 或 ``` 标记
        cleaned = re.sub(r'^```(?:python)?\s*|\s*```$', '', code_content.strip(), flags=re.MULTILINE)
        
        # 2️⃣ 生成安全路径组件
        safe_tt = re.sub(r'[^\w\s-]', '', task_type).strip().replace(' ', '_')
        safe_mt = re.sub(r'[^\w\s-]', '', main_topic).strip().replace(' ', '_')
        safe_st = re.sub(r'[^\w\s-]', '', subtopic).strip().replace(' ', '_')
        safe_task_id = re.sub(r"[^\w\s-]", "", str(task_id)).strip().replace(" ", "_") or "unknown"

        dir_path = os.path.join(base_dir, safe_tt, safe_mt, safe_st)
        filename = f"{safe_task_id}.py"
        filepath = os.path.join(dir_path, filename)
        
        # 3️⃣ 创建目录 + 保存文件
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cleaned)
        
        return True, filepath
        
    except Exception as e:
        return False, str(e)


def build_mock_prompt(task_item: dict, task_type: str, main_topic: str, subtopic: str) -> str:
    """构建 mock prompt."""
    task_id = task_item.get("id", "unknown")
    user_prompt = task_item.get("user_prompt", "")
    tools_used = task_item.get("tools_used", [])
    expected_answer = task_item.get("expected_answer", "")
    final_answer = task_item.get("final_answer", "")
    
    new_data_json = json.dumps({
        "task_type": task_type,
        "main_topic": main_topic,
        "subtopic": subtopic,
        "id": task_id,
        "user_prompt": user_prompt,
        "tools_used": tools_used,
        "expected_answer": expected_answer,
        "final_answer": final_answer
    }, ensure_ascii=False, indent=2)
    
    pattern = r'(# Data to Process\s*\n)\{[\s\S]*?\}(?=\s*\n#)'
    replacement = r'\1' + new_data_json
    result = re.sub(pattern, replacement, prompt_from_zh, flags=re.MULTILINE)
    
    if result == prompt_from_zh:
        result = re.sub(
            r'\{\s*"id":\s*"task_\d+"[\s\S]*?"final_answer":\s*"[^"]+"\s*\}',
            new_data_json,
            prompt_from_zh
        )
    
    return result


def generate_single_tool(
    task_item: dict,
    task_type: str,
    main_topic: str,
    subtopic: str,
    output_dir: str,
    retry_count: int = 2,
    rate_limit_delay: float = 0.5
) -> GenerationResult:
    """
    生成单个 tool 的核心函数（可并行执行）.
    
    注意：不再接收 llm 参数，改为从线程局部存储获取
    """
    start_time = time.time()
    task_id = task_item.get("id", "unknown")
    thread_name = threading.current_thread().name
    
    try:
        # 从线程局部存储获取 GPT54 实例（线程安全）
        llm = get_thread_llm()
        time.sleep(rate_limit_delay) 
        
        # 构建 prompt
        mock_prompt = build_mock_prompt(task_item, task_type, main_topic, subtopic)
        
        # 带重试的 LLM 调用
        code_response = None
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                code_response = llm.get_completion(mock_prompt)
                if code_response and not str(code_response).startswith("Error:"):
                    break
                last_error = str(code_response)[:100]
            except Exception as e:
                last_error = str(e)[:100]
                if attempt < retry_count:
                    time.sleep(1.0 * (attempt + 1))  # 指数退避
        
        if not code_response or str(code_response).startswith("Error:"):
            return GenerationResult(
                task_key=f"{task_type}/{main_topic}/{subtopic}/{task_id}",
                task_id=task_id,
                filepath="",
                success=False,
                error=f"LLM 生成失败：{last_error}",
                duration=time.time() - start_time,
            )
        
        # 保存文件
        success, result = save_mock_code(
            code_response, task_type, main_topic, subtopic, task_id, output_dir
        )
        
        if success:
            return GenerationResult(
                task_key=f"{task_type}/{main_topic}/{subtopic}/{task_id}",
                task_id=task_id,
                filepath=result,
                success=True,
                duration=time.time() - start_time,
            )
        else:
            return GenerationResult(
                task_key=f"{task_type}/{main_topic}/{subtopic}/{task_id}",
                task_id=task_id,
                filepath="",
                success=False,
                error=f"保存失败：{result}",
                duration=time.time() - start_time,
            )
            
    except Exception as e:
        return GenerationResult(
            task_key=f"{task_type}/{main_topic}/{subtopic}/{task_id}",
            task_id=task_id,
            filepath="",
            success=False,
            error=f"[{thread_name}] {str(e)}",
            duration=time.time() - start_time,
        )


def collect_all_tasks(
    input_dir: str,
    task_types: List[str],
) -> List[Dict]:
    """收集所有待处理的 tasks."""
    all_tasks = []
    
    for tt in task_types:
        tt_dir = os.path.join(input_dir, tt)
        if not os.path.exists(tt_dir):
            print(f"跳过：{tt_dir} 不存在")
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
                    
                    for task_item in data["tasks"]:
                        all_tasks.append({
                            "task_key": f"{tt}/{mt_dir}/{subtopic}/{task_item.get('id', 'unknown')}",
                            "task_item": task_item,
                            "task_type": tt,
                            "main_topic": mt_dir,
                            "subtopic": subtopic,
                            "source_file": filepath,
                        })
                except Exception as e:
                    print(f"读取失败 {filepath}: {e}")
    
    return all_tasks


def select_tasks_balanced_by_type(all_tasks: List[Dict], max_tasks: Optional[int], task_types: List[str]) -> List[Dict]:
    if max_tasks is None or max_tasks <= 0 or not all_tasks:
        return all_tasks

    ordered_types = [tt for tt in task_types if tt in {str(task.get("task_type")) for task in all_tasks}]
    buckets: Dict[str, List[Dict]] = {tt: [] for tt in ordered_types}
    for task in all_tasks:
        task_type = str(task.get("task_type", ""))
        if task_type in buckets:
            buckets[task_type].append(task)

    for tt in ordered_types:
        buckets[tt].sort(key=lambda item: str(item.get("task_key", "")))

    total_available = sum(len(bucket) for bucket in buckets.values())
    target_total = min(max_tasks, total_available)
    if target_total <= 0:
        return []

    base_quota = target_total // len(ordered_types)
    remainder = target_total % len(ordered_types)
    quotas: Dict[str, int] = {
        tt: base_quota + (1 if index < remainder else 0)
        for index, tt in enumerate(ordered_types)
    }

    selected: List[Dict] = []
    selected_keys = set()

    # Phase 1: per-type quota selection.
    for tt in ordered_types:
        for task in buckets[tt][: quotas[tt]]:
            task_key = str(task.get("task_key", ""))
            if task_key in selected_keys:
                continue
            selected.append(task)
            selected_keys.add(task_key)

    # Phase 2: refill from remaining tasks in round-robin order until target_total.
    cursor = {tt: quotas[tt] for tt in ordered_types}
    while len(selected) < target_total:
        progressed = False
        for tt in ordered_types:
            bucket = buckets[tt]
            while cursor[tt] < len(bucket):
                task = bucket[cursor[tt]]
                cursor[tt] += 1
                task_key = str(task.get("task_key", ""))
                if task_key in selected_keys:
                    continue
                selected.append(task)
                selected_keys.add(task_key)
                progressed = True
                break
            if len(selected) >= target_total:
                break
        if not progressed:
            break

    return selected


def gentool_parallel(
    input_dir: str = "tasks",
    output_dir: str = "tools",
    task_types: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    max_workers: int = 5,
    retry_count: int = 2,
    rate_limit_delay: float = 0.5,
    checkpoint_path: Optional[str] = None,
    reset_checkpoint: bool = False,
) -> Dict:
    """
    并行生成所有 tools（线程安全版本）
    
    Args:
        input_dir: 任务数据输入目录
        output_dir: 输出目录
        task_types: 任务类型列表
        max_tasks: 最多生成多少个 task
        max_workers: 最大并发数（受 API 速率限制）
        retry_count: 失败重试次数
        rate_limit_delay: 每个任务间的最小延迟（秒）
        checkpoint_path: checkpoint 文件路径
        reset_checkpoint: 是否重置 checkpoint
    
    Returns:
        统计信息 dict
    """
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
    
    print(f"收集 tasks...")
    all_tasks = collect_all_tasks(input_dir, task_types)
    all_tasks = [task for task in all_tasks if not checkpoint.is_completed(str(task.get("task_key", "")))]
    all_tasks = select_tasks_balanced_by_type(all_tasks, max_tasks, task_types)
    
    if not all_tasks:
        print("没有找到任何 tasks!")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"共发现 {len(all_tasks)} 个 tasks")
    if max_tasks is not None and max_tasks > 0:
        print(f"任务上限：{max_tasks}")
    print(f"并发数：{max_workers}, 重试次数：{retry_count}")
    
    results: List[GenerationResult] = []
    success_count = 0
    failed_count = 0
    
    # 使用线程池并行执行，每个工作线程有独立的 GPT54 实例
    with ThreadPoolExecutor(
        max_workers=max_workers,
        initializer=worker_initializer,  # 每个线程启动时调用
    ) as executor:
        # 提交所有任务（不再传递 llm 实例）
        future_to_task = {
            executor.submit(
                generate_single_tool,
                task["task_item"],
                task["task_type"],
                task["main_topic"],
                task["subtopic"],
                output_dir,
                retry_count,
                rate_limit_delay
            ): task
            for task in all_tasks
        }
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(all_tasks), desc="🚀 生成 tools", unit="task") as pbar:
            for future in as_completed(future_to_task):
                task_info = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        success_count += 1
                        pbar.set_postfix({"✅": success_count, "❌": failed_count})
                    else:
                        failed_count += 1
                        print(f"\n  {task_info['task_item'].get('id')}: {result.error}")
                    checkpoint.record_result(
                        str(task_info.get("task_key", result.task_key)),
                        {
                            "task_key": str(task_info.get("task_key", result.task_key)),
                            "task_id": result.task_id,
                            "filepath": result.filepath,
                            "success": result.success,
                            "error": result.error,
                            "duration": round(result.duration, 2),
                        },
                    )
                    
                    pbar.update(1)
                    
                    # 速率限制：控制 API 调用频率
                    # if rate_limit_delay > 0:
                    #     time.sleep(rate_limit_delay)
                        
                except Exception as e:
                    failed_count += 1
                    print(f"\n  异常：{e}")
                    pbar.update(1)
    
    # 生成统计报告
    summary = {
        "total": len(all_tasks) + stats["completed"],
        "success": success_count + stats["completed"],
        "failed": failed_count,
        "success_rate": f"{(success_count + stats['completed'])/max(1, len(all_tasks) + stats['completed'])*100:.1f}%",
        "results": [
            {
                "task_key": r.task_key,
                "task_id": r.task_id,
                "filepath": r.filepath,
                "success": r.success,
                "error": r.error,
                "duration": round(r.duration, 2),
            }
            for r in results
        ],
        "checkpoint": checkpoint.get_stats(),
    }
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"生成完成!")
    print(f"{'='*60}")
    print(f"总任务数：   {summary['total']}")
    print(f"成功：       {summary['success']} ✅")
    print(f"失败：       {summary['failed']}")
    print(f"成功率：     {summary['success_rate']}")
    print(f"{'='*60}")
    
    # 保存详细报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "generation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"报告已保存：{report_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行生成 tools")
    parser.add_argument("--input-dir", type=str, default="tasks")
    parser.add_argument("--output-dir", type=str, default="tools")
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        default=[],
        help="Task type to include (repeatable). Defaults to sequential, parallel, mixture.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of collected tasks")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--rate-limit-delay", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint state before running")
    args = parser.parse_args()

    gentool_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task_types=args.task_types or ["sequential", "parallel", "mixture"],
        max_tasks=args.max_tasks,
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        rate_limit_delay=args.rate_limit_delay,
        checkpoint_path=args.checkpoint_path,
        reset_checkpoint=args.reset,
    )
