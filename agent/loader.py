"""任务与工具模块的动态加载器."""
import importlib.util
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import logger


_MODULE_CACHE: Dict[str, Any] = {}
_MODULE_CACHE_LOCK = threading.Lock()
_TOOL_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
_TOOL_INDEX_LOCK = threading.Lock()


def find_task_files(tasks_dir: Path, pattern: str = "**/*.json") -> List[Path]:
    """递归查找所有任务配置文件."""
    return list(tasks_dir.glob(pattern))


def load_task(task_path: Path) -> List[Dict[str, Any]]:
    """加载单个任务配置文件."""
    with task_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    
    if isinstance(payload, dict) and "tasks" in payload:
        tasks_value = payload.get("tasks")
        if isinstance(tasks_value, list):
            tasks = tasks_value
        elif isinstance(tasks_value, dict):
            tasks = [tasks_value]
        else:
            raise ValueError(f"Invalid 'tasks' field in {task_path}: expected list or dict")
    elif isinstance(payload, list):
        tasks = payload
    elif isinstance(payload, dict):
        tasks = [payload]
    else:
        raise ValueError(f"Invalid payload in {task_path}: expected dict or list")

    relative_path = Path(task_path.name)
    if "tasks" in task_path.parts:
        tasks_index = task_path.parts.index("tasks")
        relative_path = Path(*task_path.parts[tasks_index + 1 :])
    else:
        tasks_root = task_path.parents[1] if len(task_path.parents) > 1 else task_path.parent
        try:
            relative_path = task_path.relative_to(tasks_root)
        except ValueError:
            relative_path = Path(task_path.name)
    
    results = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"Invalid task item in {task_path}: expected dict")
        # 附加元数据
        task["_meta"] = {
            "source_file": str(task_path),
            "relative_path": str(relative_path),
        }
        results.append(task)
    
    return results


def load_tool_module(tool_path: Path) -> Any:
    """动态加载工具模块."""
    cache_key = str(tool_path.resolve())
    with _MODULE_CACHE_LOCK:
        cached = _MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    module_name = f"tool_{tool_path.stem}_{abs(hash(cache_key))}"
    spec = importlib.util.spec_from_file_location(module_name, str(tool_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {tool_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE[cache_key] = module
    logger.debug(f"Loaded tool module: {module_name} from {tool_path}")
    return module


def _task_relative_from_tool_parts(parts: Tuple[str, ...]) -> Optional[str]:
    if len(parts) >= 4 and parts[0] in {"sequential", "parallel", "mixture"}:
        # baseline tools: task_type/main_topic/subtopic/task_id.py
        if len(parts) == 4:
            return f"{parts[0]}/{parts[1]}/{Path(parts[2]).with_suffix('.json').name}"
        # categorized exception tools: task_type/category/main_topic/subtopic/task_id.py
        if len(parts) >= 5:
            return f"{parts[0]}/{parts[2]}/{Path(parts[3]).with_suffix('.json').name}"
    return None


def _build_tool_index(tools_dir: Path) -> Dict[str, Any]:
    direct_file_map: Dict[str, Path] = {}
    exact_task_map: Dict[Tuple[str, str], Path] = {}
    fallback_dir_map: Dict[str, Path] = {}

    for path in sorted(tools_dir.rglob("*.py")):
        try:
            rel = path.relative_to(tools_dir)
        except ValueError:
            continue
        rel_key = rel.as_posix()
        direct_file_map.setdefault(rel_key, path)

        task_relative = _task_relative_from_tool_parts(rel.parts)
        if task_relative is None:
            continue
        task_id = path.stem
        exact_task_map.setdefault((task_relative, task_id), path)
        fallback_dir_map.setdefault(task_relative.removesuffix(".json"), path)

    return {
        "direct_file_map": direct_file_map,
        "exact_task_map": exact_task_map,
        "fallback_dir_map": fallback_dir_map,
    }


def _get_tool_index(tools_dir: Path) -> Dict[str, Any]:
    cache_key = str(tools_dir.resolve())
    with _TOOL_INDEX_LOCK:
        cached = _TOOL_INDEX_CACHE.get(cache_key)
        if cached is None:
            cached = _build_tool_index(tools_dir)
            _TOOL_INDEX_CACHE[cache_key] = cached
        return cached


def discover_tool_path(
    tools_dir: Path,
    task_relative_path: str,
    task_id: Optional[str] = None,
    *,
    silent: bool = False,
) -> Optional[Path]:
    """根据任务的相对路径找到对应的工具文件."""
    task_path = Path(task_relative_path)
    is_exception_tools = "tools_exception" in str(tools_dir)
    index = _get_tool_index(tools_dir)
    direct_file_map: Dict[str, Path] = index["direct_file_map"]
    exact_task_map: Dict[Tuple[str, str], Path] = index["exact_task_map"]
    fallback_dir_map: Dict[str, Path] = index["fallback_dir_map"]
    task_relative_norm = task_path.as_posix()
    task_dir_key = task_path.with_suffix("").as_posix()

    if task_id:
        exact_match = exact_task_map.get((task_relative_norm, str(task_id)))
        if exact_match is not None:
            return exact_match
        if is_exception_tools and not silent:
            logger.warning(
                f"Missing exact exception tool for task_id={task_id} for {task_relative_path}; refusing scoped fallback"
            )
            return None
        return None

    direct_match = direct_file_map.get(task_path.with_suffix(".py").as_posix())
    if direct_match is not None:
        return direct_match

    fallback_match = fallback_dir_map.get(task_dir_key)
    if fallback_match is not None:
        if not silent:
            logger.warning(f"Scoped fallback tool match: {fallback_match}")
        return fallback_match
    
    return None
