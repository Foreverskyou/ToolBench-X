"""通用工具函数."""
import logging
import sys
from pathlib import Path
import os

# 配置日志
_handlers = [logging.StreamHandler(sys.stdout)]
if os.getenv("LOG_TO_FILE", "0").lower() in {"1", "true", "yes", "on"}:
    _handlers.append(logging.FileHandler(Path("logs/agent.log"), encoding="utf-8", mode="a"))

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger("agent")

# Suppress overly chatty request-level logs from HTTP clients during evaluation runs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def ensure_dirs(*dirs: Path) -> None:
    """确保目录存在."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
