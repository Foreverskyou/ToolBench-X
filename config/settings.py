"""配置管理，支持环境变量和默认值."""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent / ".env")

# 项目路径
ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "tools"
TASKS_DIR = ROOT_DIR / "tasks"
LOGS_DIR = ROOT_DIR / "logs"

# LLM 配置
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
LLM_TEMPERATURE = float(os.environ["LLM_TEMPERATURE"]) if "LLM_TEMPERATURE" in os.environ else None
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# 确保目录存在
from agent.utils import ensure_dirs
ensure_dirs(LOGS_DIR)
