# -*- coding: utf-8 -*-
"""Global config - all paths anchored to H drive, no C drive usage.
支持从 .env 文件加载环境变量，便于部署时配置。"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "__cache"
MODELS = CACHE / "models"
ARTIFACTS = ROOT / "logs"
ARTIFACTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

os.environ.setdefault("PIP_CACHE_DIR", str(CACHE / "pip"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
os.environ.setdefault("ONNXRUNTIME_CACHE", str(CACHE / "onnx"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "15666"))
OCR_RETRY = int(os.getenv("OCR_RETRY", "3"))
SLIDE_SCALES = [1.0, 0.9, 0.8]
ADMIN_KEY = os.getenv("ADMIN_KEY", "changeme")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
CACHE_LIMIT = int(os.getenv("CACHE_LIMIT", "1000"))
INFER_TIMEOUT = int(os.getenv("INFER_TIMEOUT", "30000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

# 全局偏移补偿（单位：像素）。键为 type_code 或 "default"。
# 例：{"default": [0, 0], "1005": [2, -1]}
COORD_OFFSETS = {
    "default": [0, 0],
    # "1005": [2, -1],
    # "1006": [1, 0],
}
