# -*- coding: utf-8 -*-
"""启动入口 — 直接运行即可启动验证码识别服务。"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 确保所有缓存走 H 盘
os.environ.setdefault("PIP_CACHE_DIR", os.path.join(os.getcwd(), "__cache", "pip"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), "__cache"))
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "__cache", "hf"))
# 从 api.server 导入并启动
from api.server import app, HOST, PORT, APP_VERSION
import uvicorn
if __name__ == "__main__":
    print(f"syandaV8 {APP_VERSION} — 启动于 http://{HOST}:{PORT}")
    print(f"支持 {len(__import__('solver.registry').registry.REGISTRY)} 种题型")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
