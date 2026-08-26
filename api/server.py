# -*- coding: utf-8 -*-
"""FastAPI 服务层 - 验证码识别 API (打码狗风格). v2.2: +/metrics +/batch_solve"""
import time, threading, hashlib, json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from collections import defaultdict, deque
from typing import Optional, List
from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
import uvicorn
from solver.registry import REGISTRY, solve
from config import HOST, PORT, RATE_LIMIT_PER_MINUTE, INFER_TIMEOUT
from functools import lru_cache
from solver.ocr import preload_ocr_models
from solver.click import preload_click_models
from api.response import ApiResponse, success, error, ErrorCode

app = FastAPI(title="syandaV8", version="2.2.0", docs_url="/docs")


# ---------- 单接口限流 ----------
_rate_lock = threading.Lock()
_rate_buckets = defaultdict(lambda: deque())


def _rate_limit(client_ip: str):
    """基于 client_ip 的滑动窗口限流，窗口 60 秒。"""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True
    now = time.time()
    window = 60.0
    with _rate_lock:
        q = _rate_buckets[client_ip]
        while q and q[0] < now - window:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MINUTE:
            return False
        q.append(now)
        return True


# ---------- 推理超时 ----------
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="solver_")


def _solve_with_timeout(type_code, **params):
    """在独立线程中执行推理，支持超时。"""
    future = _executor.submit(solve, type_code, **params)
    try:
        return future.result(timeout=INFER_TIMEOUT / 1000.0)
    except FutureTimeout:
        future.cancel()
        return error(ErrorCode.TIMEOUT, "推理超时", type_code=type_code)


@app.on_event("startup")
async def _startup():
    """启动时预热所有模型，避免首次请求冷启动。"""
    try:
        preload_ocr_models()
        preload_click_models()
    except Exception as e:
        print(f"模型预热失败（不影响运行）: {e}")


# ---------- 结果缓存 ----------
_CACHE_LIMIT = 1000
_cache = {}
_cache_lock = threading.Lock()
_cache_hit = 0
_cache_miss = 0


def _cache_key(req: SolveRequest) -> str:
    """基于题型和图片内容生成缓存键。"""
    extra_json = json.dumps(req.extra, sort_keys=True, default=str) if req.extra else ""
    raw = f"{req.type}:{req.image}:{req.gap_image or ''}:{extra_json}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cached_solve(req: SolveRequest):
    """带缓存的求解，命中直接返回缓存结果。"""
    global _cache_hit, _cache_miss
    key = _cache_key(req)
    with _cache_lock:
        if key in _cache:
            _cache_hit += 1
            return _cache[key]
    res = _do_solve(req)
    with _cache_lock:
        if len(_cache) >= _CACHE_LIMIT:
            _cache.pop(next(iter(_cache)))
        _cache[key] = res
        _cache_miss += 1
    return res

# ---------- 监控统计 ----------
_start = time.time()
_lock = threading.Lock()
_stats = {"total": 0, "err": 0, "by_type": {}}  # by_type[code] = {count, cost_sum, ok}

def _record(type_code, cost_ms, ok):
    with _lock:
        _stats["total"] += 1
        if not ok:
            _stats["err"] += 1
        b = _stats["by_type"].setdefault(type_code, {"count": 0, "cost_sum": 0.0, "ok": 0})
        b["count"] += 1
        b["cost_sum"] += cost_ms
        if ok:
            b["ok"] += 1

class SolveRequest(BaseModel):
    type: int
    image: str
    gap_image: Optional[str] = None
    extra: Optional[dict] = None

class SolveResponse(BaseModel):
    code: int
    message: str = ""
    type: int = 0
    name: str = ""
    data: dict = {}
    conf: float = 0.0
    cost_ms: float = 0.0

def _build_params(type_code, req):
    params = {}
    if type_code in (1001, 1002, 1003):
        params["img"] = req.image
    elif type_code in (1004, 1010, 1012, 1020):
        params["bg"] = req.image
        if req.gap_image:
            params["gap"] = req.gap_image
    elif type_code in (1005, 1015, 1022):
        params["img"] = req.image
        params["words"] = (req.extra or {}).get("words", [])
    elif type_code in (1006, 1017, 1023):
        params["img"] = req.image
        params["icons"] = (req.extra or {}).get("icons", [])
    elif type_code == 1007:
        params["img"] = req.image
        params["phrase"] = (req.extra or {}).get("phrase")
    elif type_code == 1008:
        params["img"] = req.image
        params["question"] = (req.extra or {}).get("question")
    elif type_code in (1018, 1019):
        params["img"] = req.image
        params["pos"] = (req.extra or {}).get("positions")
    elif type_code in (1016, 1021):
        params["img"] = req.image
        params["count"] = (req.extra or {}).get("count", 3)
    return params

def _do_solve(req: SolveRequest):
    t0 = time.time()
    type_code = req.type
    entry = REGISTRY.get(type_code)
    if not entry:
        return SolveResponse(code=-1, message="unsupported " + str(type_code), type=type_code)
    params = _build_params(type_code, req)
    result = _solve_with_timeout(type_code, **params)
    cost = round((time.time() - t0) * 1000, 1)
    data = result.get("data", result)
    conf = 0.0
    if isinstance(data, dict):
        conf = data.get("confidence") or data.get("conf") or 0.0
    else:
        conf = result.get("confidence") or 0.0
    ok = result.get("code", 0) == 0 and bool(data)
    _record(type_code, cost, ok)
    return SolveResponse(code=result.get("code", 0),
        message="ok" if ok else result.get("error", ""),
        type=type_code, name=entry["name"], data=data, conf=round(float(conf or 0.0), 4), cost_ms=cost)

@app.get("/")
async def root():
     return success({"service": "syandaV8", "version": "2.2.0", "types": len(REGISTRY), "status": "running"})

@app.get("/types")
async def list_types():
     return success({"types": [{"code": k, "name": v["name"]} for k, v in sorted(REGISTRY.items())]})

@app.get("/metrics")
async def metrics():
    with _lock:
        uptime = round(time.time() - _start, 1)
        by_type = {}
        for k, v in _stats["by_type"].items():
            by_type[str(k)] = {"count": v["count"], "avg_ms": round(v["cost_sum"]/v["count"], 2) if v["count"] else 0,
                               "ok_count": v["ok"], "ok_rate": round(v["ok"]/v["count"], 4) if v["count"] else 0}
        return success({"uptime_s": uptime, "total_requests": _stats["total"],
                 "error_requests": _stats["err"], "by_type": by_type,
                 "cache": {"size": len(_cache), "hit": _cache_hit, "miss": _cache_miss}})

@app.post("/solve", response_model=SolveResponse)
async def solve_captcha(req: SolveRequest, request: Request):
     client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
     if not _rate_limit(client_ip):
         return error(ErrorCode.RATE_LIMIT, "请求过于频繁，请稍后再试")
     return _cached_solve(req)

@app.post("/solve/retry", response_model=SolveResponse)
async def solve_retry(req: SolveRequest, attempts: int = 3):
    t0 = time.time()
    type_code = req.type
    entry = REGISTRY.get(type_code)
    if not entry:
        return SolveResponse(code=-1, message="unsupported", type=type_code)
    best_res, best_conf = None, -1.0
    for i in range(max(1, min(attempts, 10))):
        params = _build_params(type_code, req)
        r = solve(type_code, **params)
        d = r.get("data", r)
        c = (d.get("confidence") if isinstance(d, dict) else 0.0) or 0.0
        if c > best_conf:
            best_conf, best_res = c, r
    data = best_res.get("data", best_res) if best_res else {}
    conf = (data.get("confidence") if isinstance(data, dict) else 0.0) or 0.0
    ok = bool(data)
    _record(type_code, round((time.time()-t0)*1000, 1), ok)
    cost = round((time.time() - t0) * 1000, 1)
    return SolveResponse(code=0, message="ok", type=type_code, name=entry["name"],
        data=data, conf=round(float(conf), 4), cost_ms=cost)

class BatchRequest(BaseModel):
    items: List[SolveRequest]

@app.post("/batch_solve")
async def batch_solve(req: BatchRequest):
     """批量接口：一次多张图，逐项识别，返回列表。上限 50。"""
     items = req.items[:50]
     results = [_cached_solve(i) for i in items]
     return {"code": 0, "count": len(results), "results": [r.dict() for r in results]}

@app.post("/apiv1/ocr")
async def apiv1_ocr(image: str = Form(...), type: int = Form(1001)):
     result = solve(type, img=image)
     return success(result.get("data", {}), type_code=type)

@app.post("/apiv1/slide")
async def apiv1_slide(bg: str = Form(...), gap: str = Form(None)):
     result = solve(1004, bg=bg, gap=gap)
     return success(result.get("data", {}), type_code=1004)

if __name__ == "__main__":
    print("captcha solver 2.2 on " + HOST + ":" + str(PORT) + "  types=" + str(len(REGISTRY)))
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
