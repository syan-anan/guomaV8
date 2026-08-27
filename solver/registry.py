# -*- coding: utf-8 -*-
"""题型注册表 — 将 19 个题型映射到具体的求解函数。"""
from solver.ocr import solve_ocr
from solver.crnn63_ocr import solve_ocr63
from solver.slide import detect_gap_multiscale
from solver.trajectory import generate_track
from solver.click import solve_click_word, solve_click_icon, solve_click_pass
from solver.logic import solve_word_order, solve_spatial_reasoning, solve_nine_grid
from config import COORD_OFFSETS


def _apply_offset(type_code, result):
    """对坐标型结果应用全局偏移补偿。"""
    key = str(type_code)
    dx, dy = COORD_OFFSETS.get(key, COORD_OFFSETS.get("default", [0, 0]))
    if dx == 0 and dy == 0:
        return result
    # 补偿滑动距离
    if "distance" in result:
        result["distance"] = round(result["distance"] + dx)
    # 补偿点选坐标
    if "points" in result and isinstance(result["points"], list):
        for p in result["points"]:
            if isinstance(p, dict):
                if "x" in p:
                    p["x"] = round(p["x"] + dx)
                if "y" in p:
                    p["y"] = round(p["y"] + dy)
    # 补偿单点坐标
    for k in ("x", "y"):
        if k in result and isinstance(result[k], (int, float)):
            offset = dx if k == "x" else dy
            result[k] = round(result[k] + offset)
    return result


from solver.engines._1008_math import solve_math_v8


def _json_safe(o):
    """递归清理 numpy 类型，保证 API JSON 可序列化"""
    import numpy as _np
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    return o

def _handle_slide(bg, gap=None):
    gap_result = detect_gap_multiscale(bg, gap)
    distance = gap_result["distance"]
    track = generate_track(distance) if distance > 0 else []
    y = gap_result.get("y", 0)
    return {"distance": distance, "y": y, "track": track, "method": gap_result["method"], "confidence": gap_result["confidence"]}

# 题型注册表
REGISTRY = {
    # 文字识别
    1001: {"name": "英数混合", "fn": lambda img, **kw: solve_ocr63(img, 1001)},
    1002: {"name": "纯数字", "fn": lambda img, **kw: solve_ocr63(img, 1002)},
    1003: {"name": "纯字母", "fn": lambda img, **kw: solve_ocr63(img, 1003)},
    # 极验综合
    1010: {"name": "极验-二代三代通用", "fn": lambda bg, gap=None, **kw: _handle_slide(bg, gap)},
    # 极验三代
    1004: {"name": "极验-三代滑动", "fn": lambda bg, gap=None, **kw: _handle_slide(bg, gap)},
    1005: {"name": "极验-三代点选(选字)", "fn": lambda img, words=None, **kw: solve_click_word(img, words or [])},
    1006: {"name": "极验-三代点选(选物)", "fn": lambda img, icons=None, **kw: solve_click_icon(img, icons or [])},
    1007: {"name": "极验-三代点选(语序选择)", "fn": lambda img, phrase=None, **kw: solve_word_order(img, phrase)},
    1008: {"name": "极验-三代空间推理", "fn": lambda img, question=None, **kw: solve_spatial_reasoning(img, question)},
    1019: {"name": "极验-三代九宫格", "fn": lambda img, pos=None, **kw: solve_nine_grid(img, pos)},
    # 极验四代
    1012: {"name": "极验-四代滑动", "fn": lambda bg, gap=None, **kw: _handle_slide(bg, gap)},
    1015: {"name": "极验-四代选汉字", "fn": lambda img, words=None, **kw: solve_click_word(img, words or [])},
    1016: {"name": "极验-四代点过", "fn": lambda img, count=3, **kw: solve_click_pass(img, count)},
    1017: {"name": "极验-四代点图标", "fn": lambda img, icons=None, **kw: solve_click_icon(img, icons or [])},
    1018: {"name": "极验-四代九宫格", "fn": lambda img, pos=None, **kw: solve_nine_grid(img, pos)},
    # 易盾
    1020: {"name": "易盾-滑动拼图", "fn": lambda bg, gap=None, **kw: _handle_slide(bg, gap)},
    1021: {"name": "易盾-无感(点过)", "fn": lambda img, count=3, **kw: solve_click_pass(img, count)},
    1022: {"name": "易盾-点字", "fn": lambda img, words=None, **kw: solve_click_word(img, words or [])},
    1023: {"name": "易盾-点图标", "fn": lambda img, icons=None, **kw: solve_click_icon(img, icons or [])},
}

def solve(type_code: int, **params) -> dict:
     """统一入口：根据题型号路由到对应求解函数。"""
     entry = REGISTRY.get(type_code)
     if not entry:
         return {"code": -1, "error": f"不支持的题型: {type_code}"}
     try:
         result = entry["fn"](**params)
         result = _apply_offset(type_code, result)
         return _json_safe({"code": 0, "type": type_code, "name": entry["name"], "data": result})
     except Exception as e:
         return _json_safe({"code": -2, "error": str(e), "type": type_code})

from solver.engines._1008_math import solve_math_v8
