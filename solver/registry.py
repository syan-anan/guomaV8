# -*- coding: utf-8 -*-
"""题型注册表 v2 — 将原有 19 个题型映射到求解函数，并启用闲置引擎 + v2 轨迹/滑块/逻辑。"""
from solver.ocr import solve_ocr
from solver.slide import detect_gap_multiscale as _legacy_gap_ms
try:
    from solver.slide_v2 import detect_gap_multiscale_v2 as _v2_gap_ms
except Exception:
    _v2_gap_ms = None

try:
    from solver.trajectory_v2 import generate_track as _trk_v2
except Exception:
    from solver.trajectory import generate_track as _trk_v2

from solver.click import solve_click_word, solve_click_icon, solve_click_pass
from solver.logic import solve_word_order, solve_spatial_reasoning, solve_nine_grid
try:
    from solver.logic_v2 import solve_word_order_v2, solve_spatial_reasoning_v2
    # fallback chain: try v2 first
    def _wo(img_src, target_phrase=None):
        try:
            return solve_word_order_v2(img_src, target_phrase)
        except Exception:
            return solve_word_order(img_src, target_phrase)
    def _sr(img_src, question=None):
        try:
            return solve_spatial_reasoning_v2(img_src, question)
        except Exception:
            return solve_spatial_reasoning(img_src, question)
except Exception:
    def _wo(img_src, target_phrase=None):
        return solve_word_order(img_src, target_phrase)
    def _sr(img_src, question=None):
        return solve_spatial_reasoning(img_src, question)

from config import COORD_OFFSETS


def detect_gap_multiscale(bg, gap):
    """优先使用 v2 融合，失败回退旧版"""
    if _v2_gap_ms is not None:
        r = _v2_gap_ms(bg, gap)
        if r and r.get("confidence", 0) > 0:
            return r
        legacy = _legacy_gap_ms(bg, gap)
        return legacy or {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}
    r = _legacy_gap_ms(bg, gap)
    return r or {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}


def _apply_offset(type_code, result):
    """对坐标型结果应用全局偏移补偿。"""
    key = str(type_code)
    dx, dy = COORD_OFFSETS.get(key, COORD_OFFSETS.get("default", [0, 0]))
    if dx == 0 and dy == 0:
        return result
    if "distance" in result:
        result["distance"] = round(result["distance"] + dx)
    if "points" in result and isinstance(result["points"], list):
        for p in result["points"]:
            if isinstance(p, dict):
                if "x" in p: p["x"] = round(p["x"] + dx)
                if "y" in p: p["y"] = round(p["y"] + dy)
    for k in ("x", "y"):
        if k in result and isinstance(result[k], (int, float)):
            offset = dx if k == "x" else dy
            result[k] = round(result[k] + offset)
    return result


def _json_safe(o):
    """递归清理 numpy 类型"""
    import numpy as _np
    if isinstance(o, dict): return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.ndarray,)): return o.tolist()
    return o


def _handle_slide(bg, gap=None):
    gap_result = detect_gap_multiscale(bg, gap)
    distance = gap_result["distance"]
    track = _trk_v2(distance) if distance > 0 else []
    y = gap_result.get("y", 0)
    return {"distance": distance, "y": y, "track": track,
            "method": gap_result["method"], "confidence": gap_result["confidence"]}


# 闲置引擎懒加载（避免启动时全部加载）
_engine_cache = {}
def _engine(name):
    if name not in _engine_cache:
        mod = __import__(f"solver.engines.{name}", fromlist=["*"])
        _engine_cache[name] = mod
    return _engine_cache[name]


def _rotation(img_src):
    mod = _engine("_1009_rotation")
    if hasattr(mod, "RotationSolver"):
        return mod.RotationSolver().solve(img_src)
    if hasattr(mod, "solve_rotation_v8"):
        return mod.solve_rotation_v8(img_src)
    return {"code": -3, "error": "no rotation fn found"}


def _long_slider(img_src, gap=None):
    mod = _engine("_1011_long_slider")
    cls = getattr(mod, "LongSliderSolver", None) or getattr(mod, "solve_long_slider", None)
    if cls is None: raise RuntimeError("long slider unavailable")
    inst = cls() if callable(cls) and hasattr(cls, "solve") else None
    if inst is not None: return inst.solve(img_src, gap)
    return cls(img_src, gap)


def _match3(img_src):
    mod = _engine("_1014_match3_solver")
    cls = getattr(mod, "Match3Solver", None)
    if cls is None: raise RuntimeError("match3 unavailable")
    return cls().solve(img_src)


def _drawing(img_src):
    mod = _engine("_1015_drawing_canvas")
    cls = getattr(mod, "DrawingCanvasSolver", None) or getattr(mod, "DrawingSolver", None)
    if cls is None: raise RuntimeError("drawing unavailable")
    return cls().solve(img_src)


def _calendar(img_src):
    mod = _engine("_1016_calendar_picker")
    cls = getattr(mod, "CalendarPickerSolver", None) or getattr(mod, "CalendarSolver", None)
    if cls is None: raise RuntimeError("calendar unavailable")
    return cls().solve(img_src)


def _combo(img_src):
    mod = _engine("_1017_image_text_combo")
    cls = getattr(mod, "ImageTextComboSolver", None) or getattr(mod, "ComboSolver", None)
    if cls is None: raise RuntimeError("combo unavailable")
    return cls().solve(img_src)


REGISTRY = {
    # 文字识别
    1001: {"name": "英数混合", "fn": lambda img, **kw: solve_ocr(img, 1001)},
    1002: {"name": "纯数字", "fn": lambda img, **kw: solve_ocr(img, 1002)},
    1003: {"name": "纯字母", "fn": lambda img, **kw: solve_ocr(img, 1003)},
    # 极验综合
    1010: {"name": "极验-二代三代通用", "fn": lambda bg, gap=None, **kw: _handle_slide(bg, gap)},
    # 极验三代
    1004: {"name": "极验-三代滑动", "fn": lambda bg, gap=None, **kw: _handle_slide(bg, gap)},
    1005: {"name": "极验-三代点选(选字)", "fn": lambda img, words=None, **kw: solve_click_word(img, words or [])},
    1006: {"name": "极验-三代点选(选物)", "fn": lambda img, icons=None, **kw: solve_click_icon(img, icons or [])},
    1007: {"name": "极验-三代点选(语序选择)", "fn": lambda img, phrase=None, **kw: _wo(img, phrase)},
    1008: {"name": "极验-三代空间推理", "fn": lambda img, question=None, **kw: _sr(img, question)},
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
    # 新增：之前闲置的引擎
    1009: {"name": "图形-旋转滑块", "fn": lambda img, **kw: _rotation(img)},
    1014: {"name": "趣味-消消乐", "fn": lambda img, **kw: _match3(img)},
    1024: {"name": "趣味-描边画线", "fn": lambda img, **kw: _drawing(img)},
    1025: {"name": "趣味-日历选择", "fn": lambda img, **kw: _calendar(img)},
    1026: {"name": "趣味-图文组合点选", "fn": lambda img, **kw: _combo(img)},
    1011: {"name": "长滑块专用", "fn": lambda bg, gap=None, **kw: _long_slider(bg, gap)},
}

UNKNOWN_TYPE_FALLBACK = 1002


_UNKNOWN_ALNUM = 1001
_UNKNOWN_DIGIT = UNKNOWN_TYPE_FALLBACK


def _solve_unknown(type_code, params):
    """未注册题型：走 1002（该分支在提升后的路由里已经保留字母）。

    1002 返回空时才补一次 1001，避免老配置下把混合字符整段吞掉。
    """
    digit = REGISTRY.get(_UNKNOWN_DIGIT)
    alnum = REGISTRY.get(_UNKNOWN_ALNUM)
    if digit is None:
        return {"code": -1, "error": f"不支持的题型: {type_code}"}
    result = digit["fn"](**params)
    text = str((result or {}).get("text") or (result or {}).get("raw") or "")
    if not text and alnum is not None:
        alt = alnum["fn"](**params)
        alt_text = str((alt or {}).get("text") or (alt or {}).get("raw") or "")
        if alt_text:
            return _json_safe({"code": 0, "type": type_code, "name": alnum["name"],
                               "data": _apply_offset(type_code, alt),
                               "auto_routed": True, "inferred_type": _UNKNOWN_ALNUM})
    return _json_safe({"code": 0, "type": type_code, "name": digit["name"],
                       "data": _apply_offset(type_code, result),
                       "auto_routed": True, "inferred_type": _UNKNOWN_DIGIT})


def solve(type_code: int, **params) -> dict:
    """统一入口。

    未知题型（未注册的 type_code）不再直接报错：只要带了图像，就自动落到通用
    OCR 分支（1002），并在结果里标 auto_routed / inferred_type。没有图像才报错。
    """
    entry = REGISTRY.get(type_code)
    auto = False
    if not entry:
        has_img = any(params.get(k) for k in ("img", "image", "path", "file", "image_path"))
        if not has_img:
            return {"code": -1, "error": f"不支持的题型: {type_code}"}
        return _solve_unknown(type_code, params)
    try:
        result = entry["fn"](**params)
        result = _apply_offset(type_code, result)
        out = {"code": 0, "type": type_code, "name": entry["name"], "data": result}
        if auto:
            out["auto_routed"] = True
            out["inferred_type"] = UNKNOWN_TYPE_FALLBACK
        return _json_safe(out)
    except Exception as e:
        return _json_safe({"code": -2, "error": str(e), "type": type_code,
                           "auto_routed": auto})
    try:
        result = entry["fn"](**params)
        result = _apply_offset(type_code, result)
        return _json_safe({"code": 0, "type": type_code, "name": entry["name"], "data": result})
    except Exception as e:
        return _json_safe({"code": -2, "error": str(e), "type": type_code})
