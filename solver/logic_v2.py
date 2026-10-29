# -*- coding: utf-8 -*-
"""逻辑推理引擎 v2 — 用 ddddocr 字符级识别替换像素坐标排序，提升 1007/1008 语义能力"""
import re
import cv2
import numpy as np
from solver.click import detect_text_regions
from solver.utils import load_to_cv
from solver.ocr import _get_model, _clean_seq

def _ocr_boxes(img_bgr):
    """调用 ddddocr beta 拿带框识别，返回 [{text, conf, cx, cy}, ...]"""
    out = []
    try:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return out
        import ddddocr
        res = _get_model("beta").crop_txt(buf.tobytes())
        if isinstance(res, list):
            for it in res:
                if not isinstance(it, dict):
                    continue
                box = it.get("box") or it.get("position") or []
                txt = (it.get("text") or "").strip()
                if not box or len(box) < 4:
                    continue
                xs = [b[0] for b in box]
                ys = [b[1] for b in box]
                out.append({"text": txt, "conf": float(it.get("confidence", 0.0) or 0.0),
                            "cx": int((min(xs) + max(xs)) / 2), "cy": int((min(ys) + max(ys)) / 2)})
    except Exception:
        pass
    return out

def solve_word_order_v2(img_src, target_phrase=None):
    """1007 语序选择 v2: OCR 语义识别 + 按 phrase 顺序匹配坐标"""
    img = load_to_cv(img_src)
    if img is None:
        return {"points": [], "error": "image load failed"}
    
    regions = detect_text_regions(img_src)
    ocrs = _ocr_boxes(img)
    # 合并: 优先 OCR 带文本的，再补区域检测结果
    text_map = {}
    for o in ocrs:
        t = _clean_seq(o["text"])
        if t:
            for ch in t:
                if ch not in text_map:
                    text_map[ch] = (o["cx"], o["cy"], o["conf"])
    for r in regions:
        rt = (r.get("text") or "").upper().strip()
        if not rt:
            continue
        for ch in rt:
            if ch not in text_map:
                text_map[ch] = (r["cx"], r["cy"], 0.0)
    
    points = []
    if target_phrase:
        phrase = [str(ch).upper().strip() for ch in target_phrase if str(ch).strip()]
        for ch in phrase:
            if ch in text_map:
                cx, cy, conf = text_map[ch]
                points.append({"x": cx, "y": cy, "text": ch, "confidence": conf})
        return {"points": points, "count": len(points), "engine": "logic_v2"}
    # 无 phrase 时: 按 OCR 置信度 + 位置输出
    items = [(cx, cy, conf) for _, (cx, cy, conf) in text_map.items()]
    items.sort(key=lambda p: (p[1] // 30, p[0]))
    for cx, cy, conf in items:
        points.append({"x": cx, "y": cy, "text": "", "confidence": conf})
    return {"points": points, "count": len(points), "engine": "logic_v2"}

def solve_spatial_reasoning_v2(img_src, question=None):
    """1008 空间推理 v2: OCR + 空间排序，按 question 文本过滤目标"""
    img = load_to_cv(img_src)
    if img is None:
        return {"points": [], "error": "image load failed"}
    ocrs = _ocr_boxes(img)
    regions = detect_text_regions(img_src)
    merged = []
    for o in ocrs:
        t = _clean_seq(o["text"])
        if t:
            merged.append({"text": t, "cx": o["cx"], "cy": o["cy"], "conf": o["conf"]})
    seen = set()
    for r in regions:
        rt = (r.get("text") or "").upper().strip()
        if rt and (r["cx"], r["cy"]) not in seen:
            merged.append({"text": rt, "cx": r["cx"], "cy": r["cy"], "conf": 0.0})
    # 按 question 过滤
    if question:
        q = _clean_seq(str(question))
        if q:
            hit = [m for m in merged if q in m["text"] or m["text"] in q]
            if hit:
                merged = hit
    merged.sort(key=lambda m: (m["cy"], m["cx"]))
    points = [{"x": m["cx"], "y": m["cy"], "text": m["text"], "confidence": m["conf"]} for m in merged]
    return {"points": points, "count": len(points), "engine": "logic_v2"}
