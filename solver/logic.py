# -*- coding: utf-8 -*-
"""逻辑推理引擎 — 1007 语序选择 / 1008 空间推理 / 1018 九宫格 / 1019 三代九宫格。"""
from solver.click import detect_text_regions, mark_duplicate_regions
from solver.utils import load_to_cv

def solve_word_order(img_src, target_phrase=None):
    """语序选择 1007：按提示 phrase 顺序依次点选对应字符"""
    regions = detect_text_regions(img_src)
    if not regions:
        return {"points": [], "error": "未检测到文字区域"}
    if target_phrase:
        ordered = []
        used = set()
        phrase = [str(ch).upper().strip() for ch in target_phrase if str(ch).strip()]
        for ch in phrase:
            best, best_kind = None, 3
            for ri, r in enumerate(regions):
                if ri in used:
                    continue
                rt = r["text"].upper().strip()
                if not rt:
                    continue
                if rt == ch:
                    if best_kind > 0:
                        best, best_kind = (ri, r), 0
                elif ch in rt and best_kind > 1:
                    best, best_kind = (ri, r), 1
                elif len(rt) == 1 and best_kind > 2:
                    best, best_kind = (ri, r), 2
            if best:
                ri, r = best
                used.add(ri)
                mark_duplicate_regions(ri, regions, used)
                ordered.append({"x": r["cx"], "y": r["cy"], "text": r["text"], "target": ch})
        return {"points": ordered, "count": len(ordered)}
    regions.sort(key=lambda r: (r["y"] // 30, r["x"]))
    points = [{"x": r["cx"], "y": r["cy"], "text": r["text"]} for r in regions]
    return {"points": points, "count": len(points)}

def solve_spatial_reasoning(img_src, question=None):
    """空间推理（1008）：检测文字区域，按空间排序返回。"""
    regions = detect_text_regions(img_src)
    if not regions:
        return {"points": [], "error": "未检测到区域"}
    regions.sort(key=lambda r: (r["y"], r["x"]))
    points = [{"x": r["cx"], "y": r["cy"], "text": r["text"]} for r in regions]
    return {"points": points, "count": len(points)}

def solve_nine_grid(img_src, positions=None):
    """九宫格（1018/1019）：按格子编号点击。
    编号: 1 2 3 / 4 5 6 / 7 8 9"""
    img = load_to_cv(img_src)
    h, w = img.shape[:2]
    gw, gh = w // 3, h // 3
    cells = {}
    for idx in range(9):
        col, row = idx % 3, idx // 3
        cells[idx + 1] = {"x": col * gw + gw // 2, "y": row * gh + gh // 2}
    if positions:
        points = [cells[p] for p in positions if p in cells]
    else:
        points = [cells[i] for i in range(1, 10)]
    return {"points": points, "grid_cells": cells}
