# -*- coding: utf-8 -*-
"""滑块缺口检测 — alpha mask 模板匹配优先。"""
import numpy as np, cv2
from solver.preprocess import load as _load


def _load_bgr(src):
    return _load(src, "bgr")


def _load_rgba(src):
    return _load(src, "unchanged")


def _find_contours(bg):
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(bg_gray, 50, 150)
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)
    c, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return c


def detect_gap(bg_img_src, gap_img_src=None, scale=1.0):
    bg = _load_bgr(bg_img_src)
    h, w = bg.shape[:2]
    if scale != 1.0:
        bg = cv2.resize(bg, None, fx=scale, fy=scale)
    contours = _find_contours(bg)
    results = []

    # 1. Alpha mask template matching
    if gap_img_src:
        gap = _load_rgba(gap_img_src)
        if gap is not None and min(gap.shape[:2]) > 15:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            if gap.shape[2] == 4:
                alpha = gap[:,:,3]
                gap_bgr = cv2.cvtColor(gap[:,:,:3], cv2.COLOR_RGB2BGR)
                gap_gray = cv2.cvtColor(gap_bgr, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
                res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCORR_NORMED, mask=mask)
            else:
                gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.3:
                results.append({"distance": round(max_loc[0]/scale), "y": max_loc[1],
                                "method": "tmpl", "confidence": round(max_val, 4)})

    # 2. Edge detection (fallback)
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 10 or ch < 10 or cw > w * 0.6:
            continue
        if 0.5 < ch / max(cw, 1) < 2.5:
            candidates.append(x)
    if candidates:
        x = sorted(candidates)[0]
        results.append({"distance": round(x/scale), "y": 0, "method": "edge", "confidence": 0.7})

    if not results:
        return {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}

    # 3. 优先返回模板匹配结果，排除边缘假阳性
    tmpl = [r for r in results if r["method"] == "tmpl"]
    if tmpl:
        best = tmpl[0]
        # 排除边缘假阳性（x<5或y<5）
        if best["distance"] < 5 or best["y"] < 5:
            edge = [r for r in results if r["method"] == "edge"]
            if edge:
                return edge[0]
            return {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}
        # 边界校正：y<20时用边缘修正
        if best["y"] < 20:
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < 15 or ch < 15 or cw > w * 0.6:
                    continue
                if abs(y - best["y"]) < 20 and abs(x - best["distance"]) < 30:
                    best = {"distance": round(x/scale), "y": y, "method": "tmpl+edge", "confidence": best["confidence"]}
        return best
    return max(results, key=lambda r: r["confidence"])


def detect_gap_multiscale(bg_img_src, gap_img_src=None):
    best = None
    for scale in [1.0, 0.9]:
        r = detect_gap(bg_img_src, gap_img_src, scale=scale)
        if r["confidence"] > 0 and (best is None or r["confidence"] > best["confidence"]):
            best = r
    return best or {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}
