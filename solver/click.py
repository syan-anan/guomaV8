# -*- coding: utf-8 -*-
"""Click engine - word/icon/pass selection for click captchas. v2: 轮廓兜底检测 + alpha-mask 图标匹配"""
import pathlib
import numpy as np, cv2
import ddddocr
from solver.utils import load_to_cv, cv_to_pil
from solver.preprocess import load as _load

_ocr_det = None
def _det():
    global _ocr_det
    if _ocr_det is None:
        _ocr_det = ddddocr.DdddOcr(det=True, show_ad=False)
    return _ocr_det

_ocr_cls = None
def _cls():
    global _ocr_cls
    if _ocr_cls is None:
        _ocr_cls = ddddocr.DdddOcr(show_ad=False)
    return _ocr_cls

def _load_icon(src):
     """加载图标模板，保留 alpha（IMREAD_UNCHANGED）"""
     return _load(src, "unchanged")


def preload_click_models():
     """预热点选模型，避免首次请求冷启动。"""
     _det()
     _cls()

def _region_key(x1, y1, x2, y2):
    return (x1 // 8, y1 // 8, x2 // 8, y2 // 8)


def mark_duplicate_regions(selected_index, regions, used,
                           max_center_dist=10.0, min_iou=0.55):
    """Treat near-duplicate detector boxes as one selected glyph."""
    sel = regions[selected_index]
    sx1, sy1 = sel["x"], sel["y"]
    sx2, sy2 = sx1 + sel["w"], sy1 + sel["h"]

    for ri, region in enumerate(regions):
        if ri == selected_index or ri in used:
            continue
        dx = region["cx"] - sel["cx"]
        dy = region["cy"] - sel["cy"]
        if dx * dx + dy * dy <= max_center_dist * max_center_dist:
            used.add(ri)
            continue

        x1, y1 = region["x"], region["y"]
        x2, y2 = x1 + region["w"], y1 + region["h"]
        ix1, iy1 = max(sx1, x1), max(sy1, y1)
        ix2, iy2 = min(sx2, x2), min(sy2, y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = sel["w"] * sel["h"] + region["w"] * region["h"] - inter
        if union > 0 and inter / union >= min_iou:
            used.add(ri)

def _color_regions(img, min_chan=190, min_area=60, close_k=5):
    """文字/图标颜色较深，背景亮噪声：min通道 < min_chan 即为前景区域"""
    mn = np.min(img.astype(np.int16), axis=2)
    fg = (mn < min_chan).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    h, w = img.shape[:2]
    boxes = []
    for k in range(1, n):
        x, y, cw, ch, area = stats[k]
        if area < min_area:
            continue
        if cw < 10 or ch < 12:
            continue
        if cw > w * 0.6 or ch > h * 0.6:
            continue
        boxes.append((x, y, x + cw, y + ch))
    return boxes

def _contour_regions(img, min_area=120):
    """Canny + 轮廓 -> 候选文字框（兜底检测）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    boxes = []
    for c in cnts:
        x, y, cw, ch = map(int, cv2.boundingRect(c))
        if cw < 12 or ch < 12 or cw > w * 0.6 or ch > h * 0.6:
            continue
        if cw * ch < min_area:
            continue
        boxes.append((x, y, x + cw, y + ch))
    # 轻量合并：仅当两个框高度重叠且水平距离 < 6px 时合并（保留单字符框）
    merged = []
    for b in sorted(boxes, key=lambda b: b[0]):
        if merged:
            lm = merged[-1]
            vertical_overlap = min(b[3], lm[3]) - max(b[1], lm[1])
            if vertical_overlap > 0 and b[0] - lm[2] < 6 and b[0] - lm[2] > -2:
                merged[-1] = (min(lm[0], b[0]), min(lm[1], b[1]), max(lm[2], b[2]), max(lm[3], b[3]))
                continue
        merged.append(b)
    return merged

def detect_text_regions(img_src):
    img = load_to_cv(img_src)
    pil = cv_to_pil(img)
    regions = []
    seen = set()
    def add_region(x1, y1, x2, y2, conf):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        k = _region_key(x1, y1, x2, y2)
        if k in seen:
            return
        seen.add(k)
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, buf = cv2.imencode(".png", crop)
        try:
            text = _cls().classification(buf.tobytes()).strip()
        except Exception:
            text = ""
        regions.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                        "cx": int((x1 + x2) // 2), "cy": int((y1 + y2) // 2),
                        "text": text, "conf": float(conf)})
    # 1) syandaV8 det
    try:
        results = _det().detection(pil)
        for box in results:
            if len(box) >= 5:
                x1, y1, x2, y2, conf = map(int, box[:5])
            else:
                x1, y1, x2, y2 = map(int, box)
                conf = 100
            if conf < 40:
                continue
            add_region(x1, y1, x2, y2, conf / 100.0)
    except Exception:
        pass
    # 2) 轮廓兜底（补齐 det 漏检）
    for (x1, y1, x2, y2) in _contour_regions(img):
        add_region(x1 - 2, y1 - 2, x2 + 2, y2 + 2, 0.6)
    # 2.5) 颜色分割兜底（补齐小字/低对比文字：背景亮噪声 vs 深色文字）
    for (x1, y1, x2, y2) in _color_regions(img):
        add_region(x1, y1, x2, y2, 0.65)
    # 3) 若仍为空：全图粗检（边缘密度）
    if not regions:
        h, w = img.shape[:2]
        for y in range(0, h, 20):
            for x in range(0, w, 20):
                add_region(x, y, min(w, x + 60), min(h, y + 40), 0.4)
    return regions

def solve_click_word(img_src, target_words):
    regions = detect_text_regions(img_src)
    points = []
    used = set()
    targets = [str(w).upper().strip() for w in target_words]
    for word in targets:
        best, best_kind = None, 3  # 0=精确 1=包含 2=模糊
        for ri, r in enumerate(regions):
            if ri in used:
                continue
            rt = r["text"].upper().strip()
            if not rt:
                continue
            if rt == word:
                if best_kind > 0:
                    best, best_kind = (ri, r), 0
            elif word in rt and best_kind > 1:
                best, best_kind = (ri, r), 1
            elif len(word) == 1 and len(rt) == 1 and best_kind > 2:
                best, best_kind = (ri, r), 2
        if best:
            ri, r = best
            used.add(ri)
            mark_duplicate_regions(ri, regions, used)
            points.append({"x": r["cx"], "y": r["cy"], "text": r["text"]})
    return {"points": points, "count": len(points), "total": len(targets)}
def _local_peaks(res, k=6, min_dist=18):
    """从匹配响应图取前 k 个局部峰（避免同模板多个候选位置）"""
    kernel = np.ones((min_dist * 2 + 1, min_dist * 2 + 1), np.uint8)
    dilated = cv2.dilate(res, kernel)
    peaks = (res == dilated) & (res > 1e-4)
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return [(max_loc[0], max_loc[1], float(max_val))]
    vals = res[ys, xs]
    order = np.argsort(vals)[::-1][:k]
    return [(int(xs[i]), int(ys[i]), float(vals[i])) for i in order]

def _nms_dedup(cands, min_dist):
    """跨模板非极大值抑制：中心距离 < min_dist 只保留 conf 最高的"""
    cands = sorted(cands, key=lambda c: -c[2])
    chosen = []
    for cx, cy, conf, idx in cands:
        if any((cx - ox) ** 2 + (cy - oy) ** 2 < min_dist ** 2 for ox, oy, _, _ in chosen):
            continue
        chosen.append((cx, cy, conf, idx))
    return chosen

def _color_connected(img, tmpl_color, tmpl_size, dist_th=80, min_area=200, max_area=4000):
    """颜色距离图 -> 前景 mask -> 连通域，返回候选中心列表 (cx, cy, area)"""
    dist = np.linalg.norm(img.astype(np.float32) - tmpl_color.astype(np.float32), axis=2)
    fg = (dist < dist_th).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    tw, th = tmpl_size
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        if abs(w - tw) > max(20, tw * 0.7) or abs(h - th) > max(20, th * 0.7):
            continue
        out.append((float(cent[k][0]), float(cent[k][1]), int(area)))
    out.sort(key=lambda c: -c[2])
    return out

def _color_connected(img, tmpl_color, tmpl_size, dist_th=80, min_area=200, max_area=4000):
    """颜色距离图 -> 前景 mask -> 连通域，返回候选中心列表 (cx, cy, area)"""
    dist = np.linalg.norm(img.astype(np.float32) - tmpl_color.astype(np.float32), axis=2)
    fg = (dist < dist_th).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    tw, th = tmpl_size
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        if abs(w - tw) > max(20, tw * 0.7) or abs(h - th) > max(20, th * 0.7):
            continue
        out.append((float(cent[k][0]), float(cent[k][1]), int(area)))
    return out

def solve_click_icon(img_src, icon_templates):
    bg = load_to_cv(img_src)
    # 候选： (template_idx, cx, cy, cdist, score)
    cands = []
    fallback_tmpl = []
    for i, tmpl_b64 in enumerate(icon_templates):
        try:
            tmpl = _load_icon(tmpl_b64)
        except Exception:
            continue
        if tmpl is None or min(tmpl.shape[:2]) < 5:
            continue
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        if tmpl.ndim == 3 and tmpl.shape[2] == 4:
            alpha = tmpl[:, :, 3]
            tmpl_bgr = tmpl[:, :, :3]
            tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
            tmpl_px = tmpl_bgr[alpha > 30]
        else:
            tmpl_bgr = tmpl
            tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
            mask = np.ones(tmpl_gray.shape[:2], np.uint8) * 255
            tmpl_px = tmpl_bgr.reshape(-1, tmpl_bgr.shape[2])
        if len(tmpl_px) == 0:
            continue
        tmpl_color = tmpl_px.mean(axis=0)
        tw, thh = tmpl_gray.shape[1], tmpl_gray.shape[0]
        conn = _color_connected(bg, tmpl_color, (tw, thh))
        if not conn:
            fallback_tmpl.append((i, tmpl, tmpl_gray, mask, tmpl_color, tw, thh))
            continue
        for cx, cy, area in conn:
            # 候选区域平均色 -> 颜色距离（区分同色系）
            roi = bg[int(cy) - thh // 2:int(cy) - thh // 2 + thh,
                     int(cx) - tw // 2:int(cx) - tw // 2 + tw]
            if roi.shape[:2] == (thh, tw) and roi.size:
                cdist = float(np.linalg.norm(roi.mean(axis=(0, 1)) - tmpl_color))
            else:
                cdist = 999.0
            cands.append((i, cx, cy, cdist, 0.99))
    # 兜底：模板匹配 + 颜色重排
    for i, tmpl, tmpl_gray, mask, tmpl_color, tw, thh in fallback_tmpl:
        if tmpl.ndim == 3 and tmpl.shape[2] == 4:
            res = cv2.matchTemplate(bg_gray, tmpl_gray, cv2.TM_CCORR_NORMED, mask=mask)
            th = 0.50
        else:
            res = cv2.matchTemplate(bg_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            th = 0.55
        best = None
        for px, py, val in _local_peaks(res, k=12, min_dist=10):
            if val < th:
                continue
            roi = bg[py:py + thh, px:px + tw]
            m = mask > 0
            if roi.shape[:2] != mask.shape[:2]:
                continue
            roi_px = roi[m]
            if len(roi_px) == 0:
                continue
            cdist = float(np.linalg.norm(roi_px.mean(axis=0) - tmpl_color))
            if best is None or cdist < best[0]:
                best = (cdist, px + tw // 2, py + thh // 2, val)
        if best is not None:
            cands.append((i, best[1], best[2], best[0], best[3]))
    # 贪心全局分配：按颜色距离升序，位置冲突时给更匹配的模板
    cands.sort(key=lambda c: (c[3], -c[4]))
    dedup_dist = 18
    assigned_pos = []
    used_tmpl = set()
    points = []
    for i, cx, cy, cdist, score in cands:
        if i in used_tmpl:
            continue
        if any((cx - ox) ** 2 + (cy - oy) ** 2 < dedup_dist ** 2 for ox, oy in assigned_pos):
            continue
        assigned_pos.append((cx, cy))
        used_tmpl.add(i)
        points.append({"x": int(cx), "y": int(cy), "conf": round(float(score), 4), "index": i})
    return {"points": points, "count": len(points)}
def solve_click_pass(img_src, count=3):
    regions = detect_text_regions(img_src)
    regions.sort(key=lambda r: r["conf"], reverse=True)
    selected = regions[:count]
    points = [{"x": r["cx"], "y": r["cy"], "text": r["text"], "conf": r["conf"]} for r in selected]
    return {"points": points, "count": len(points)}
