import os
os.chdir("H:\\qinglong\\syandaV8")

def w(fname, content):
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    print("  written:", fname)

# trajectory.py
w("solver/trajectory.py", """# -*- coding: utf-8 -*-
"""Trajectory engine - bezier curve + acceleration + jitter."""
import math, random

def bezier_point(t, points):
    n = len(points) - 1
    x, y = 0.0, 0.0
    for i, p in enumerate(points):
        coeff = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        x += p[0] * coeff
        y += p[1] * coeff
    return (round(x, 1), round(y, 1))

def generate_track(distance, y_offset_range=(-5, 5), steps=None):
    if steps is None:
        steps = max(20, min(80, distance // 2 + random.randint(-5, 5)))
    y_offset = random.randint(*y_offset_range)
    mid = distance * (0.6 + random.random() * 0.15)
    ctrl_points = [
        (0, 0),
        (mid * 0.3, y_offset + random.randint(-3, 0)),
        (mid * 0.1, y_offset + random.randint(0, 3)),
        (mid, y_offset + random.randint(-2, 2)),
        (distance * 0.85, y_offset + random.randint(-1, 1)),
        (distance, y_offset * 0.5),
    ]
    track = []
    for i in range(steps):
        t = i / (steps - 1)
        eased_t = t ** (0.4 + random.random() * 0.1) if t < 0.3 else t
        if t > 0.85:
            eased_t = 0.85 + (t - 0.85) * 0.3
        pt = bezier_point(min(eased_t, 1.0), ctrl_points)
        jitter = random.randint(-1, 1) if random.random() < 0.15 else 0
        track.append({"x": pt[0] + jitter, "y": pt[1] + random.randint(-1, 1), "t": round(t * 1000)})
    track[-1] = {"x": distance, "y": y_offset, "t": 1000}
    return track
""")

# slide.py
w("solver/slide.py", """# -*- coding: utf-8 -*-
"""Slide gap detection - template matching + edge analysis + multi-scale."""
import numpy as np, cv2
from solver.utils import load_to_cv
from config import SLIDE_SCALES

def detect_gap(bg_img_src, gap_img_src=None, scale=1.0):
    bg = load_to_cv(bg_img_src)
    if bg is None:
        raise ValueError("bg image load failed")
    h, w = bg.shape[:2]
    if scale != 1.0:
        bg = cv2.resize(bg, None, fx=scale, fy=scale)
    results = []
    if gap_img_src:
        gap = load_to_cv(gap_img_src)
        if gap is not None and gap.shape[0] > 20 and gap.shape[1] > 20:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
            if gap.shape[2] == 4:
                alpha = gap[:, :, 3]
                _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
                res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
            else:
                res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.4:
                x = max_loc[0] + gap_gray.shape[1] // 2
                results.append({"distance": round(x / scale), "method": "tmpl", "confidence": round(max_val, 4)})
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(bg_gray, 50, 150)
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 10 or ch < 10 or cw > w * 0.6:
            continue
        if 0.5 < ch / max(cw, 1) < 2.5:
            candidates.append(x)
    if candidates:
        x = sorted(candidates)[0]
        results.append({"distance": round(x / scale), "method": "edge", "confidence": 0.7})
    sobelx = cv2.Sobel(bg_gray, cv2.CV_64F, 1, 0, ksize=3)
    _, thresh = cv2.threshold(np.abs(sobelx), 80, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    cols = np.sum(thresh, axis=0)
    if np.max(cols) > 0:
        mid = w // 2
        peak_idx = np.argmax(cols[:mid]) if np.max(cols[:mid]) > 0 else mid // 2
        results.append({"distance": round(peak_idx / scale), "method": "sobel", "confidence": 0.6})
    if not results:
        return {"distance": 0, "method": "none", "confidence": 0.0}
    return max(results, key=lambda r: r["confidence"])

def detect_gap_multiscale(bg_img_src, gap_img_src=None):
    best = None
    for scale in SLIDE_SCALES:
        r = detect_gap(bg_img_src, gap_img_src, scale=scale)
        if r["confidence"] > 0 and (best is None or r["confidence"] > best["confidence"]):
            best = r
    return best or {"distance": 0, "method": "none", "confidence": 0.0}
""")

# click.py
w("solver/click.py", """# -*- coding: utf-8 -*-
"""Click engine - word/icon/pass selection for click captchas."""
import numpy as np, cv2
import ddddocr
from solver.utils import load_to_cv, cv_to_pil

_ocr_det = None
def _det():
    global _ocr_det
    if _ocr_det is None:
        _ocr_det = syandaV8.DdddOcr(det=True, show_ad=False)
    return _ocr_det

_ocr_cls = None
def _cls():
    global _ocr_cls
    if _ocr_cls is None:
        _ocr_cls = syandaV8.DdddOcr(show_ad=False)
    return _ocr_cls

def detect_text_regions(img_src):
    img = load_to_cv(img_src)
    pil = cv_to_pil(img)
    det = _det()
    ocr = _cls()
    results = det.detection(pil)
    regions = []
    for box in results:
        if len(box) >= 5:
            x1, y1, x2, y2, conf = map(int, box[:5])
        else:
            x1, y1, x2, y2 = map(int, box)
            conf = 100
        if conf < 50:
            continue
        crop = img[y1:y2, x1:x2]
        _, buf = cv2.imencode(".png", crop)
        text = ocr.classification(buf.tobytes())
        regions.append({"x": x1, "y": y1, "w": x2-x1, "h": y2-y1,
                        "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                        "text": text.strip(), "conf": conf/100.0})
    return regions

def solve_click_word(img_src, target_words):
    regions = detect_text_regions(img_src)
    points = []
    for word in target_words:
        best, best_score = None, 0
        for r in regions:
            if word in r["text"] or any(c in r["text"] for c in word):
                score = len(set(word) & set(r["text"]))
                if score > best_score:
                    best, best_score = r, score
        if best:
            points.append({"x": best["cx"], "y": best["cy"], "text": best["text"]})
    return {"points": points, "count": len(points), "total": len(target_words)}

def solve_click_icon(img_src, icon_templates):
    bg = load_to_cv(img_src)
    points = []
    for i, tmpl_b64 in enumerate(icon_templates):
        tmpl = load_to_cv(tmpl_b64)
        if tmpl is None or tmpl.shape[0] < 5:
            continue
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(bg_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.5:
            cx = max_loc[0] + tmpl_gray.shape[1] // 2
            cy = max_loc[1] + tmpl_gray.shape[0] // 2
            points.append({"x": cx, "y": cy, "conf": round(max_val, 4), "index": i})
    return {"points": points, "count": len(points)}

def solve_click_pass(img_src, count=3):
    regions = detect_text_regions(img_src)
    regions.sort(key=lambda r: r["conf"], reverse=True)
    selected = regions[:count]
    points = [{"x": r["cx"], "y": r["cy"], "text": r["text"], "conf": r["conf"]} for r in selected]
    return {"points": points, "count": len(points)}
""")

print("ALL DONE")
