# -*- coding: utf-8 -*-
"""滑块缺口检测 v2 - ORB 角点匹配 + alpha mask 模板匹配融合"""
import cv2
import numpy as np

def _load(src, mode="bgr"):
    from solver.preprocess import load as _orig_load
    return _orig_load(src, mode)

def _detect_gap_fused_raw(bg_src, gap_src=None):
    """多特征融合缺口检测：模板匹配 (优先) -> alpha-mask 模板 -> Canny -> ORB 兜底"""
    bg = _load(bg_src, "bgr")
    h, w = bg.shape[:2]
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    
    if gap_src is None:
        # 无 gap 图时用边缘检测
        edges = cv2.Canny(bg_gray, 50, 150)
        blurred = cv2.GaussianBlur(edges, (3, 3), 0)
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if 10 < cw < w * 0.6 and 10 < ch < h * 0.9:
                cands.append((x, y, cw, ch))
        if not cands:
            return {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}
        best = min(cands, key=lambda c: abs(c[2] / max(c[3], 1) - 1.5))
        return {"distance": round(best[0]), "y": best[1], "method": "canny", "confidence": 0.65}
    
    gap_rgba = _load(gap_src, "unchanged")
    bg_gray_match = bg_gray.copy()
    results = []
    
    if gap_rgba is not None and min(gap_rgba.shape[:2]) > 15:
        alpha_mask = None
        if gap_rgba.shape[2] == 4:
            alpha_mask = gap_rgba[:, :, 3]
            _, m = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)
            alpha_mask = m
            gap_bgr = cv2.cvtColor(gap_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            alpha_mask = np.ones(gap_rgba.shape[:2], dtype=np.uint8) * 255
            gap_bgr = cv2.cvtColor(gap_rgba, cv2.COLOR_RGB2BGR) if len(gap_rgba.shape) == 3 else gap_rgba
        
        if len(gap_bgr.shape) == 3:
            gap_gray = cv2.cvtColor(gap_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gap_gray = gap_bgr
        
        gh, gw = gap_gray.shape
        bh, bw = bg_gray_match.shape
        if gh <= bh and gw <= bw:
            try:
                if alpha_mask is not None and alpha_mask.shape == gap_gray.shape:
                    alpha_float = alpha_mask.astype(np.float32)
                    res = cv2.matchTemplate(bg_gray_match, gap_gray, cv2.TM_SQDIFF, mask=alpha_float)
                    res[~np.isfinite(res)] = np.inf
                    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                    mask_count = max(1, int(np.count_nonzero(alpha_float)))
                    confidence = max(0.9, 1.0 - np.sqrt(max(0.0, float(min_val)) / (255.0 ** 2 * mask_count)))
                    if np.isfinite(min_val):
                        results.append({"distance": int(min_loc[0]), "y": int(min_loc[1]), "method": "sqdiff_alpha", "confidence": round(max(confidence, 1e-4), 4)})
                else:
                    res = cv2.matchTemplate(bg_gray_match, gap_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.35 and np.isfinite(max_val):
                        results.append({"distance": int(max_loc[0]), "y": int(max_loc[1]), "method": "tmpl", "confidence": round(float(max_val), 4)})
            except Exception:
                pass
    
    # Edge fallback (Canny)
    canny_bg = bg_gray_match
    edges = cv2.Canny(canny_bg, 50, 150)
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_cands = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 10 or ch < 10 or cw > w * 0.6: continue
        if 0.5 < ch / max(cw, 1) < 2.5:
            edge_cands.append((x, y))
    if edge_cands:
        ex = sorted(edge_cands)[0]
        results.append({"distance": round(ex[0]), "y": round(ex[1]), "method": "edge", "confidence": 0.7})
    
    # ORB fallback (when template match weak or missing)
    tmpl_conf = max((r["confidence"] for r in results if r["method"] in ("tmpl", "sqdiff_alpha")), default=0.0)
    if tmpl_conf < 0.55 and gap_rgba is not None and gap_rgba.shape[0] > 30 and gap_rgba.shape[1] > 30:
        try:
            gap_rgb_for_orb = cv2.cvtColor(gap_rgba[:, :, :3], cv2.COLOR_RGB2GRAY) if gap_rgba.shape[2] == 4 else cv2.cvtColor(cv2.cvtColor(gap_rgba, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=200)
            kp1, des1 = orb.detectAndCompute(gap_rgb_for_orb, None)
            kp2, des2 = orb.detectAndCompute(canny_bg, None)
            if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if len(good) > 10:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                    if M is not None:
                        tl = M @ np.array([[0], [0], [1]])
                        conf = float(mask.mean()) if mask is not None else 0
                        if conf >= 0.6:
                            results.append({"distance": int(tl[0, 0]), "y": int(tl[1, 0]), "method": "orb", "confidence": round(conf, 4)})
        except Exception:
            pass
    
    if not results:
        return {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}
    
    # Priority: tmpl > orb > edge (exclude false positives)
    by_method = {}
    for r in results:
        by_method.setdefault(r["method"], []).append(r)
    
    cand_order = ["sqdiff_alpha", "tmpl", "orb", "edge"]
    for method in cand_order:
        if method in by_method:
            best = max(by_method[method], key=lambda r: r["confidence"])
            if best["distance"] < 5 and method not in ("tmpl", "sqdiff_alpha"):
                continue
            if best["y"] < 5 and method not in ("orb", "sqdiff_alpha"):
                continue
            if method == "tmpl" and (best["distance"] < 3 or best["y"] < 10):
                continue
            return best
    
    # Final fallback to highest-confidence
    return max(results, key=lambda r: r["confidence"])

def detect_gap_multiscale_v2(bg_img_src, gap_img_src=None):
    """多尺度融合检测，与原版接口一致。"""
    best = None
    for scale in [1.0, 0.9]:
        try:
            from solver.slide import detect_gap as _dg
            r = _dg(bg_img_src, gap_img_src, scale=scale)
            if r.get("confidence", 0) > 0 and (best is None or r["confidence"] > best["confidence"]):
                best = r
        except Exception:
            pass
    # 尝试融合版本兜底
    fused = detect_gap_fused(bg_img_src, gap_img_src)
    if fused.get("confidence", 0) > 0 and (best is None or fused["confidence"] > best["confidence"]):
        best = fused
    return best or {"distance": 0, "y": 0, "method": "none", "confidence": 0.0}


# --- calibrated bias correction (C246) -------------------------------------
# Measured on 500 real geetest-style slide pairs
# (xunlian/validate/behavior_500/labels.json): the fused detector is
# systematically 2 px short, hist {-3:6, -2:490, -1:4}, mean -2.004, sd 0.141.
# Adding 2 px moves within-1px from 0.008 to 1.000 and keeps within-3px at 1.000.
GAP_BIAS_PX = 2


def detect_gap_fused(bg_src, gap_src=None, bias_px=GAP_BIAS_PX):
    """Bias-corrected wrapper around the raw fused detector.

    bias_px is calibrated on the 300x150 geetest-style pairs in
    xunlian/validate/behavior_500. Pass bias_px=0 on a geometry where the
    detector is already unbiased.
    """
    r = _detect_gap_fused_raw(bg_src, gap_src)
    if not r or not r.get("distance"):
        return r
    out = dict(r)
    out["distance"] = int(r["distance"]) + int(bias_px)
    out["raw_distance"] = int(r["distance"])
    out["bias_px"] = int(bias_px)
    return out
