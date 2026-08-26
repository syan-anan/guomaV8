# -*- coding: utf-8 -*-
"""
DDDDOCR V8 完美版求解器 (Final V8 Integration)
集成多引擎预处理、尾部增强（解决漏字）、严格题型过滤及投票机制。
"""
import io
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    import warnings
    warnings.warn("OpenCV (cv2) not found.")

import ddddocr


# ==================== 核心配置 ====================
TYPE_RANGES = {
    1001: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    1002: "0123456789",
    1003: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
}

CORRECTION_DIGIT = str.maketrans({'O':'0','Q':'0','G':'0','C':'0','D':'0','Z':'2','S':'5','$':'5'})
CORRECTION_ALPHA = str.maketrans({'1':'l','I':'l','|':'l','0':'O','8':'B'})


class V8Solver:
    def __init__(self):
        self._ocr = syandaV8.DdddOcr(show_ad=False)
        
    def solve(self, img_src: Union[str, bytes], type_code: int = 1001) -> Dict:
        start_time = time.perf_counter()
        self._current_type = type_code
        
        try:
            img = self._load_image(img_src)
        except Exception as e:
            return {"code": -1, "text": "", "confidence": 0.0, "cost_ms": 0, "error": str(e)}

        candidates = []
        # 基础变体
        candidates.append(self._run_one(img, "basic"))
        candidates.append(self._run_one(img, "autocontrast"))
        
        # 高级变体
        if HAS_CV2:
            candidates.append(self._run_one(img, "binary"))
            candidates.append(self._run_one(img, "clahe"))
            
            # 尾部增强
            img_tail = self._augment_tail(img)
            candidates.append(self._run_one(img_tail, "tail_basic"))
            candidates.append(self._run_one(img_tail, "tail_clahe"))

        valid_candidates = [c for c in candidates if c["text"]]
        
        if not valid_candidates:
            return {"code": -2, "text": "", "confidence": 0.0, "cost_ms": round((time.perf_counter()-start_time)*1000)}

        best_text, best_conf = self._vote_system(valid_candidates)
        
        return {
            "code": 0,
            "text": best_text,
            "confidence": round(best_conf, 4),
            "cost_ms": round((time.perf_counter() - start_time) * 1000)
        }

    def _load_image(self, src):
        if isinstance(src, str):
            if len(src) > 200 and "," in src:
                import base64
                bts = base64.b64decode(src.split(",")[-1])
            else:
                with open(src, "rb") as f: bts = f.read()
        else:
            bts = src
        return Image.open(io.BytesIO(bts)).convert("RGB")

    def _run_one(self, img: Image.Image, mode: str) -> Dict:
        processed = self._apply_mode(img, mode)
        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        try:
            res = self._ocr.classification(buf.getvalue())
            clean_text = self._smart_post(res)
            return {"text": clean_text, "raw": res}
        except:
            return {"text": None, "raw": ""}

    def _apply_mode(self, img: Image.Image, mode: str) -> Image.Image:
        target_h = 64
        w, h = img.size
        target_w = max(1, int(w * target_h / h))
        
        if mode == "basic":
            return img.resize((target_w, target_h), Image.LANCZOS).convert("RGB")
        elif mode == "autocontrast":
            p = img.convert("L").resize((target_w, target_h), Image.LANCZOS)
            return ImageOps.autocontrast(p, cutoff=2).convert("RGB")
        elif mode == "binary":
            arr = np.array(img.convert("L"))
            _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return Image.fromarray(binary).resize((target_w, target_h)).convert("RGB")
        elif mode == "clahe":
            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            merged = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
            return Image.fromarray(merged).resize((target_w, target_h)).convert("RGB")
        elif mode == "tail_basic":
            # tail image is already cropped+stitched, just resize the whole thing
            tw, th = img.size
            nw = tw + (tw % 10 != 0) # simple alignment fix if needed
            return img.resize((target_w, target_h), Image.LANCZOS).convert("RGB")
        elif mode == "tail_clahe":
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim==3 else arr
            clahe_img = cv2.createCLAHE(clipLimit=3.0).apply(gray)
            return Image.fromarray(clahe_img).resize((target_w, target_h)).convert("RGB")
        
        return img.resize((target_w, target_h), Image.LANCZOS).convert("RGB")

    def _augment_tail(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        crop_x = int(w * 0.6)
        tail_part = img.crop((crop_x, 0, w, h))
        tail_resized = tail_part.resize((int(tail_part.width * 1.5), int(tail_part.height * 1.5)), Image.LANCZOS)
        new_w = w + tail_resized.width
        new_img = Image.new('RGB', (new_w, h), color='white')
        new_img.paste(img, (0, 0))
        new_img.paste(tail_resized, (w, 0))
        return new_img

    def _smart_post(self, text: str) -> str:
        if not text: return ""
        txt = text.strip().upper()
        tc = self._current_type
        
        if tc == 1002:
            txt = txt.translate(CORRECTION_DIGIT)
            return re.sub(r"[^0-9]", "", txt)
        elif tc == 1003:
            txt = txt.translate(CORRECTION_ALPHA)
            return re.sub(r"[^A-Z]", "", txt)
            
        clean = re.sub(r"[^A-Za-z0-9]", "", txt)
        allowed = TYPE_RANGES.get(tc)
        if allowed:
            clean = "".join([c for c in clean if c in allowed])
        return clean

    def _vote_system(self, candidates: List[Dict]) -> Tuple[str, float]:
        texts = [c["text"] for c in candidates]
        if not texts: return "", 0.0
        
        max_len = max(len(t) for t in texts)
        if max_len == 0: return "", 0.0
        
        votes = []
        total_agreement = 0
        
        for pos in range(max_len):
            cols = [t[pos] for t in texts if pos < len(t)]
            if not cols: 
                votes.append("?")
                continue
            counter = Counter(cols)
            most_common_ch, count = counter.most_common(1)[0]
            votes.append(most_common_ch)
            total_agreement += count
            
        best_text = "".join(votes)
        vote_count = len(candidates)
        score = total_agreement / (max_len * vote_count) if vote_count > 0 else 0
        confidence = 0.5 + (score * 0.5)
        
        return best_text, confidence


_instance = V8Solver()

def solve_ocr_v8(img_src, type_code=1001):
    return _instance.solve(img_src, type_code)
