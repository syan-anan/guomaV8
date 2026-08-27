# -*- coding: utf-8 -*-
"""
DDDDOCR V9 完美融合版 (Final Integration)
整合 RapidOCR 的预处理理念、PaddleOCR 的标准化策略以及 syandaV8 的高性能推理。
核心改进：自适应阈值、形态学去噪、尾部增强、多模型投票。
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
    warnings.warn("OpenCV (cv2) not found. Advanced preprocessing disabled.")

import ddddocr


# ==================== 核心配置 ====================
TYPE_RANGES = {
    1001: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", # 英数混合
    1002: "0123456789",                   # 纯数字
    1003: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" # 纯字母
}

CORRECTION_DIGIT = str.maketrans({
    'O': '0', 'Q': '0', 'G': '0', 'C': '0', 'D': '0', 
    'Z': '2', 'S': '5', '$': '5', 'B': '8'
})
CORRECTION_ALPHA = str.maketrans({
    '1': 'l', 'I': 'l', '|': 'l', '0': 'O', '5': 'S'
})


class V9Solver:
    """
    V9 求解器：多引擎 + 深度图像清洗 + 逻辑纠错
    """
    def __init__(self):
        self._ocr = ddddocr.DdddOcr(show_ad=False)
        self._ocr_beta = ddddocr.DdddOcr(beta=True, show_ad=False)
        
    def solve(self, img_src: Union[str, bytes], type_code: int = 1001) -> Dict:
        start_time = time.perf_counter()
        self._current_type = type_code
        
        try:
            img = self._load_image(img_src)
        except Exception as e:
            return {"code": -1, "text": "", "confidence": 0.0, "cost_ms": 0, "error": str(e)}

        candidates = []
        
        # 1. 基础变体
        candidates.append(self._run_engine(img, "basic"))
        
        # 2. 高级视觉增强（需 cv2）
        if HAS_CV2:
            candidates.append(self._run_engine(img, "adaptive_avg"))
            candidates.append(self._run_engine(img, "adaptive_gaussian"))
            candidates.append(self._run_engine(img, "morph_open"))
            candidates.append(self._run_engine(img, "morph_close"))
            
            # 3. 尾部增强变体池（解决漏字）
            img_tail = self._augment_tail(img)
            candidates.append(self._run_engine(img_tail, "tail_basic"))
            candidates.append(self._run_engine(img_tail, "tail_clahe"))

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

    # --- 辅助方法 ---

    def _load_image(self, src: Union[str, bytes]) -> Image.Image:
        if isinstance(src, str):
            if len(src) > 200 and "," in src: # Base64
                import base64
                bts = base64.b64decode(src.split(",")[-1])
            else:
                with open(src, "rb") as f: bts = f.read()
        else:
            bts = src
        return Image.open(io.BytesIO(bts)).convert("RGB")

    def _run_engine(self, img: Image.Image, mode: str) -> Dict:
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
        arr = np.array(img.convert("L"))
        
        if mode == "basic":
            return img.resize((target_w, target_h), Image.LANCZOS).convert("RGB")
        elif mode == "autocontrast":
            p = img.convert("L").resize((target_w, target_h), Image.LANCZOS)
            return ImageOps.autocontrast(p, cutoff=2).convert("RGB")
        elif mode == "adaptive_avg":
            binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(binary).resize((target_w, target_h)).convert("RGB")
        elif mode == "adaptive_gaussian":
            binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(binary).resize((target_w, target_h)).convert("RGB")
        elif mode == "morph_open":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
            _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(binary).resize((target_w, target_h)).convert("RGB")
        elif mode == "morph_close":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            filled = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
            _, binary = cv2.threshold(filled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(binary).resize((target_w, target_h)).convert("RGB")
        elif mode.startswith("tail"):
            tw, th = img.size
            nw = tw + (tw % 10 != 0)
            resized = img.resize((target_w, target_h), Image.LANCZOS)
            if "clahe" in mode and HAS_CV2:
                gray = np.array(resized.convert("L"))
                lab = cv2.cvtColor(cv2.merge([gray]*3), cv2.COLOR_RGB2LAB)
                l = lab[:,:,0]
                clahe = cv2.createCLAHE(clipLimit=3.0).apply(l)
                return Image.fromarray(cv2.cvtColor(cv2.merge([clahe]*3), cv2.COLOR_LAB2RGB))
            return resized

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


_instance = V9Solver()

def solve(img_src, type_code=1001):
    return _instance.solve(img_src, type_code)