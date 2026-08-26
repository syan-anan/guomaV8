import os
os.chdir("H:/qinglong/syandaV8")
f = open("solver/ocr.py", "w", encoding="utf-8")
f.write("""
# -*- coding: utf-8 -*-
import re, collections
import numpy as np, cv2, syandaV8
from solver.utils import load_to_cv
from config import OCR_RETRY

CHARSETS = {
    1001: set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    1002: set("0123456789"),
    1003: set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}

_ocr_list = None
def _get_ocrs():
    global _ocr_list
    if _ocr_list is None:
        _ocr_list = [
            syandaV8.DdddOcr(show_ad=False),
            syandaV8.DdddOcr(show_ad=False, beta=True),
        ]
    return _ocr_list

def filter_result(text, charset):
    return "".join(c.upper() for c in text if c.upper() in charset)

def _get_versions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    versions = [img]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    versions.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    _, clahe_bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(cv2.cvtColor(clahe_bw, cv2.COLOR_GRAY2BGR))
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    scaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, scaled_bw = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(cv2.cvtColor(scaled_bw, cv2.COLOR_GRAY2BGR))
    inv = cv2.bitwise_not(otsu)
    versions.append(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR))
    return versions

def solve_ocr(img_src, type_code=1001):
    ocrs = _get_ocrs()
    img = load_to_cv(img_src)
    charset = CHARSETS.get(type_code, CHARSETS[1001])
    versions = _get_versions(img)
    candidates = []
    for v in versions:
        for o in ocrs:
            for _ in range(OCR_RETRY):
                try:
                    raw = o.classification(cv2.imencode(".png", v)[1].tobytes())
                    if raw:
                        candidates.append(raw)
                except:
                    pass
    filtered = [filter_result(c, charset) for c in candidates if filter_result(c, charset)]
    if not filtered:
        best = max(candidates, key=len) if candidates else ""
        return {"text": filter_result(best, charset), "raw": best, "confidence": 0}
    counter = collections.Counter(filtered)
    best_len = max(len(x) for x in filtered)
    long_ones = [x for x in filtered if len(x) == best_len]
    result = collections.Counter(long_ones).most_common(1)[0][0]
    if len(result) < 4:
        longer = [x for x in filtered if len(x) == 4]
        if longer:
            result = collections.Counter(longer).most_common(1)[0][0]
    return {"text": result, "raw": result, "confidence": min(1.0, len(long_ones) / len(candidates))}
""")
f.close()
print("ocr.py v2 written")
