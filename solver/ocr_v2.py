"""OCR engine V2: Fusion of PaddleOCR/RapidOCR/EasyOCR best practices."""
import io, re, time
import numpy as np
import cv2
import ddddocr
from collections import Counter

_models = {}
def _get_model(name):
    if name not in _models:
        _models[name] = ddddocr.DdddOcr(show_ad=False) if name == "default" else ddddocr.DdddOcr(beta=True, show_ad=False)
    return _models[name]

# --- 借鉴自 RapidOCR & PaddleOCR：高效预处理变体 ---
def _preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    v = []
    
    # 1. 基础二值化 (OTSU) - Tesseract 最常用且对验证码极有效
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v.append(("otsu", cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR)))
    
    # 2. 自适应阈值 (高斯加权) - 解决光照不均问题
    adj = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    v.append(("adapt", cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)))
    
    # 3. CLAHE + OTSU - 提升对比度后再二值化 (PaddleOCR 推荐)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    _, binary_clahe = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v.append(("clahe_otsu", cv2.cvtColor(binary_clahe, cv2.COLOR_GRAY2BGR)))
    
    # 4. 双边滤波降噪 + OTSU - RapidOCR 风格 (去噪同时保留文字边缘)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
    _, binary_bilat = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v.append(("bilateral", cv2.cvtColor(binary_bilat, cv2.COLOR_GRAY2BGR)))
    
    # 5. 反色处理 (应对白底黑字 vs 黑底白字混淆)
    inv = cv2.bitwise_not(gray)
    _, binary_inv = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v.append(("inv_otsu", cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR)))
    
    return v

def _encode(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()

# --- 借鉴自 EasyOCR：强力后处理纠错与字符合并逻辑 ---
_CORRECTION_MAP = str.maketrans({
    'O': '0', 'Q': '0', 'o': '0',
    'I': '1', 'l': '1', '|': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5', '$': '5',
    'B': '8', 'b': '8',
})

def _smart_post(text, type_code=1001):
    if not text: return ""
    cleaned = text.strip()
    
    # 合并连续重复字符 (如 dddocr 常把 11 识别为 1_1 或拉长)
    compact = []
    prev = None
    for ch in cleaned:
        if ch != prev: compact.append(ch)
        prev = ch
    cleaned = "".join(compact)
    
    # 强制映射纠正：利用字典替换极易混淆的字符
    mapped = cleaned.translate(_CORRECTION_MAP)
    
    # 按题型做严格的正则过滤
    if type_code == 1002: return re.sub(r"[^0-9]", "", mapped)
    elif type_code == 1003: return re.sub(r"[^A-Za-z]", "", mapped).upper()
    
    res = re.sub(r"[^A-Za-z0-9]", "", mapped).upper()
    alpha_num_count = sum(1 for c in res if c.isalnum())
    if len(res) > 0 and alpha_num_count / len(res) < 0.6: return ""
    return res

def _char_vote(results):
    valid = [re.sub(r"[^A-Za-z0-9]", "", r).upper() for r in results if r]
    if not valid: return "", 0.0
    
    max_len = max(len(x) for x in valid)
    if max_len == 0: return "", 0.0
    
    # Pad shorter strings to max_len so they align for voting
    padded = []
    for s in valid:
        padded.append(s.ljust(max_len, '_'))
    valid = padded
        
    chars = []
    agrees = 0
    for pos in range(max_len):
        cols = [x[pos] for x in valid]
        ch, nv = Counter(cols).most_common(1)[0]
        if ch != '_':
            chars.append(ch)
            agrees += nv
            
    series = "".join(chars)
    ratio = agrees / (max_len * len(valid)) if valid else 0.0
    return series, round(ratio, 4)

# 辅助函数加载图片
def _decode_src(img_src):
    if isinstance(img_src, str):
        if len(img_src) > 200 or "," in img_src:
            from solver.utils import b64_to_bytes
            b = b64_to_bytes(img_src)
        else:
            with open(img_src, "rb") as f: b = f.read()
    elif isinstance(img_src, bytes): b = img_src
    else: b = img_src.read()
    
    if b is None: raise ValueError("Empty image data")
    
    from PIL import Image
    img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def solve_ocr_v2(img_src, type_code=1001):
    t0 = time.perf_counter()
    try:
        img = _decode_src(img_src)
    except Exception as e:
        return {"code": -2, "error": "pic load fail: " + str(e), "text": "", "confidence": 0.0}

    versions = _preprocess(img)
    all_results = []
    
    for vname, vimg in versions:
        vbytes = _encode(vimg)
        for mname in ("default", "beta"):
            try:
                txt = _get_model(mname).classification(vbytes)
                clean_txt = _smart_post(txt, type_code)
                if clean_txt:
                    all_results.append(clean_txt)
            except: continue
            
    if not all_results:
        return {"code": -3, "error": "recog failed", "text": "", "confidence": 0.0}
        
    best, conf = _char_vote(all_results)
    result = {
        "code": 0, 
        "text": best, 
        "raw": " ".join(list(set(all_results))), 
        "confidence": conf, 
        "votes": len(all_results), 
        "agreement": len(all_results)
    }
    result["cost_ms"] = round((time.perf_counter()-t0)*1000, 1)
    return result
