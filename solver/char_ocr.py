# -*- coding: utf-8 -*-
"""单字符 CNN 滑动窗口识别：整图 -> 字符序列
用法: from solver.char_ocr import char_ocr; text = char_ocr(img_bgr, max_len=6)
"""
import io, pathlib
import numpy as np, cv2
from solver.utils import b64_to_bytes as _b64

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SIZE = 64
_sess = None
def _session():
    global _sess
    if _sess is not None:
        return _sess
    try:
        import onnxruntime as ort
        p = pathlib.Path(__file__).resolve().parent.parent / "cnn" / "char_cnn.onnx"
        if not p.exists():
            return None
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        _sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
    except Exception:
        _sess = None
    return _sess

def _classify_window(img, x, w):
    """对 (x, w) 窗口分类，返回 (char, prob)"""
    h, W = img.shape[:2]
    y0 = max(0, (h - 46) // 2)
    y1 = min(h, y0 + 46)
    x0 = max(0, x); x1 = min(W, x + w)
    if x1 - x0 < 12:
        return None, 0.0
    crop = img[y0:y1, x0:x1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    arr = (g.astype(np.float32) / 255.0 - 0.5) / 0.5
    xb = arr[np.newaxis, np.newaxis, :, :]
    logits = _session().run(None, {"input": xb})[0][0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    idx = int(np.argmax(probs))
    return CHARS[idx], float(probs[idx])

def char_ocr(img_src, max_len=6):
    """整图 -> (text, avg_conf)"""
    sess = _session()
    if sess is None:
        return "", 0.0
    if isinstance(img_src, str):
        b = open(img_src, "rb").read() if len(img_src) < 200 else _b64(img_src)
    elif isinstance(img_src, bytes):
        b = img_src
    else:
        b = img_src.read()
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(io.BytesIO(b)).convert("RGB")); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, W = img.shape[:2]
    # 滑动窗口打分：每个 x 位置取多宽度最佳
    step = 3
    scores = np.zeros(W); chars = [''] * W
    widths = [26, 30, 34, 38, 42, 46]
    for x in range(0, W - 12, step):
        best_c, best_p = '', 0.0
        for w in widths:
            c, p = _classify_window(img, x, w)
            if p > best_p:
                best_c, best_p = c, p
        scores[x] = best_p; chars[x] = best_c
    # 峰值提取（间隔 > 18px）
    peaks = []
    i = 0
    while i < W:
        if scores[i] < 0.3:
            i += 1; continue
        j = i
        while j + step < W and scores[j + step] > scores[j]:
            j += step
        # 检查左右是否更高（真峰）
        k = i
        while k - step >= 0 and scores[k - step] > scores[k]:
            k -= step
        if k == i and j == i:
            if not peaks or i - peaks[-1][0] > 18:
                peaks.append((i, chars[i], scores[i]))
            else:
                # 同一区域取更高分
                if scores[i] > peaks[-1][2]:
                    peaks[-1] = (i, chars[i], scores[i])
        i = j + step
    peaks.sort(key=lambda p: -p[2])
    # 保留 top max_len 个且从左到右
    peaks.sort(key=lambda p: p[0])
    peaks = peaks[:max_len]
    text = "".join(p[1] for p in peaks)
    conf = np.mean([p[2] for p in peaks]) if peaks else 0.0
    return text, float(conf)
