# -*- coding: utf-8 -*-
"""自训练 CNN 第二引擎（4/5/6 位验证码）— onnxruntime 推理。
模型: cnn/captcha_cnn.onnx, 输入 1x1x48x256 灰度, 输出 1x6x37 (A-Z0-9 + blank)
"""
import io
import pathlib
import numpy as np
import cv2
from solver.utils import b64_to_bytes as _b64

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # 36
BLANK = 36
H, W = 48, 256
MAXLEN = 6
NUM_CLS = 37  # 36 chars + blank

_sess = None
def _session():
    global _sess
    if _sess is not None:
        return _sess
    try:
        import onnxruntime as ort
        p = pathlib.Path(__file__).resolve().parent.parent / "cnn" / "captcha_cnn.onnx"
        if not p.exists():
            return None
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
    except Exception:
        _sess = None
    return _sess

def _preprocess(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    scale = H / g.shape[0]
    nw = min(W, max(1, int(g.shape[1] * scale)))
    g = cv2.resize(g, (nw, H), interpolation=cv2.INTER_CUBIC)
    arr = np.zeros((H, W), dtype=np.float32)
    arr[:, :nw] = g.astype(np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5
    return arr[np.newaxis, np.newaxis, :, :]

def _decode(logits):
    # ONNX 输出可能是 (B, MAXLEN*NUM_CLS) 或 (B, MAXLEN, NUM_CLS)
    if logits.ndim == 2 and logits.shape[1] == MAXLEN * NUM_CLS:
        logits = logits.reshape(logits.shape[0], MAXLEN, NUM_CLS)
    # softmax -> 概率，取每位置最大概率均值作为置信度
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    idx = np.argmax(probs, axis=-1)[0]  # (6,)
    confs = np.max(probs, axis=-1)[0]
    text = "".join(CHARS[i] if i < BLANK else "" for i in idx)
    mean_conf = float(np.mean(confs))
    return text, mean_conf

def cnn_ocr(img_src):
    """返回 (text, conf) 或 (None, 0.0)。失败返回 (None, 0.0) 不抛错。"""
    sess = _session()
    if sess is None:
        return None, 0.0
    try:
        if isinstance(img_src, str):
            if len(img_src) > 200 or "," in img_src:
                b = _b64(img_src)
            else:
                b = open(img_src, "rb").read()
        elif isinstance(img_src, bytes):
            b = img_src
        else:
            b = img_src.read()
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        x = _preprocess(img)
        logits = sess.run(None, {"input": x})[0]  # (1,6,37)
        return _decode(logits)
    except Exception:
        return None, 0.0

def merge_engines(dd_text, dd_conf, cnn_text, cnn_conf, type_code):
    """双引擎投票（保守版）：仅在 CNN 长度相同且置信度显著更高时采用，
    避免弱 CNN 覆盖正确结果。返回 (text, source, conf)"""
    dd = (dd_text or "").strip().upper()
    cn = (cnn_text or "").strip().upper()
    if not cn:
        return dd, "dd", dd_conf
    if not dd:
        return cn, "cnn", cnn_conf
    # 保守：只有长度一致且 CNN 置信度显著高时才用 CNN
    if len(cn) == len(dd) and cnn_conf > dd_conf + 0.10:
        return cn, "cnn", cnn_conf
    return dd, "dd", dd_conf
