# -*- coding: utf-8 -*-
"""CRNN2 第二引擎推理封装：cnn/crnn2_ocr.onnx -> (text, conf)"""
import pathlib
import numpy as np
import cv2

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BLANK = 36
H, W = 52, 264
T = 16
NUM_CLS = 37

_sess = None
def _session():
    global _sess
    if _sess is not None:
        return _sess
    try:
        import onnxruntime as ort
        p = pathlib.Path(__file__).resolve().parent.parent / "cnn" / "crnn2_ocr.onnx"
        if not p.exists():
            return None
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
    except Exception:
        _sess = None
    return _sess

def _preprocess(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if g.shape[1] > W:
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
    arr = np.full((H, W), 255, dtype=np.float32)
    nw = g.shape[1]
    x0 = (W - nw) // 2
    arr[:, x0:x0+nw] = g.astype(np.float32)
    return (arr / 255.0)[np.newaxis, np.newaxis, :, :].astype(np.float32)

def _decode(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    idx = np.argmax(probs, axis=-1)[0]
    confs = np.max(probs, axis=-1)[0]
    s = []; prev = None; cs = []
    for t in range(T):
        v = int(idx[t])
        if v != BLANK and v != prev:
            s.append(CHARS[v])
            cs.append(confs[t])
        prev = v
    text = "".join(s)
    conf = float(np.mean(cs)) if cs else 0.0
    return text, conf

def _read_bytes(img_src):
    if isinstance(img_src, (str, pathlib.Path)):
        p = pathlib.Path(img_src)
        if p.exists():
            return p.read_bytes()
        if isinstance(img_src, str) and (len(img_src) > 200 or "," in img_src):
            from solver.utils import b64_to_bytes as _b64
            return _b64(img_src)
        return None
    if isinstance(img_src, bytes):
        return img_src
    if hasattr(img_src, "read"):
        return img_src.read()
    return None

def crnn_ocr(img_src):
    sess = _session()
    if sess is None:
        return "", 0.0
    try:
        b = _read_bytes(img_src)
        if not b:
            return "", 0.0
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            import io
            from PIL import Image
            img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        x = _preprocess(img)
        logits = sess.run(None, {"input": x})[0]
        return _decode(logits)
    except Exception:
        return "", 0.0
