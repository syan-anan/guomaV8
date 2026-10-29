# -*- coding: utf-8 -*-
"""Second-style 1002 reader.

The deployed crnn63 route emits 0-5 characters on a dark, dashed-line 150x50
digit style that the xxapi generator now serves (C253: 19/300 verified through
the legacy path; C262: 40/40 empty or truncated). crnn63_t2b1.onnx reads that
style at 70-80% on fresh samples, every answer checked against the API oracle
(C256, C268).

Logit fusion across the three t2 models and any confidence-based arbitration
both fail here (C265, C268): confidence does not track correctness and fusing
different CTC alignments collapses the decode. So: one model, no fusion.

Style gate: a 7-feature logistic classifier fitted in C272 keeps 100% holdout
accuracy on 1000 mixed samples, routes 0 of 36 old dark single-digit rows to the
new side, and puts the fresh style at p >= 0.65. Only rows classified as the new
style are handed to this reader, so the old route keeps every row it can read.
"""
import pathlib
import threading

import cv2
import numpy as np

_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BLANK = 62
_MODEL = "crnn63_t2b4.onnx"
_MU = [141.466181, 0.264141, 0.118009, 57.890162, 0.14569, 8.176, 7.847]
_SD = [77.666788, 0.368717, 0.080124, 37.159495, 0.031233, 4.769174, 4.561534]
_W = [-2.145508, 2.198314, 0.007181, -2.120377, 3.220493, -1.333175, -1.222119]
_B = -7.469126
STYLE_THRESHOLD = 0.65
_lock = threading.Lock()
_state = {"session": None, "dir": None}


def _session(path):
    with _lock:
        if _state["session"] is not None and _state["dir"] == path:
            return _state["session"]
        import onnxruntime as ort
        s = ort.InferenceSession(str(path / _MODEL), providers=["CPUExecutionProvider"])
        _state["session"] = (s, s.get_inputs()[0].name)
        _state["dir"] = path
        return _state["session"]


def _features(g):
    im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bb, gr, r = cv2.split(im.astype(np.int16))
    colorfulness = float(np.mean(np.abs(r - gr) + np.abs(gr - bb) + np.abs(r - bb)))
    lum = float(g.mean())
    dark = float((g < 60).mean())
    mid = float(((g >= 60) & (g < 140)).mean())
    edges = cv2.Canny(g, 50, 150)
    ed = float((edges > 0).mean())
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=45, maxLineGap=8)
    nl = 0 if lines is None else len(lines)
    diag = 0
    if lines is not None:
        for l in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in l]
            if abs(x2 - x1) > 1 and abs((y2 - y1) / (x2 - x1)) < 0.6:
                diag += 1
    return [lum, dark, mid, colorfulness, ed, nl, diag]


def style_score(g):
    """Probability that the image belongs to the new dark dashed-line style."""
    f = np.array(_features(g), dtype=np.float64)
    z = (f - np.array(_MU)) / np.array(_SD)
    v = float(z @ np.array(_W) + _B)
    return 1.0 / (1.0 + np.exp(-v))


def _read_decoded(g, cnn_dir):
    s, inp = _session(cnn_dir)
    x = cv2.resize(g, (160, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    logits = s.run(None, {inp: x[None, None]})[0][0]
    e = logits - logits.max(-1, keepdims=True)
    p = np.exp(e)
    p /= p.sum(-1, keepdims=True)
    conf = float(p.max(-1).mean())
    idx = logits.argmax(-1)
    out, prev = [], None
    for v in idx:
        v = int(v)
        if v != _BLANK and v != prev:
            out.append(_CHARS[v])
        prev = v
    return "".join(out), conf


def read(raw, cnn_dir):
    """Return (text, confidence) for a 1002 captcha, or ('', 0.0)."""
    if not raw:
        return "", 0.0
    g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return "", 0.0
    return _read_decoded(g, cnn_dir)


def read_new_style_tta(raw, cnn_dir, shifts=(-2, 2)):
    """CNN-gated reader with 3-way shift TTA (C301/C306)."""
    if not raw:
        return "", 0.0
    g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return "", 0.0
    try:
        if cnn_style_score(raw) < CNN_THRESHOLD:
            return "", 0.0
    except Exception:
        return "", 0.0
    h, w = g.shape
    variants = [g]
    for dx in shifts:
        variants.append(cv2.warpAffine(g, np.float32([[1, 0, dx], [0, 1, 0]]), (w, h),
                                       borderMode=cv2.BORDER_REPLICATE))
    import collections
    preds = []
    confs = []
    for v in variants:
        t, c = _read_decoded(v, cnn_dir)
        preds.append(t)
        confs.append(c)
    counts = collections.Counter(len(x) for x in preds)
    length = counts.most_common(1)[0][0]
    cand = [x for x in preds if len(x) == length]
    if not cand:
        return "", 0.0
    vote = "".join(collections.Counter(x[i] for x in cand).most_common(1)[0][0]
                   for i in range(length))
    return vote, float(sum(confs) / len(confs))


def read_if_new_style(raw, cnn_dir):
    """Only answer when the image is clearly the new style."""
    if not raw:
        return "", 0.0
    g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return "", 0.0
    try:
        if style_score(g) < STYLE_THRESHOLD:
            return "", 0.0
    except Exception:
        return "", 0.0
    return read(raw, cnn_dir)

# --- C285/C287: CNN style discriminator (100% separation on 6128 old / 949 new) ---
CNN_MODEL = "t2_style_cnn.onnx"
CNN_THRESHOLD = 0.9
_cnn_state = {"session": None, "dir": None}


def _cnn_session(path):
    with _lock:
        if _cnn_state["session"] is not None and _cnn_state["dir"] == path:
            return _cnn_state["session"]
        import onnxruntime as ort
        s = ort.InferenceSession(str(path / CNN_MODEL), providers=["CPUExecutionProvider"])
        _cnn_state["session"] = (s, s.get_inputs()[0].name)
        _cnn_state["dir"] = path
        return _cnn_state["session"]


def cnn_style_score(raw):
    """Probability that the image is the new xxapi 150x50 dark-dashed style."""
    if not raw:
        return 0.0
    g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0.0
    s, inp = _cnn_session(pathlib.Path(__file__).resolve().parent.parent / "cnn")
    x = cv2.resize(g, (160, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    v = float(s.run(None, {inp: x[None, None]})[0][0])
    return 1.0 / (1.0 + np.exp(-v))


def read_if_new_style_cnn(raw, cnn_dir):
    """CNN-gated reader: returns text only for the new style."""
    if not raw:
        return "", 0.0
    try:
        if cnn_style_score(raw) < CNN_THRESHOLD:
            return "", 0.0
    except Exception:
        return "", 0.0
    return read(raw, cnn_dir)
