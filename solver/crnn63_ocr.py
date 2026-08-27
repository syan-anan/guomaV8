# -*- coding: utf-8 -*-
"""Case-sensitive CRNN63 inference used as a gray release for OCR types."""
import json
import pathlib
import re

import cv2
import numpy as np


HEIGHT, WIDTH = 48, 160
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "cnn" / "crnn63_ocr.onnx"
META_PATH = MODEL_PATH.with_suffix(".json")

_metadata = {"chars": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", "blank": 62}
_session = None
# Conservative per-type ddddocr rescue gates, tuned on sample_bank_1500.
FALLBACK_THRESHOLDS = {1001: 0.78, 1002: 0.72, 1003: 0.76}
_fallback_models = {}


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        if META_PATH.exists():
            _metadata.update(json.loads(META_PATH.read_text(encoding="utf-8")))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
    return _session


def _read_bytes(image_source):
    if isinstance(image_source, (str, pathlib.Path)):
        path = pathlib.Path(image_source)
        if path.exists():
            return path.read_bytes()
        text = str(image_source)
        if len(text) > 200 or "," in text:
            import base64
            return base64.b64decode(text.split(",", 1)[-1])
        return None
    if isinstance(image_source, bytes):
        return image_source
    if hasattr(image_source, "read"):
        return image_source.read()
    return None


def _preprocess(raw):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        from PIL import Image
        import io
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return (image.astype(np.float32) / 255.0)[None, None, :, :]


def _get_fallback_model(type_code):
    name = "default" if type_code == 1002 else "beta"
    if name not in _fallback_models:
        import ddddocr

        kwargs = {"beta": True} if name == "beta" else {}
        _fallback_models[name] = ddddocr.DdddOcr(show_ad=False, **kwargs)
    return _fallback_models[name]


def _fallback_text(raw, type_code):
    """Rescue low-confidence cases without letting the second engine drive normal traffic."""
    try:
        result = str(_get_fallback_model(type_code).classification(raw)).strip().upper()
        if type_code == 1002:
            return "".join(c for c in result if c.isdigit())
        if type_code == 1003:
            return "".join(c for c in result if c.isalpha())
        return "".join(c for c in result if c.isalnum())
    except Exception:
        return ""


def crnn63_ocr(image_source):
    """Return (case-sensitive text, confidence)."""
    try:
        raw = _read_bytes(image_source)
        if not raw:
            return "", 0.0
        logits = _get_session().run(None, {"input": _preprocess(raw)})[0]
        maximum = logits.max(axis=-1, keepdims=True)
        exponent = np.exp(logits - maximum)
        probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
        tokens = probabilities[0].argmax(axis=-1)
        scores = probabilities[0].max(axis=-1)
        chars = _metadata["chars"]
        blank = int(_metadata["blank"])
        result, confidences, previous = [], [], None
        for token, score in zip(tokens, scores):
            token = int(token)
            if token != blank and token != previous:
                result.append(chars[token])
                confidences.append(float(score))
            previous = token
        return "".join(result), float(np.mean(confidences)) if confidences else 0.0
    except Exception:
        return "", 0.0


def solve_ocr63(image_source, type_code=1001):
    try:
        raw = _read_bytes(image_source)
    except Exception:
        raw = None
    if raw:
        text, confidence = crnn63_ocr(raw)
    else:
        text, confidence = "", 0.0
    if type_code == 1002:
        clean = "".join(c for c in text if c.isdigit())
    elif type_code == 1003:
        clean = "".join(c for c in text if c.isalpha())
    else:
        clean = "".join(c for c in text if c.isalnum())
    threshold = FALLBACK_THRESHOLDS.get(int(type_code))
    engine_name = "crnn63"
    if threshold is not None and clean and raw and confidence < threshold:
        rescued = _fallback_text(raw, int(type_code))
        if len(rescued) >= 2:
            clean = rescued
            text = rescued
            engine_name = "crnn63+ddddocr"
    return {
        "code": 0 if clean else -3,
        "text": clean,
        "raw": text,
        "confidence": round(confidence, 4),
        "engine": engine_name,
    }
