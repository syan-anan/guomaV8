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
_ensemble_sessions = None
# Conservative per-type ddddocr rescue gates, tuned on sample_bank_1500.
FALLBACK_THRESHOLDS = {1001: 0.65, 1002: 0.72, 1003: 0.76}
_fallback_models = {}
ARITHMETIC_RE = re.compile(
    r"^\s*(\d+)\s*([+\-x×*/÷])\s*(\d+)\s*(?:=|＝).*$",
    re.IGNORECASE,
)
_type_sessions = {}
_type_inferers = {}


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


def _get_sessions():
    global _ensemble_sessions
    if _ensemble_sessions is not None:
        return _ensemble_sessions
    base_session = _get_session()
    sessions = [{"session": base_session, "weight": 1.0}]
    base_dir = pathlib.Path(MODEL_PATH).resolve().parent
    for donor in _metadata.get("ensemble", []):
        donor_path = pathlib.Path(donor.get("model", ""))
        if not donor_path.is_absolute():
            donor_path = base_dir / donor_path
        donor_session = ort.InferenceSession(
            str(donor_path), sess_options=base_session.get_session_options(),
            providers=["CPUExecutionProvider"]
        )
        sessions.append({"session": donor_session, "weight": float(donor.get("weight", 0.0))})
    _ensemble_sessions = sessions
    return sessions


def _get_session_for_type(type_code=None):
    if type_code is None:
        return _get_session()
    try:
        type_key = str(int(type_code))
    except (TypeError, ValueError):
        return _get_session()

    if type_key in _type_sessions:
        return _type_sessions[type_key]

    base_session = _get_session()
    _get_session()
    routed = _metadata.get("type_models", {}).get(type_key)
    if not routed:
        return base_session

    import onnxruntime as ort

    model_path = pathlib.Path(routed.get("model", ""))
    if not model_path.is_absolute():
        model_path = pathlib.Path(MODEL_PATH).resolve().parent / model_path
    session = ort.InferenceSession(
        str(model_path),
        sess_options=base_session.get_session_options(),
        providers=["CPUExecutionProvider"],
    )
    _type_sessions[type_key] = session
    return session


def _get_type_inferer(type_code=None):
    if type_code is None:
        base_session = _get_session()
        return lambda image_input, raw=None: base_session.run(None, {"input": image_input})[0]
    try:
        type_key = str(int(type_code))
    except (TypeError, ValueError):
        base_session = _get_session()
        return lambda image_input, raw=None: base_session.run(None, {"input": image_input})[0]

    base_session = _get_session()
    if type_key in _type_inferers:
        return _type_inferers[type_key]

    routed = _metadata.get("type_models", {}).get(type_key)
    if not routed or "ensemble" not in routed:
        session = _get_session_for_type(type_code)
        inferer = lambda image_input, raw=None: session.run(None, {"input": image_input})[0]
        if routed and routed.get("tta"):
            inferer = _with_tta(inferer, routed["tta"], routed.get("tta_gate_threshold"))
        switch_path = routed.get("switch_model")
        if switch_path:
            import onnxruntime as ort
            switch_path = pathlib.Path(switch_path)
            if not switch_path.is_absolute():
                switch_path = MODEL_PATH.resolve().parent / switch_path
            switch_session = ort.InferenceSession(
                str(switch_path),
                sess_options=base_session.get_session_options(),
                providers=["CPUExecutionProvider"],
            )
            switch_threshold = routed.get("switch_threshold")
            switch_max_length = routed.get("switch_max_length")
            def infer_with_switch(image_input, raw=None, base_inferer=inferer):
                logits = base_inferer(image_input, raw)
                maximum = logits.max(axis=-1, keepdims=True)
                exponent = np.exp(logits - maximum)
                probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
                tokens = logits[0].argmax(axis=-1)
                scores = probabilities[0].max(axis=-1)
                char_scores = []
                switch_chars = []
                previous = None
                chars_meta = _metadata.get("chars", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
                for token, score in zip(tokens, scores):
                    token = int(token)
                    if token != int(_metadata.get("blank", 62)) and token != previous:
                        char_scores.append(float(score))
                        switch_chars.append(chars_meta[token])
                    previous = token
                confidence_hit = bool(char_scores) and switch_threshold is not None and float(np.mean(char_scores)) < float(switch_threshold)
                length_hit = switch_max_length is not None and len(switch_chars) <= int(switch_max_length)
                if confidence_hit or length_hit:
                    return switch_session.run(None, {"input": image_input})[0]
                return logits
            inferer = infer_with_switch
        _type_inferers[type_key] = inferer
        return inferer

    import onnxruntime as ort

    base_dir = pathlib.Path(MODEL_PATH).resolve().parent
    gate_path = routed.get("base")
    gate_session = None
    if gate_path:
        gate_model_path = pathlib.Path(gate_path)
        if not gate_model_path.is_absolute():
            gate_model_path = base_dir / gate_model_path
        gate_session = ort.InferenceSession(
            str(gate_model_path),
            sess_options=base_session.get_session_options(),
            providers=["CPUExecutionProvider"],
        )
    gate_threshold = routed.get("base_threshold")
    fusion_mode = routed.get("mode", "probs")
    members = []
    for donor in routed["ensemble"]:
        model_path = pathlib.Path(donor.get("model", ""))
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        session = ort.InferenceSession(
            str(model_path),
            sess_options=base_session.get_session_options(),
            providers=["CPUExecutionProvider"],
        )
        members.append((session, float(donor.get("weight", 0.0)), donor.get("tta"),
                        donor.get("height"), donor.get("width")))

    # Optional low-luminance (dark background) specialist path. Captcha styles drawn on a
    # black background are routed here when the mean luminance of the preprocessed input
    # is below dark_threshold; every other input keeps the normal path.
    dark_members = []
    dark_mode = routed.get("dark_mode", fusion_mode)
    for donor in routed.get("dark_ensemble", []) or []:
        model_path = pathlib.Path(donor.get("model", ""))
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        session = ort.InferenceSession(
            str(model_path),
            sess_options=base_session.get_session_options(),
            providers=["CPUExecutionProvider"],
        )
        dark_members.append((session, float(donor.get("weight", 0.0)), donor.get("tta"),
                             donor.get("height"), donor.get("width")))
    dark_threshold = routed.get("dark_threshold")

    # Optional disagreement gate for the dark branch (D6.50).
    #
    # Two dark specialists can be strong on different axes: one better at casing,
    # the other better at content.  Plain weighted voting cannot express that, because
    # whichever model has more members simply wins every casefold agreement too.
    # This gate instead compares the two decoded strings:
    #     casefold equal  -> keep the primary (better at casing)
    #     casefold differ -> take the arbiter (better at content)
    # Measured on holdout900's 707 real dark 1001 images: casefold unchanged (663)
    # while exact rises from 71 to 380.
    dark_gate_cfg = routed.get("dark_disagree_gate")
    dark_gate_sessions = None
    if dark_gate_cfg:
        dark_gate_sessions = []
        for role in ("primary", "arbiter"):
            spec = dark_gate_cfg.get(role) or {}
            if spec.get("use_dark_ensemble"):
                dark_gate_sessions.append(None)
                continue
            mp = pathlib.Path(spec.get("model", ""))
            if not mp.is_absolute():
                mp = base_dir / mp
            s = ort.InferenceSession(
                str(mp),
                sess_options=base_session.get_session_options(),
                providers=["CPUExecutionProvider"],
            )
            dark_gate_sessions.append((s, spec.get("height"), spec.get("width")))
    _gate_blank = int(_metadata.get("blank", 62))
    _gate_chars = _metadata.get("chars", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    def _gate_decode(logits):
        out, prev = [], None
        for token in logits[0].argmax(axis=-1):
            token = int(token)
            if token != _gate_blank and token != prev:
                out.append(_gate_chars[token])
            prev = token
        return "".join(out)

    def _member_input(image_input, variant, height=None, width=None, raw=None):
        """Per-member test-time augmentation + optional per-member input geometry.

        A route member may declare "tta": "<variant name>" (one of the names produced by
        _tta_variants).  The variant is applied to this member's own input only, so a
        single strong checkpoint can be averaged with its own shifted/sharpened copies
        without touching the other members.  Members without the key keep the exact
        previous behaviour.

        A member may additionally declare "height"/"width" to run at its own geometry
        (for example a 64x160 dark specialist fused with the 48x160 members).  Its input
        is then rebuilt from the raw image at that size.  Without those keys this function
        is byte-for-byte the previous behaviour.
        """
        base = image_input
        if height and width and raw is not None:
            base = _preprocess_at(raw, int(height), int(width), _luminance_policy(type_code))
        if not variant or variant == "original":
            return base
        gray = np.clip(np.asarray(base)[0, 0] * 255.0, 0, 255).astype(np.uint8)
        for name, candidate in _tta_variants(gray):
            if name == variant:
                if base is image_input:
                    return _tta_input(candidate)
                return _tta_input_at(candidate, int(width), int(height))
        return base

    def _mix(members_list, mode, image_input, raw=None):
        mixed = None
        for entry in members_list:
            session, weight = entry[0], entry[1]
            member_input = _member_input(image_input, entry[2] if len(entry) > 2 else None,
                                        entry[3] if len(entry) > 3 else None,
                                        entry[4] if len(entry) > 4 else None, raw)
            logits = session.run(None, {"input": member_input})[0]
            if mode == "logits":
                component = logits
            else:
                maximum = logits.max(axis=-1, keepdims=True)
                exponent = np.exp(logits - maximum)
                probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
                component = probabilities if mode == "probs" else np.log(probabilities + 1e-12)
            mixed = weight * component if mixed is None else mixed + weight * component
        return mixed

    def infer_ensemble(image_input, raw=None):
        if dark_members and dark_threshold is not None:
            luminance = float(np.asarray(image_input).mean()) * 255.0
            if luminance < float(dark_threshold):
                if dark_gate_sessions:
                    primary, arbiter = dark_gate_sessions
                    if primary is None:
                        p_logits = _mix(dark_members, dark_mode, image_input, raw)
                    else:
                        p_logits = primary[0].run(None, {"input": _member_input(
                            image_input, None, primary[1], primary[2], raw)})[0]
                    if arbiter is None:
                        # Reuse the incumbent dark ensemble verbatim, so the branch that
                        # takes the arbiter is byte-identical to the pre-gate behaviour.
                        a_logits = _mix(dark_members, dark_mode, image_input, raw)
                    else:
                        a_logits = arbiter[0].run(None, {"input": _member_input(
                            image_input, None, arbiter[1], arbiter[2], raw)})[0]
                    if _gate_decode(p_logits).casefold() == _gate_decode(a_logits).casefold():
                        return p_logits
                    return a_logits
                return _mix(dark_members, dark_mode, image_input, raw)
        if gate_session is not None and gate_threshold is not None:
            base_logits = gate_session.run(None, {"input": image_input})[0]
            maximum = base_logits.max(axis=-1, keepdims=True)
            exponent = np.exp(base_logits - maximum)
            probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
            scores = probabilities[0].max(axis=-1)
            tokens = base_logits[0].argmax(axis=-1)
            char_scores = []
            previous = None
            for token, score in zip(tokens, scores):
                token = int(token)
                if token != int(_metadata.get("blank", 62)) and token != previous:
                    char_scores.append(float(score))
                previous = token
            if char_scores and float(np.mean(char_scores)) >= float(gate_threshold):
                return base_logits
        return _mix(members, fusion_mode, image_input, raw)

    inferer = infer_ensemble
    if routed.get("tta"):
        inferer = _with_tta(inferer, routed["tta"], routed.get("tta_gate_threshold"))
    switch_path = routed.get("switch_model")
    if switch_path:
        import onnxruntime as ort
        switch_path = pathlib.Path(switch_path)
        if not switch_path.is_absolute():
            switch_path = MODEL_PATH.resolve().parent / switch_path
        switch_session = ort.InferenceSession(
            str(switch_path),
            sess_options=base_session.get_session_options(),
            providers=["CPUExecutionProvider"],
        )
        switch_threshold = routed.get("switch_threshold")
        switch_max_length = routed.get("switch_max_length")
        def infer_with_switch(image_input, raw=None, base_inferer=inferer):
            logits = base_inferer(image_input, raw)
            maximum = logits.max(axis=-1, keepdims=True)
            exponent = np.exp(logits - maximum)
            probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
            tokens = logits[0].argmax(axis=-1)
            scores = probabilities[0].max(axis=-1)
            char_scores = []
            switch_chars = []
            previous = None
            chars_meta = _metadata.get("chars", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            for token, score in zip(tokens, scores):
                token = int(token)
                if token != int(_metadata.get("blank", 62)) and token != previous:
                    char_scores.append(float(score))
                    switch_chars.append(chars_meta[token])
                previous = token
            confidence_hit = bool(char_scores) and switch_threshold is not None and float(np.mean(char_scores)) < float(switch_threshold)
            length_hit = switch_max_length is not None and len(switch_chars) <= int(switch_max_length)
            if confidence_hit or length_hit:
                return switch_session.run(None, {"input": image_input})[0]
            return logits
        inferer = infer_with_switch
    _type_inferers[type_key] = inferer
    return inferer


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


_LUMINANCE_LUTS = {}


def _luminance_lut(gamma):
    key = round(float(gamma), 3)
    lut = _LUMINANCE_LUTS.get(key)
    if lut is None:
        lut = np.array([min(255, int((i / 255.0) ** key * 255.0 + 0.5)) for i in range(256)], dtype=np.uint8)
        _LUMINANCE_LUTS[key] = lut
    return lut


def _luminance_policy(type_code):
    """Route-level switch: brighten low-luminance captures before the model sees them.

    The honest isolated bench showed dark-background rows score 2/20 as-is but 13/20 after a
    gamma lift, while the trained distribution is light-background. Absent from a route means no
    change, so every pre-existing route keeps its exact behaviour.
    """
    if type_code is None:
        return None
    try:
        type_key = str(int(type_code))
    except (TypeError, ValueError):
        return None
    routed = (_metadata.get("type_models") or {}).get(type_key) or {}
    policy = routed.get("luminance_policy") or {}
    if not policy.get("enabled"):
        return None
    return {"trigger": float(policy.get("trigger_gray_mean_lt", 0.0)),
            "gamma": float(policy.get("gamma", 1.0)),
            "invert": bool(policy.get("invert", False)),
            "autolevel": bool(policy.get("autolevel", False))}


def _apply_luminance_policy(gray, policy):
    if not policy:
        return gray
    if float(gray.mean()) >= policy["trigger"]:
        return gray
    image = cv2.bitwise_not(gray) if policy["invert"] else gray
    if policy["autolevel"]:
        low = float(np.percentile(image, 1))
        high = float(np.percentile(image, 99))
        image = np.clip((image.astype(np.float32) - low) * 255.0 / max(1.0, high - low), 0, 255).astype(np.uint8)
    gamma = policy["gamma"]
    if gamma != 1.0:
        image = cv2.LUT(image, _luminance_lut(gamma))
    return image


def _preprocess(raw, policy=None):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        from PIL import Image
        import io
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    image = _apply_luminance_policy(image, policy)
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return (image.astype(np.float32) / 255.0)[None, None, :, :]


def _letterbox(gray, height, width, pad_value=None):
    """Aspect-preserving fit: scale to `height` keeping the native ratio, then pad right.

    Added for the letterbox branches.  The default member path stays a pure stretch, so
    every pre-existing route is unaffected.
    """
    h, w = gray.shape[:2]
    inner = int(max(1, min(width, round(height * w / float(h)))))
    resized = cv2.resize(gray, (inner, height), interpolation=cv2.INTER_AREA)
    if inner == width:
        return resized
    if pad_value is None:
        border = np.concatenate([resized[:, :1].ravel(), resized[:, -1:].ravel(),
                                 resized[:1, :].ravel(), resized[-1:, :].ravel()])
        pad_value = int(np.median(border))
    canvas = np.full((height, width), pad_value, dtype=np.uint8)
    canvas[:, :inner] = resized
    return canvas


def _tta_variants(gray):
    height, width = gray.shape[:2]
    yield "original", gray
    for scale in (0.96, 1.04):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
        yield f"scale_{scale}", cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    for dx, dy in ((-2, 0), (2, 0), (0, -1), (0, 1)):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        yield f"shift_{dx}_{dy}", cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    yield "contrast_090", cv2.convertScaleAbs(gray, alpha=0.90, beta=6)
    yield "contrast_110", cv2.convertScaleAbs(gray, alpha=1.10, beta=-6)
    yield "blur_03", cv2.GaussianBlur(gray, (3, 3), 0.35)
    for angle in (-2, 2):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        yield f"rotate_{angle}", cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    for gamma in (0.85, 1.15):
        lut = np.array([min(255, int((i / 255.0) ** gamma * 255.0 + 0.5)) for i in range(256)], dtype=np.uint8)
        yield f"gamma_{int(round(gamma * 100))}", cv2.LUT(gray, lut)
    yield "clahe_2", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    yield "sharpen", cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    _ch, _cw = int(round(height * 0.9)), int(round(width * 0.9))
    _y0, _x0 = (height - _ch) // 2, (width - _cw) // 2
    yield "crop_90", cv2.resize(gray[_y0:_y0 + _ch, _x0:_x0 + _cw], (width, height), interpolation=cv2.INTER_AREA)
    for dx, dy in ((1, -1), (-1, 1)):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        yield f"shift_{dx}_{dy}", cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 0.98)
    yield "scale_0.98", cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    for gamma in (0.70, 1.30):
        lut = np.array([min(255, int((i / 255.0) ** gamma * 255.0 + 0.5)) for i in range(256)], dtype=np.uint8)
        yield f"gamma_{int(round(gamma * 100))}", cv2.LUT(gray, lut)
    yield "autocontrast", cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    yield "equalize", cv2.equalizeHist(gray)
    yield "median_3", cv2.medianBlur(gray, 3)


def _tta_input(gray):
    resized = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[None, None, :, :]


def _tta_input_at(gray, width, height):
    """Same normalisation as _tta_input but at an explicit geometry (per-member size)."""
    resized = cv2.resize(gray, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[None, None, :, :]


def _preprocess_at(raw, height, width, policy=None):
    """Same normalisation as _preprocess but at an explicit geometry (per-member size)."""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        from PIL import Image
        import io
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    image = _apply_luminance_policy(image, policy)
    image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return (image.astype(np.float32) / 255.0)[None, None, :, :]


def _with_tta(inferer, tta_names, gate_threshold=None):
    def run_tta(image_input, raw=None):
        base_logits = inferer(image_input, raw)
        if gate_threshold is not None:
            maximum = base_logits.max(axis=-1, keepdims=True)
            exponent = np.exp(base_logits - maximum)
            probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
            tokens = base_logits[0].argmax(axis=-1)
            scores = probabilities[0].max(axis=-1)
            char_scores = []
            previous = None
            for token, score in zip(tokens, scores):
                token = int(token)
                if token != int(_metadata.get("blank", 62)) and token != previous:
                    char_scores.append(float(score))
                previous = token
            if char_scores and float(np.mean(char_scores)) >= float(gate_threshold):
                return base_logits
        gray = np.clip(image_input[0, 0] * 255.0, 0, 255).astype(np.uint8)
        mixed = None
        count = 0
        for name, variant in _tta_variants(gray):
            if name not in tta_names:
                continue
            output = inferer(_tta_input(variant), raw)
            mixed = output if mixed is None else mixed + output
            count += 1
        return mixed / max(count, 1)
    return run_tta


def _get_fallback_model(type_code, beta=None):
    if beta is None:
        name = "default" if type_code == 1002 else "beta"
    else:
        name = "beta" if beta else "default"
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


def _arithmetic_answer(text):
    """Evaluate simple OCR-visible arithmetic like 9x4=? or 12+3=?."""
    match = ARITHMETIC_RE.match(str(text or "").replace("−", "-").replace("＝", "="))
    if not match:
        return None
    left = int(match.group(1))
    operator = match.group(2).lower()
    right = int(match.group(3))
    try:
        if operator in ("+",):
            value = left + right
        elif operator in ("-",):
            value = left - right
        elif operator in ("x", "×", "*"):
            value = left * right
        elif operator in ("/", "÷"):
            value = left // right if right and left % right == 0 else left / right
        else:
            return None
    except Exception:
        return None
    return str(int(value)) if float(value).is_integer() else str(value)



def _get_noise_inferer(type_code, spec):
    """Inferer for the low-luminance (dense-static) regime, built from a route-level spec.

    Kept separate from _get_type_inferer so the clean-traffic path is untouched: rows above the
    threshold never call this, which is what makes a noise switch unable to regress real3/real19.
    """
    key = '%s|noise|%s' % (type_code, json.dumps(spec, sort_keys=True))
    cached = _type_inferers.get(key)
    if cached is not None:
        return cached
    import onnxruntime as ort

    base_session = _get_session()
    base_dir = pathlib.Path(MODEL_PATH).resolve().parent

    def _session_for(model_name):
        model_path = pathlib.Path(model_name)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        return ort.InferenceSession(str(model_path), sess_options=base_session.get_session_options(),
                                    providers=["CPUExecutionProvider"])

    mode = spec.get('mode', 'logits')
    donors = spec.get('ensemble')
    if donors:
        members = [(_session_for(d.get('model', '')), float(d.get('weight', 0.0))) for d in donors]
    else:
        members = [(_session_for(spec.get('model', '')), 1.0)]

    def infer(image_input, raw=None):
        mixed = None
        for session, weight in members:
            logits = session.run(None, {'input': image_input})[0]
            if mode == 'logits':
                component = logits
            else:
                maximum = logits.max(axis=-1, keepdims=True)
                exponent = np.exp(logits - maximum)
                probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
                component = probabilities if mode == 'probs' else np.log(probabilities + 1e-12)
            mixed = weight * component if mixed is None else mixed + weight * component
        return mixed

    _type_inferers[key] = infer
    return infer


def crnn63_ocr(image_source, type_code=None):
    """Return (case-sensitive text, confidence)."""
    try:
        raw = _read_bytes(image_source)
        if not raw:
            return "", 0.0
        image_input = _preprocess(raw, _luminance_policy(type_code))
        inferer = _get_type_inferer(type_code)
        switch = None
        if type_code is not None:
            try:
                switch = ((_metadata.get("type_models") or {}).get(str(int(type_code))) or {}).get("noise_switch")
            except (TypeError, ValueError):
                switch = None
        if switch:
            raw_gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            metric = switch.get("metric", "luminance")
            if metric == "median_diff" and raw_gray is not None:
                edge = float(np.abs(raw_gray.astype(np.float32) -
                                    cv2.medianBlur(raw_gray, 3).astype(np.float32)).mean())
                if edge >= float(switch.get("threshold", 12.0)):
                    inferer = _get_noise_inferer(type_code, switch)
            elif raw_gray is not None:
                luminance = float(raw_gray.mean())
                if luminance < float(switch.get("threshold", 200.0)):
                    inferer = _get_noise_inferer(type_code, switch)
        logits = inferer(image_input, raw)
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


_arbiter_sessions = {}


def _case_arbitrate(raw, result, type_code):
    """casefold 中性仲裁：仅当仲裁模型与当前答案 casefold 相同但字面不同、且置信度达标时替换。
    因此对 casefold 指标可证明零影响，只移动 exact（真实大小写敏感站点的能力轴）。"""
    try:
        arbiters = _metadata.get("case_arbiters") or []
        if not arbiters or not raw or not result or result.get("code") != 0:
            return result
        text = result.get("text") or ""
        if not text:
            return result
        import onnxruntime as _ort
        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return result
        height, width = gray.shape[:2]
        key = str(int(type_code)) if type_code is not None else None
        base_dir = pathlib.Path(MODEL_PATH).resolve().parent
        chars = _metadata.get("chars", "")
        blank = int(_metadata.get("blank", 62))
        changed = False
        for arb in arbiters:
            when = arb.get("when") or {}
            if when.get("width") is not None and int(when["width"]) != int(width):
                continue
            if when.get("height") is not None and int(when["height"]) != int(height):
                continue
            lum_lt = when.get("luminance_lt")
            lum_gt = when.get("luminance_gt")
            if lum_lt is not None or lum_gt is not None:
                mean_lum = float(gray.mean())
                if lum_lt is not None and not (mean_lum < float(lum_lt)):
                    continue
                if lum_gt is not None and not (mean_lum > float(lum_gt)):
                    continue
            types = arb.get("types")
            if types and key not in [str(t) for t in types]:
                continue
            path = pathlib.Path(arb.get("model", ""))
            if not path.is_absolute():
                path = base_dir / path
            sess = _arbiter_sessions.get(str(path))
            if sess is None:
                sess = _ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                _arbiter_sessions[str(path)] = sess
            ah = int(arb.get("height") or HEIGHT)
            aw = int(arb.get("width") or WIDTH)
            rz = cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA)
            tensor = (rz.astype(np.float32) / 255.0)[None, None, :, :]
            atext, aconf = _domain_decode(sess.run(None, {"input": tensor})[0], chars, blank)
            atext = "".join(c for c in atext if c.isalnum())
            if not atext or aconf < float(arb.get("min_confidence", 0.0)):
                continue
            if atext != text and atext.casefold() == text.casefold():
                text = atext
                changed = True
        if changed:
            result = dict(result)
            result["text"] = text
            result["raw"] = text
            result["engine"] = str(result.get("engine", "")) + "+casefix"
        return result
    except Exception:
        return result


def solve_ocr63(image_source, type_code=1001):
    try:
        raw = _read_bytes(image_source)
    except Exception:
        raw = None
    return _case_arbitrate(raw, _solve_ocr63_core(image_source, type_code), type_code)


def _solve_ocr63_core(image_source, type_code=1001):
    try:
        raw = _read_bytes(image_source)
    except Exception:
        raw = None
    if raw:
        if int(type_code) == 1002:
            try:
                from solver.t2_1002 import read_if_new_style_cnn as _t2c_read
                _t2c_text, _t2c_conf = _t2c_read(raw, MODEL_PATH.parent)
                _t2c_clean = "".join(c for c in _t2c_text if c.isdigit())
                if 4 <= len(_t2c_clean) <= 8:
                    return {"code": 0, "text": _t2c_clean, "raw": _t2c_text,
                            "confidence": round(float(_t2c_conf), 4), "engine": "t2_1002"}
            except Exception:
                pass
        domain = domain_vote(raw, int(type_code))
        if domain:
            dtext, dconf = domain
            if int(type_code) == 1002:
                dclean = "".join(c for c in dtext if c.isdigit())
            elif int(type_code) == 1003:
                dclean = "".join(c for c in dtext if c.isalpha())
            else:
                dclean = "".join(c for c in dtext if c.isalnum())
            if dclean:
                return {"code": 0, "text": dclean, "raw": dtext,
                        "confidence": round(float(dconf), 4), "engine": "domain_vote"}
        text, confidence = crnn63_ocr(raw, int(type_code))
    else:
        text, confidence = "", 0.0
    if type_code == 1002:
        clean = "".join(c for c in text if c.isdigit())
    elif type_code == 1003:
        clean = "".join(c for c in text if c.isalpha())
    else:
        clean = "".join(c for c in text if c.isalnum())
    if int(type_code) == 1002 and len(clean) < 4 and raw:
        try:
            from solver.t2_1002 import read_if_new_style as _t2_read
            _t2_text, _t2_conf = _t2_read(raw, MODEL_PATH.parent)
            _t2_clean = "".join(c for c in _t2_text if c.isdigit())
            if 4 <= len(_t2_clean) <= 8:
                return {"code": 0, "text": _t2_clean, "raw": _t2_text,
                        "confidence": round(float(_t2_conf), 4), "engine": "t2_1002"}
        except Exception:
            pass
    arithmetic = None
    if int(type_code) == 1002 and text and not text.isdigit():
        beta_text = str(_get_fallback_model(1002, beta=True).classification(raw)).strip()
        arithmetic = _arithmetic_answer(beta_text)
        if arithmetic is not None:
            return {
                "code": 0,
                "text": arithmetic,
                "raw": beta_text,
                "confidence": round(confidence, 4),
                "engine": "crnn63+arithmetic_rescue",
            }
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


_domain_sessions = {}


def _domain_session(path):
    session = _domain_sessions.get(path)
    if session is None:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _domain_sessions[path] = session
    return session


def _pos4_decode(logits):
    """位置式 4x10 头：每位置 argmax，置信度 = 4 个位置最大 softmax 的均值。"""
    x = logits[0]
    maximum = x.max(axis=-1, keepdims=True)
    exponent = np.exp(x - maximum)
    probs = exponent / exponent.sum(axis=-1, keepdims=True)
    text = "".join(str(int(v)) for v in probs.argmax(axis=-1))
    return text, float(np.mean(probs.max(axis=-1)))


def _domain_decode(logits, chars, blank):
    """Greedy CTC decode plus the same confidence definition used by the D6.0O harness
    (mean over all timesteps of the max softmax probability)."""
    maximum = logits.max(axis=-1, keepdims=True)
    exponent = np.exp(logits - maximum)
    probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
    tokens = probabilities[0].argmax(axis=-1)
    confidence = float(np.mean(probabilities[0].max(axis=-1)))
    text, previous = [], None
    for token in tokens:
        token = int(token)
        if token != blank and token != previous:
            text.append(chars[token])
        previous = token
    return "".join(text), confidence


def _ensure_metadata():
    """冷启动保护：任何消费 _metadata 的入口先确保路由元数据已加载。

    _get_session() 只在主 session 创建时加载 metadata；在冷进程（或重置 _metadata 的
    测试里）第一次调用 domain_vote 时会直接跳过 domain 分支。这里补上懒加载。"""
    if not (_metadata.get("domain_branches") or _metadata.get("case_pattern_heads")) and META_PATH.exists():
        try:
            _metadata.update(json.loads(META_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass


def domain_vote(image_source, type_code=None):
    """Weighted-confidence vote over a dimension-keyed domain ensemble.

    Members keep their own native input size, so this cannot be expressed as a logit mix.
    Returns (text, confidence) or None when no branch matches or the winner is below the
    branch min_confidence (caller then falls back to the normal pipeline)."""
    _ensure_metadata()
    branches = _metadata.get("domain_branches") or []
    if not branches:
        return None
    try:
        raw = _read_bytes(image_source)
        if not raw:
            return None
        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        height, width = gray.shape[:2]
    except Exception:
        return None
    key = str(int(type_code)) if type_code is not None else None
    base_dir = pathlib.Path(MODEL_PATH).resolve().parent
    for branch in branches:
        when = branch.get("when") or {}
        if when.get("width") is not None and int(when["width"]) != int(width):
            continue
        if when.get("height") is not None and int(when["height"]) != int(height):
            continue
        # 亮度条件（可选）：灰度均值。用于把同一尺寸下的暗底/亮底拆成不同专才分支。
        lum_lt = when.get("luminance_lt")
        lum_gt = when.get("luminance_gt")
        if lum_lt is not None or lum_gt is not None:
            mean_lum = float(gray.mean())
            if lum_lt is not None and not (mean_lum < float(lum_lt)):
                continue
            if lum_gt is not None and not (mean_lum > float(lum_gt)):
                continue
        types = branch.get("types")
        if types and key not in [str(t) for t in types]:
            continue
        chars = branch.get("chars") or _metadata.get("chars", "")
        blank = int(branch.get("blank", _metadata.get("blank", 62)))
        scores = {}
        total_weight = 0.0
        fusion_mode = str(branch.get("fusion", "vote")).lower()
        logit_acc = [None]
        variant_cache = {}
        color_gray = [None]
        for member in branch.get("members", []):
            path = pathlib.Path(member.get("model", ""))
            if not path.is_absolute():
                path = base_dir / path
            session = _domain_session(str(path))
            member_height = int(member.get("height") or HEIGHT)
            member_width = int(member.get("width") or WIDTH)
            member_tta = member.get("tta")
            member_gray = gray
            if member_tta and member_tta != "original":
                member_gray = variant_cache.get(member_tta)
                if member_gray is None:
                    for _vname, _vcand in _tta_variants(gray):
                        if _vname == member_tta:
                            member_gray = _vcand
                            variant_cache[member_tta] = _vcand
                            break
                if member_gray is None:
                    raise SystemExit("domain_vote: unknown tta variant " + str(member_tta))
            if str(member.get("kind", "")).lower() == "pos4":
                if color_gray[0] is None:
                    _bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    color_gray[0] = None if _bgr is None else cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
                src = color_gray[0]
                if src is None:
                    continue
                # pos4 成员暂不支持 TTA（灰度域 TTA 变体对 RGB 输入无意义）
                _interp = cv2.INTER_LINEAR if str(member.get("interp", "")).lower() == "linear" else cv2.INTER_AREA
                resized = cv2.resize(src, (member_width, member_height), interpolation=_interp)
                tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, :, :, :]
                logits = session.run(None, {"input": tensor})[0]
                if fusion_mode == "logits":
                    # Probability-level fusion: accumulate per-position distributions instead of
                    # voting on the decoded string.  Keeps the full 4x10 information, so a strong
                    # member can carry a position the others are split on.
                    _x = logits[0]
                    _mx = _x.max(axis=-1, keepdims=True)
                    _ex = np.exp(_x - _mx)
                    _pr = _ex / _ex.sum(axis=-1, keepdims=True)
                    _w = float(member.get("weight", 1.0))
                    logit_acc[0] = _w * _pr if logit_acc[0] is None else logit_acc[0] + _w * _pr
                    continue
                text, confidence = _pos4_decode(logits)
            else:
                if str(member.get("fit", "")).lower() == "letterbox":
                    resized = _letterbox(member_gray, member_height, member_width)
                else:
                    resized = cv2.resize(member_gray, (member_width, member_height), interpolation=cv2.INTER_AREA)
                tensor = (resized.astype(np.float32) / 255.0)[None, None, :, :]
                logits = session.run(None, {"input": tensor})[0]
                text, confidence = _domain_decode(logits, chars, blank)
            weight = float(member.get("weight", 1.0))
            total_weight += weight
            if text:
                scores[text] = scores.get(text, 0.0) + weight * confidence
        if not scores:
            if fusion_mode == "logits" and logit_acc[0] is not None:
                _pr = logit_acc[0]
                _pr = _pr / _pr.sum(axis=-1, keepdims=True)
                winner = "".join(str(int(v)) for v in _pr.argmax(axis=-1))
                confidence = float(np.mean(_pr.max(axis=-1)))
                if confidence < float(branch.get("min_confidence", 0.0)):
                    return None
                return winner, confidence
            return None
        winner = max(scores, key=scores.get)
        confidence = scores[winner] / total_weight if total_weight else 0.0
        if confidence < float(branch.get("min_confidence", 0.0)):
            return None
        return winner, confidence
    return None


_charfix_sessions = {}

_DEFAULT_CHARFIX_GROUPS = [["0", "o"], ["1", "i", "l"], ["8", "b"], ["2", "z"], ["5", "s"],
                           ["6", "g"], ["9", "g", "q"], ["4", "a"], ["u", "v"], ["c", "e"], ["7", "1"]]


def _charfix_session(path):
    session = _charfix_sessions.get(path)
    if session is None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        _charfix_sessions[path] = session
    return session


def char_fix(text, image_source, type_code=None):
    """同形字改判后处理（可选能力，路由里没有 char_fix_branches 时完全无副作用）。

    对主链路已产出的文本，按标签长度把原图均匀等分成单字列，喂给单字分类器；
    只在「当前字符所在的同形组」内做条件 argmax，组内最优不是当前字符且组内概率 >= min_confidence
    时才替换。返回 (new_text, n_changes)。

    crop_mode=native_window（新增）：不再把单字列拉伸到固定框（那会抹掉字号，所以老的
    同形字模型只能是 casefold 的），而是以列中心为轴取一个固定宽度的原生像素窗口、
    等比放大到模型输入尺寸，字相对基线的高度得以保留，于是同一个头能同时回答
    0/O/o 和 1/i/I/l/L。判定基准也从 lower 后的 current 改成原始 original，
    否则 O->o 方向永远改不动（'o' == current 会被当成无需修改）。
    """
    if not text:
        return text, 0
    _ensure_metadata()
    branches = _metadata.get("char_fix_branches") or []
    if not branches:
        return text, 0
    try:
        raw = _read_bytes(image_source)
        if not raw:
            return text, 0
        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return text, 0
        height, width = gray.shape[:2]
    except Exception:
        return text, 0
    key = str(int(type_code)) if type_code is not None else None
    base_dir = pathlib.Path(MODEL_PATH).resolve().parent
    for branch in branches:
        when = branch.get("when") or {}
        if when.get("width") is not None and int(when["width"]) != int(width):
            continue
        if when.get("height") is not None and int(when["height"]) != int(height):
            continue
        lum_lt = when.get("luminance_lt")
        lum_gt = when.get("luminance_gt")
        if lum_lt is not None or lum_gt is not None:
            mean_lum = float(gray.mean())
            if lum_lt is not None and not (mean_lum < float(lum_lt)):
                continue
            if lum_gt is not None and not (mean_lum > float(lum_gt)):
                continue
        types = branch.get("types")
        if types and key not in [str(t) for t in types]:
            continue
        chars = branch.get("chars") or "0123456789abcdefghijklmnopqrstuvwxyz"
        index = {c: i for i, c in enumerate(chars)}
        groups = branch.get("groups") or _DEFAULT_CHARFIX_GROUPS
        group_of = {}
        for members in groups:
            for member in members:
                group_of.setdefault(member, set()).update(members)
        tau = float(branch.get("min_confidence", 0.9))
        model_path = pathlib.Path(branch.get("model", ""))
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        session = _charfix_session(str(model_path))
        crop_h = int(branch.get("crop_height") or 24)
        crop_w = int(branch.get("crop_width") or 32)
        crop_mode = branch.get("crop_mode") or "column"
        window_width = int(branch.get("window_width") or 40)
        lowered = text.lower()
        count = len(lowered)
        if count <= 0:
            return text, 0
        out = []
        changed = 0
        for position, current in enumerate(lowered):
            original = text[position] if position < len(text) else current
            members = group_of.get(current)
            if not members:
                out.append(original)
                continue
            x0 = int(round(position * width / count))
            x1 = int(round((position + 1) * width / count))
            ordered = [c for c in sorted(members) if c in index]
            if x1 <= x0 or not ordered:
                out.append(original)
                continue
            if crop_mode == "native_window":
                left = int(round((x0 + x1) / 2.0 - window_width / 2.0))
                full_h = int(gray.shape[0])
                buf = np.zeros((full_h, window_width), np.uint8)
                sx0 = max(0, left)
                sx1 = min(width, left + window_width)
                if sx1 > sx0:
                    buf[:full_h, sx0 - left:sx1 - left] = gray[:full_h, sx0:sx1]
                crop = cv2.resize(buf, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            else:
                crop = cv2.resize(gray[:, x0:x1], (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            tensor = (crop.astype(np.float32) / 255.0)[None, None, :, :]
            logits = session.run(None, {"input": tensor})[0][0]
            shifted = np.exp(logits - logits.max())
            probabilities = shifted / shifted.sum()
            sub = probabilities[[index[c] for c in ordered]]
            total = float(sub.sum())
            best_index = int(np.argmax(sub))
            best = ordered[best_index]
            probability = float(sub[best_index]) / total if total > 0 else 0.0
            if best != original and probability >= tau:
                if branch.get("preserve_case") and original.isupper():
                    out.append(best.upper())
                else:
                    out.append(best)
                changed += 1
            else:
                out.append(original)
        if changed == 0:
            return text, 0
        return "".join(out), changed
    return text, 0
_casehead_sessions = {}
CASEHEAD_SKIP = {}

_READER_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_READER_INDEX = {c: i for i, c in enumerate(_READER_CHARS)}
_READER_NEG = -1e4


def _casehead_report(reason, **kw):
    entry = CASEHEAD_SKIP.setdefault(reason, {"n": 0})
    entry["n"] += 1
    for key, value in kw.items():
        entry[key] = value


def _casehead_session(path):
    sess = _casehead_sessions.get(str(path))
    if sess is None:
        import onnxruntime as _ort
        sess = _ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _casehead_sessions[str(path)] = sess
    return sess


def _casehead_input(sess, gray, aim_h, aim_w):
    resized = cv2.resize(gray, (aim_w, aim_h), interpolation=cv2.INTER_AREA)
    tensor = (resized.astype(np.float32) / 255.0)[None, None, :, :]
    return sess.run(None, {"input": tensor})[0]



def _casehead_ctc_forced(probabilities, alpha_positions, blank):
    """CTC forced alignment over the 2N+1 extended state lattice.

    Used only as a fallback when free-decoding peak counting does not match the
    base text length, so incumbent behaviour on samples that already align is
    bit-identical. Returns {slot_index: (p_upper, p_lower)}.
    """
    import numpy as _np
    n = len(alpha_positions)
    if n == 0:
        return {}
    T = int(probabilities.shape[0])
    if T < n:
        return {}
    neg = -1e18
    small = 1e-12
    p_blank = _np.clip(probabilities[:, blank], small, None)
    p_char = _np.clip(probabilities[:, 0] + probabilities[:, 1], small, None)
    em_blank = _np.log(p_blank)
    em_char = _np.log(p_char)
    S = 2 * n + 1
    state_is_char = [(s % 2) == 1 for s in range(S)]
    alpha = _np.full((T, S), neg)
    alpha[0, 0] = em_blank[0]
    if S > 1:
        alpha[0, 1] = em_char[0]
    for t in range(1, T):
        prev = alpha[t - 1]
        for s in range(S):
            best = prev[s]
            if s >= 1 and prev[s - 1] > best:
                best = prev[s - 1]
            if s >= 2 and state_is_char[s] and prev[s - 2] > best:
                best = prev[s - 2]
            alpha[t, s] = best + (em_char[t] if state_is_char[s] else em_blank[t])
    beta = _np.full((T, S), neg)
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1]
        for s in range(S):
            best = nxt[s]
            if s + 1 < S and nxt[s + 1] > best:
                best = nxt[s + 1]
            if s + 2 < S and state_is_char[s + 2] and nxt[s + 2] > best:
                best = nxt[s + 2]
            beta[t, s] = best + (em_char[t] if state_is_char[s] else em_blank[t])
    total = alpha[T - 1, S - 1]
    if total < neg / 2:
        return {}
    out = {}
    for i in range(n):
        s = 2 * i + 1
        gamma = alpha[:, s] + beta[:, s] - total
        t_best = int(_np.argmax(gamma))
        pu = float(probabilities[t_best, 0])
        pl = float(probabilities[t_best, 1])
        tot = pu + pl
        if tot <= 0:
            continue
        out[alpha_positions[i]] = (pu / tot, pl / tot)
    return out

def _casehead_ctc_positions(sess, gray, aim_h, aim_w, chars, blank, text, force_align=False):
    """Per-position p(U)/p(l) from a CTC U/L head. None when alignment unusable."""
    logits = _casehead_input(sess, gray, aim_h, aim_w)
    if getattr(logits, "ndim", 0) == 3:
        logits = logits[0]
    shifted = np.exp(logits - float(logits.max()), dtype=np.float64)
    probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
    prev = blank
    steps = []
    for frame in probabilities:
        chosen = int(frame.argmax())
        if chosen != blank and chosen != prev:
            steps.append((chosen, frame))
        prev = chosen
    alpha_positions = [i for i, ch in enumerate(text) if ch.isalpha()]
    if len(steps) == len(text):
        slot_map = list(range(len(text)))
    elif len(steps) == len(alpha_positions):
        slot_map = alpha_positions
    else:
        _casehead_report("len_mismatch", n_text=len(text), n_alpha=len(alpha_positions),
                         n_pattern=len(steps))
        if force_align:
            forced = _casehead_ctc_forced(probabilities, alpha_positions, blank)
            if forced:
                return forced
        return None
    out = {}
    for si, (chosen, frame) in enumerate(steps):
        if chosen >= len(chars):
            continue
        pair_total = float(frame[0] + frame[1]) if len(chars) >= 2 else 0.0
        if pair_total <= 0:
            continue
        out[slot_map[si]] = (float(frame[0]) / pair_total, float(frame[1]) / pair_total)
    return out


def _casehead_reader_positions(sess, gray, aim_h, aim_w, text):
    """Forced-length Viterbi on the 63-class reader. None when alignment unusable."""
    logits = _casehead_input(sess, gray, aim_h, aim_w)
    if getattr(logits, "ndim", 0) == 3:
        logits = logits[0]
    lg = logits.astype(np.float64)
    lg = lg - lg.max(axis=1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
    T = logp.shape[0]
    N = len(text)
    if N == 0 or T < N:
        _casehead_report("reader_len_mismatch", n_text=N, n_frames=T)
        return None
    em = np.full((N, T), _READER_NEG)
    for i, ch in enumerate(text):
        cands = [c for c in (ch.upper(), ch.lower()) if c in _READER_INDEX]
        if not cands:
            _casehead_report("reader_unknown_char")
            return None
        em[i] = np.max(logp[:, [_READER_INDEX[c] for c in cands]], axis=1)
    neg = -1e18
    dp = np.full((N, T), neg)
    bk = np.zeros((N, T), dtype=np.int32)
    dp[0] = em[0]
    for i in range(1, N):
        prev = dp[i - 1]
        best = neg
        arg = 0
        row_bk = np.zeros(T, dtype=np.int32)
        for tt in range(T):
            if tt > 0 and prev[tt - 1] > best:
                best = prev[tt - 1]
                arg = tt - 1
            row_bk[tt] = arg
        cand = np.full(T, neg)
        cand[1:] = np.maximum.accumulate(prev)[:-1]
        dp[i] = em[i] + cand
        bk[i] = row_bk
    tbest = int(np.argmax(dp[-1]))
    if dp[-1][tbest] < -1e17:
        _casehead_report("reader_dp_fail")
        return None
    frames = [0] * N
    frames[N - 1] = tbest
    for i in range(N - 1, 0, -1):
        frames[i - 1] = int(bk[i][frames[i]])
    out = {}
    for i, ch in enumerate(text):
        if not ch.isalpha():
            continue
        pu = float(np.exp(logp[frames[i], _READER_INDEX[ch.upper()]]))
        pl = float(np.exp(logp[frames[i], _READER_INDEX[ch.lower()]]))
        tot = pu + pl
        if tot <= 0:
            continue
        out[i] = (pu / tot, pl / tot)
    return out


def _casehead_members(head):
    """A head is either one model, or a fusion of members with weights."""
    members = head.get("members")
    if members:
        return members
    return [{
        "model": head.get("model"),
        "align": head.get("align", "ctc_ul"),
        "weight": 1.0,
        "chars": head.get("chars"),
        "blank": head.get("blank"),
    }]


def case_pattern_fix(text, image_source, type_code=None):
    """大小写模式头（可选能力，路由里没有 case_pattern_heads 时完全无副作用）。

    只回答「每一位是上写还是下写」，逐位改大小写、不碰字母序列，因此对
    casefold 指标可证明零影响，只移动 exact。head 可以是单模型，也可以是
    members 融合（CTC U/L 头 + 63 类读者强制长度 Viterbi 对齐）。
    """
    if not text:
        return text, 0
    _ensure_metadata()
    heads = _metadata.get("case_pattern_heads") or []
    if not heads:
        return text, 0
    try:
        raw = _read_bytes(image_source)
        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            _casehead_report("decode_fail")
            return text, 0
        height, width = gray.shape[:2]
    except Exception:
        _casehead_report("read_fail")
        return text, 0
    key = str(int(type_code)) if type_code is not None else None
    base_dir = pathlib.Path(MODEL_PATH).resolve().parent
    for head in heads:
        when = head.get("when") or {}
        if when.get("width") is not None and int(when["width"]) != int(width):
            continue
        if when.get("height") is not None and int(when["height"]) != int(height):
            continue
        lum_lt = when.get("luminance_lt")
        lum_gt = when.get("luminance_gt")
        if lum_lt is not None or lum_gt is not None:
            mean_lum = float(gray.mean())
            if lum_lt is not None and not (mean_lum < float(lum_lt)):
                continue
            if lum_gt is not None and not (mean_lum > float(lum_gt)):
                continue
        types = head.get("types")
        if types and key not in [str(t) for t in types]:
            continue
        aim_h = int(head.get("height") or HEIGHT)
        aim_w = int(head.get("width") or WIDTH)
        gate = head.get("gate")
        fused = {}
        collected = []
        usable = True
        for member in _casehead_members(head):
            mpath = pathlib.Path(member.get("model", ""))
            if not mpath.is_absolute():
                mpath = base_dir / mpath
            if not mpath.exists():
                _casehead_report("missing_model", model=mpath.name)
                usable = False
                break
            sess = _casehead_session(mpath)
            mh = int(member.get("height") or aim_h)
            mw = int(member.get("width") or aim_w)
            align = member.get("align") or "ctc_ul"
            if align == "viterbi_forced":
                positions = _casehead_reader_positions(sess, gray, mh, mw, text)
            else:
                chars = member.get("chars") or head.get("chars") or "UL"
                blank = int(member.get("blank", head.get("blank", len(chars))))
                positions = _casehead_ctc_positions(sess, gray, aim_h, aim_w, chars, blank, text,
                                                   force_align=bool(member.get("force_align", head.get("force_align"))))
            if positions is None:
                if gate == "sample_conf":
                    continue
                usable = False
                break
            if gate == "sample_conf":
                confs = [max(v[0], v[1]) / max(1e-9, v[0] + v[1]) for v in positions.values()]
                collected.append((sum(confs) / max(1, len(confs)), positions, member))
                continue
            weight = float(member.get("weight", 1.0))
            only_below = member.get("only_below")
            for position, (pu, pl) in positions.items():
                slot = fused.setdefault(position, [0.0, 0.0])
                if only_below is not None:
                    tot0 = slot[0] + slot[1]
                    if tot0 > 0 and max(slot) / tot0 >= float(only_below):
                        continue
                slot[0] += weight * pu
                slot[1] += weight * pl
        picked = None
        if gate == "sample_conf":
            if collected:
                collected.sort(key=lambda item: -item[0])
                picked = collected[0][2]
                fused = {k: [v[0], v[1]] for k, v in collected[0][1].items()}
                _casehead_report("gate_pick", conf=round(collected[0][0], 4),
                                 n=len(collected))
            else:
                usable = False
        if not usable:
            continue
        src = picked if picked else head
        tau = float(src.get("min_position_confidence", head.get("min_position_confidence", 0.9)))
        tau_up = float(src.get("min_confidence_up", head.get("min_confidence_up", tau)))
        tau_lo = float(src.get("min_confidence_lo", head.get("min_confidence_lo", tau)))
        out = list(text)
        changed = 0
        for position, (pu, pl) in fused.items():
            if not (0 <= position < len(text)):
                continue
            current = text[position]
            if not current.isalpha():
                continue
            total = pu + pl
            if total <= 0:
                continue
            confidence = pu / total
            want_upper = confidence >= 0.5
            need = tau_up if want_upper else tau_lo
            if max(confidence, 1.0 - confidence) >= need and (current.isupper() != want_upper):
                out[position] = current.upper() if want_upper else current.lower()
                changed += 1
        if changed == 0:
            _casehead_report("no_change")
            continue
        candidate = "".join(out)
        if candidate.casefold() != text.casefold():
            _casehead_report("casefold_drift")
            return text, 0
        _casehead_report("fired", changed=changed)
        return candidate, changed
    return text, 0
