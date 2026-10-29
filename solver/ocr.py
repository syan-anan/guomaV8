"""OCR engine: char-level voting + 7 high-quality preprocessing variants. (optimized: fast-path + full-vote fallback)"""
import io, re, time
import numpy as np
import cv2
import ddddocr
from collections import Counter
from solver.utils import b64_to_bytes as _b64
from solver.cnn import cnn_ocr, merge_engines
from solver.crnn_ocr import crnn_ocr
from solver import crnn63_ocr as crnn63_ocr_module
from solver.crnn63_ocr import crnn63_ocr
CNN_ENABLED = True

_models = {}
def _get_model(name):
    if name not in _models:
        if name == "default":
            _models[name] = ddddocr.DdddOcr(show_ad=False)
        else:
            _models[name] = ddddocr.DdddOcr(beta=True, show_ad=False)
    return _models[name]

def preload_ocr_models():
    """预热 OCR 模型，避免首次请求冷启动。"""
    for name in ("default", "beta"):
        _get_model(name)

# 快速路径置信度阈值：>= 此值直接返回（45 样本实测 0.95 全对）
FAST_CONF = 0.95
# 各题型快速路径首选模型：1002 数字 default即可；1001/1003 用 beta（更强）
_PRIMARY = {1001: "beta", 1002: "default", 1003: "beta"}
# 各题型快速路径首选模型：1002 数字 default 即可；1001/1003 用 beta（更强）
_FAST_MODEL = {1001: "beta", 1002: "default", 1003: "beta"}

def _bytes_to_cv(b):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None: raise ValueError("bad image")
    return img

def _encode(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def _g2b(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    v = [("orig", img)]
    up2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    v.append(("up2", _g2b(up2)))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ots2 = cv2.resize(otsu, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    v.append(("otsu2x", _g2b(ots2)))
    a = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    v.append(("adapt", _g2b(a)))
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    v.append(("clahe", _g2b(c)))
    d = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    d2 = cv2.resize(d, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    v.append(("den2x", _g2b(d2)))
    inv = cv2.bitwise_not(gray)
    inv2 = cv2.resize(inv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    v.append(("inv2x", _g2b(inv2)))
    return v

def _classify(img_bytes, model_name):
    try:
        return _get_model(model_name).classification(img_bytes)
    except:
        return ""

def _classify_prob(img_bytes, model_name):
    """一次带概率的分类，返回 (text, confidence)。失败返回 ('', 0.0)"""
    try:
        r = _get_model(model_name).classification(img_bytes, probability=True)
        if isinstance(r, dict):
            return r.get("text", ""), float(r.get("confidence", 0.0) or 0.0)
        return r, 0.0
    except Exception:
        return "", 0.0

def _clean_seq(s):
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _clean_case_sensitive(s, type_code):
    """Case-preserving cleaner used for dark-background captchas, whose answers are
    case-sensitive. The legacy pipeline uppercased everything, which makes any answer
    containing lowercase letters impossible to match.

    The digit/letter confusion maps are the same domain priors the legacy _post_fix
    applied (a 1003 answer is always letters, a 1002 answer always digits), but they no
    longer destroy the case of the characters that were already correct."""
    if type_code == 1002:
        return re.sub(r"[^0-9]", "", s.translate(_DIGIT_LETTER_MAP))
    if type_code == 1003:
        return re.sub(r"[^A-Za-z]", "", s.translate(_LETTER_DIGIT_MAP))
    return re.sub(r"[^A-Za-z0-9]", "", s)

def _char_vote(results, min_len=2):
    valid = []
    for r in results:
        c = _clean_seq(r)
        if len(c) >= min_len:
            valid.append(c)
    if not valid:
        return "NONE", 0.0, 0
    max_len = max(len(x) for x in valid)
    grp = [x for x in valid if len(x) == max_len]
    chars = []
    agrees = 0
    for pos in range(max_len):
        cols = [x[pos] for x in grp]
        ch, nv = Counter(cols).most_common(1)[0]
        chars.append(ch)
        agrees += nv
    series = "".join(chars)
    ratio = agrees / (max_len * len(grp)) if grp else 0.0
    return series, round(ratio, 4), len(valid)

_LETTER_DIGIT_MAP = str.maketrans({"0":"O","1":"I","2":"Z","5":"S","8":"B"})
_DIGIT_LETTER_MAP = str.maketrans({"O":"0","I":"1","S":"5","Z":"2","B":"8","L":"1","l":"1","o":"0"})
ARITHMETIC_RE = re.compile(r"^\s*(\d+)\s*([+\-x×*/÷])\s*(\d+)\s*(?:=|＝).*$", re.IGNORECASE)

def _post_fix(text, type_code):
    if not text or text == "NONE": return text
    if type_code == 1002:
        return re.sub(r"[^0-9]", "", text.translate(_DIGIT_LETTER_MAP))
    if type_code == 1003:
        return re.sub(r"[^A-Z]", "", text.translate(_LETTER_DIGIT_MAP))
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()

def _decode_src(img_src):
    if isinstance(img_src, str):
        if len(img_src) > 200 or "," in img_src:
            b = _b64(img_src)
        else:
            with open(img_src, "rb") as f: b = f.read()
    elif isinstance(img_src, bytes):
        b = img_src
    else:
        b = img_src.read()
    return _bytes_to_cv(b)

def _src_bytes(img_src):
    """原始编码字节，未经 cv2 解码-重编码循环。"""
    if isinstance(img_src, str):
        if len(img_src) > 200 or "," in img_src:
            return _b64(img_src)
        with open(img_src, "rb") as f:
            return f.read()
    if isinstance(img_src, bytes):
        return img_src
    return img_src.read()

def fuse_crnn(dd, dc, ct, cc):
    """融合规则：R_len_conf（最优规则 63.9%）"""
    dd = (dd or "").strip().upper()
    ct = (ct or "").strip().upper()
    if not ct:
        return dd, dc, "dd"
    if not dd:
        return ct, cc, "crnn"
    Ld, Lc = len(dd), len(ct)
    if Ld == Lc:
        if cc >= dc + 0.05:
            return ct, cc, "crnn"
        return dd, dc, "dd"
    if Lc > Ld and cc >= 0.5:
        return ct, cc, "crnn"
    if Ld > Lc and dc >= 0.5:
        return dd, dc, "dd"
    return dd, dc, "dd"


def _arithmetic_answer(text):
    """Evaluate simple OCR-visible arithmetic like 9x4=? or 12+3=?."""
    match = ARITHMETIC_RE.match(str(text or "").replace("−", "-").replace("＝", "="))
    if not match:
        return None
    left, operator, right = int(match.group(1)), match.group(2).lower(), int(match.group(3))
    try:
        if operator == "+":
            value = left + right
        elif operator == "-":
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

def _full_vote(img, type_code, extra_results=None):
    """现有全量逻辑：7 变体 x 2 模型投票。extra_results 可追加额外识别串。"""
    versions = _preprocess(img)
    all_results = []
    if extra_results:
        all_results.extend(extra_results)
    for _vname, vimg in versions:
        vbytes = _encode(vimg)
        for mname in ("default", "beta"):
            txt = _classify(vbytes, mname)
            if txt and _clean_seq(txt):
                all_results.append(txt)
    if not all_results:
        return {"code": -3, "error": "recog failed", "text": "", "confidence": 0.0}
    best, conf, votes = _char_vote(all_results)
    clean = _post_fix(best, type_code)
    if not clean: clean = best
    return {"code": 0, "text": clean, "raw": best, "confidence": conf, "votes": votes, "agreement": votes}

def _finalize(result, t0, img_bytes):
    if result.get("text") and result.get("engine") != "crnn63":
        _ct, _cc = crnn_ocr(img_bytes)
        if _ct:
            _mt, _mc, _srcn = fuse_crnn(result.get("text"), result.get("confidence", 0.0), _ct, _cc)
            if _srcn == "crnn":
                result["text"] = _mt
                result["confidence"] = round(_mc, 4)
            result["engine"] = "dd+crnn"
    result["cost_ms"] = round((time.perf_counter()-t0)*1000, 1)
    return result

def _apply_char_fix(result, original_bytes, type_code):
    """同形字改判后处理。路由里没有 char_fix_branches 时是纯 no-op。"""
    try:
        text = result.get("text") or ""
        if not text:
            return result
        fixed, changed = crnn63_ocr_module.char_fix(text, original_bytes, type_code)
        if changed and fixed:
            result["char_fix"] = changed
            result["text"] = fixed
            result["raw"] = fixed
            result["engine"] = (str(result.get("engine") or "") + "+char_fix").lstrip("+")
    except Exception:
        pass
    return result

def _apply_case_pattern_heads(result, original_bytes, type_code):
    """大小写模式头后处理。路由里没有 case_pattern_heads 时是纯 no-op。"""
    try:
        text = result.get("text") or ""
        if not text:
            return result
        fixed, changed = crnn63_ocr_module.case_pattern_fix(text, original_bytes, type_code)
        if changed and fixed:
            result["case_pattern_fix"] = changed
            result["text"] = fixed
            result["raw"] = fixed
            result["engine"] = (str(result.get("engine") or "") + "+casepat").lstrip("+")
    except Exception:
        pass
    return result


def _apply_case_arbiter(result, original_bytes, type_code):
    """casefold 中性大小写仲裁（见 crnn63_ocr._case_arbitrate）。只在路由声明 case_arbiters 时生效。"""
    try:
        return crnn63_ocr_module._case_arbitrate(original_bytes, result, type_code)
    except Exception:
        return result



def _finish_fast(result, t0, img_bytes, original_bytes, type_code, routed):
    """Fast dark path: same answer as the full pipeline, minus the engines that cannot
    change it.  _use_crnn63 already forces the CRNN63 takeover whenever dark_input is
    true, so the ddddocr engines are pure cost on dark images.

    Route keys (absent -> this function is never reached):
      dark_fast_path  enable the path for this type
      fast_skip_case  skip the case-only post heads (casefold-neutral, exact-only)
    """
    result = _apply_char_fix(result, original_bytes, type_code)
    result["cost_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if not routed.get("fast_skip_case"):
        result = _apply_case_arbiter(result, original_bytes, type_code)
        result = _apply_case_pattern_heads(result, original_bytes, type_code)
    return result

def _finish(result, t0, img_bytes, original_bytes, type_code):
    return _apply_case_pattern_heads(_apply_case_arbiter(_finalize(_apply_char_fix(_use_crnn63(result, original_bytes, type_code),
                                     original_bytes, type_code), t0, img_bytes), original_bytes, type_code),
                                   original_bytes, type_code)

def _use_crnn63(result, img_bytes, type_code):
    """给已识别结果叠加 case-sensitive CRNN63。

    该模型在 1001-1003 灰度样本上明显更强；置信度高时直接采用，
    中等置信度仅在主链路较弱时接管，避免低分结果破坏高确定性样本。
    """
    if result.get("code") != 0:
        return result
    try:
        raw, conf = crnn63_ocr(img_bytes, type_code)
        clean = _post_fix(_clean_seq(raw), type_code)
        if not raw or not clean:
            return result
        cur_conf = float(result.get("confidence") or 0.0)
        routed = crnn63_ocr_module._metadata.get("type_models", {}).get(str(type_code), {})
        prefer_full = bool(routed.get("ocr_prefer_full"))
        # Dark-background captchas: the ddddocr engines are effectively blind here, so once a
        # route declares a dark specialist path the CRNN63 answer must win unconditionally.
        dark_input = False
        if (routed.get("dark_ensemble") or []) and routed.get("dark_threshold") is not None:
            try:
                dark_tensor = crnn63_ocr_module._preprocess(img_bytes)
                dark_input = float(np.asarray(dark_tensor).mean()) * 255.0 < float(routed["dark_threshold"])
            except Exception:
                dark_input = False
        switch_max_length = routed.get("switch_max_length")
        forced_fallback = switch_max_length is not None and len(clean) <= int(switch_max_length)
        # Route-scoped arithmetic guard (type_models.<type>.arithmetic_guard).
        # A CRNN retrained on digit-only corpora can become over-confident on
        # arithmetic captchas that were routed to the same type, which would
        # suppress the ddddocr arithmetic rescue below.  When the route opts in,
        # read the raw image with the beta engine first and prefer a valid
        # arithmetic answer; a plain digit captcha yields no operator, so the
        # guard is a no-op there.
        if routed.get("arithmetic_guard") and int(type_code) == 1002:
            try:
                guard_text = _classify(img_bytes, "beta")
                guard_answer = _arithmetic_answer(guard_text)
            except Exception:
                guard_text, guard_answer = "", None
            if guard_answer is not None:
                result["raw"] = guard_text
                result["text"] = guard_answer
                result["confidence"] = round(float(conf), 4)
                result["engine"] = "crnn63+arithmetic_guard"
                return result
        take = conf >= 0.90 or (conf >= 0.70 and cur_conf < 0.70)
        if dark_input:
            take = True
        if prefer_full:
            take = True
        if forced_fallback:
            take = False
        if not take and len(clean) == len(str(result.get("text") or "")) and conf > cur_conf + 0.05:
            take = True
        # Route-scoped engine-agreement guard (D6.45).  When BOTH ddddocr engines agree on the
        # same string, that mutual agreement outweighs a single CRNN63 takeover: forensics in
        # reports/d645_engine_override_forensics.json found 20+ honest-set images where the
        # default and beta engines were both correct while CRNN63 (conf ~1.0) overrode them.
        # Absent from a route means no change, so every pre-existing route keeps its behaviour.
        if take and routed.get("dd_agreement_guard") and (not dark_input or routed.get("dd_agreement_guard_dark")):
            try:
                guard_text = _classify(img_bytes, "beta")
            except Exception:
                guard_text = ""
            dd_text = str(result.get("text") or result.get("raw") or "")
            if guard_text and dd_text and guard_text.casefold() == dd_text.casefold() \
                    and clean.casefold() != guard_text.casefold():
                if (not routed.get("dd_agreement_guard_len")) or len(clean) == len(dd_text):
                    take = False
        if take:
            result["raw"] = raw
            if dark_input:
                case_sensitive = _clean_case_sensitive(raw, type_code)
                result["text"] = case_sensitive or clean
            else:
                result["text"] = clean
            result["confidence"] = round(float(conf), 4)
            result["engine"] = "crnn63"
            if type_code == 1002:
                arithmetic = _arithmetic_answer(raw)
                if arithmetic is not None:
                    result["text"] = arithmetic
                    result["engine"] = "crnn63+arithmetic"
            if type_code == 1001 and conf < 0.65 and not dark_input:
                rescued = crnn63_ocr_module._fallback_text(img_bytes, type_code)
                if len(rescued) >= 2:
                    result["text"] = rescued
                    result["raw"] = rescued
                    result["engine"] = "crnn63+ddddocr"
        else:
            result["engine"] = "dd"
            if type_code == 1002:
                evidence = f"{result.get('raw','')}{result.get('text','')}"
                if any(not char.isdigit() for char in evidence):
                    beta_text = _get_model("beta").classification(img_bytes)
                    arithmetic = _arithmetic_answer(beta_text)
                    if arithmetic is not None:
                        result["text"] = arithmetic
                        result["raw"] = beta_text
                        result["engine"] = "ddddocr+arithmetic"
    except Exception:
        pass
    return result

def solve_ocr(img_src, type_code=1001):
    t0 = time.perf_counter()
    try:
        img = _decode_src(img_src)
    except Exception as e:
        return {"code": -2, "error": "pic load fail: " + str(e), "text": "", "confidence": 0.0}

    # ---- 渐进式快速路径：1次推理(按题型选模型) -> 不足补 2次 -> 再不足全量投票 ----
    img_bytes = _encode(img)
    try:
        original_bytes = _src_bytes(img_src)
    except Exception:
        original_bytes = img_bytes
    if int(type_code) == 1002:
        try:
            from solver.t2_1002 import read_new_style_tta as _t2c_read
            _t2c_text, _t2c_conf = _t2c_read(original_bytes, crnn63_ocr_module.MODEL_PATH.parent)
            _t2c_clean = "".join(c for c in _t2c_text if c.isdigit())
            if 4 <= len(_t2c_clean) <= 8:
                return {"code": 0, "text": _t2c_clean, "raw": _t2c_text,
                        "confidence": round(float(_t2c_conf), 4), "votes": 1,
                        "agreement": 1, "engine": "t2_1002",
                        "cost_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception:
            pass
    domain = crnn63_ocr_module.domain_vote(original_bytes, type_code)
    if domain:
        dtext, dconf = domain
        domain_result = {"code": 0, "text": dtext, "raw": dtext, "confidence": round(float(dconf), 4),
                         "votes": 1, "agreement": 1, "fast": True, "engine": "domain_vote"}
        _tm = (crnn63_ocr_module._metadata.get("type_models") or {})
        _routed = _tm.get(str(int(type_code))) or {}
        if _routed.get("fast_skip_case"):
            return _finish_fast(domain_result, t0, img_bytes, original_bytes, type_code, _routed)
        return _apply_case_pattern_heads(_apply_case_arbiter(_finalize(_apply_char_fix(domain_result, original_bytes, type_code), t0, img_bytes),
                                           original_bytes, type_code), original_bytes, type_code)
    routed_tm = (crnn63_ocr_module._metadata.get("type_models") or {})
    routed = routed_tm.get(str(int(type_code))) or {}
    if (routed.get("dark_fast_path") and (routed.get("dark_ensemble") or [])
            and routed.get("dark_threshold") is not None
            and not routed.get("switch_max_length") and not routed.get("dd_agreement_guard")
            and not routed.get("arithmetic_guard")):
        try:
            _probe = crnn63_ocr_module._preprocess(original_bytes)
            _dark = float(np.asarray(_probe).mean()) * 255.0 < float(routed["dark_threshold"])
        except Exception:
            _dark = False
        if _dark:
            try:
                _raw, _conf = crnn63_ocr_module.crnn63_ocr(original_bytes, int(type_code))
            except Exception:
                _raw, _conf = "", 0.0
            _clean = _post_fix(_clean_seq(_raw), type_code) if _raw else ""
            if _raw and _clean:
                _cs = _clean_case_sensitive(_raw, type_code)
                if routed.get("fast_skip_case") and int(type_code) == 1003:
                    _cs = ""  # case head skipped: report the uppercase form
                _res = {"code": 0, "text": _cs or _clean, "raw": _raw,
                        "confidence": round(float(_conf), 4),
                        "votes": 1, "agreement": 1, "fast": True, "engine": "crnn63"}
                return _finish_fast(_res, t0, img_bytes, original_bytes, type_code, routed)
    first = _PRIMARY.get(type_code, "beta")
    second = "beta" if first == "default" else "default"

    txt1, conf1 = _classify_prob(img_bytes, first)
    if txt1 and conf1 > 0.0:
        clean = _post_fix(_clean_seq(txt1), type_code) if txt1 else ""
        if clean and len(clean) >= 2 and conf1 >= FAST_CONF:
            if len(clean) >= 5 and CNN_ENABLED:
                _ct, _cc = cnn_ocr(img_bytes)
                if _ct:
                    clean, _src, conf1 = merge_engines(clean, conf1, _ct, _cc, type_code)
            result = {"code": 0, "text": clean, "raw": txt1, "confidence": round(conf1, 4),
                    "votes": 1, "agreement": 1, "fast": True}
            return _finish(result, t0, img_bytes, original_bytes, type_code)

    txt2, conf2 = _classify_prob(img_bytes, second)
    cands = [(conf1, first, txt1), (conf2, second, txt2)]
    cands = [x for x in cands if x[2] and x[0] > 0.0]
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        conf, mname, txt = cands[0]
        clean = _post_fix(_clean_seq(txt), type_code) if txt else ""
        if clean and len(clean) >= 2 and conf >= FAST_CONF:
            if len(clean) >= 5 and CNN_ENABLED:
                _ct, _cc = cnn_ocr(img_bytes)
                if _ct:
                    clean, _src2, conf = merge_engines(clean, conf, _ct, _cc, type_code)
            result = {"code": 0, "text": clean, "raw": txt, "confidence": round(conf, 4),
                    "votes": 2, "agreement": 2, "fast": True}
            return _finish(result, t0, img_bytes, original_bytes, type_code)

    # ---- 兜底：全量投票（保留原逻辑），fast 两次结果一并参与投票 ----
    extra = [t for _, _, t in cands]


    result = _full_vote(img, type_code, extra_results=(extra or None))
    result["fast"] = False
    result["cost_ms"] = round((time.perf_counter()-t0)*1000, 1)
    return _finish(result, t0, img_bytes, original_bytes, type_code)

