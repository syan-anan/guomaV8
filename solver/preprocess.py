# -*- coding: utf-8 -*-
"""统一图片预处理流水线.

为 OCR / 滑动 / 点选 / 空间推理等所有题型提供一致的图片加载与预处理接口。
设计原则：
- 输入兼容：base64 字符串 / bytes / 本地路径 / cv2 数组 / PIL.Image
- 输出稳定：BGR 格式 cv2 数组（与 syandaV8 / OpenCV 对齐）
- 可配置：通过 pipeline 字典组合预处理步骤，便于复现和调优
"""
import io, os, base64
import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# 1. 统一加载
# ---------------------------------------------------------------------------
def load(src, color_mode="bgr"):
    """把任意输入转为 cv2 数组。

    Args:
        src: base64 字符串 / bytes / 本地路径 / cv2 数组 / PIL.Image
        color_mode: 'bgr', 'rgb', 'gray', 'unchanged'
    """
    if isinstance(src, np.ndarray):
        img = src
    elif isinstance(src, Image.Image):
        img = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    elif isinstance(src, (str, bytes)):
        img = _load_from_str_or_bytes(src)
    else:
        raise TypeError(f"不支持的图片输入类型: {type(src)}")

    if img is None:
        raise ValueError("图片解码失败")
    return _ensure_color_mode(img, color_mode)


def _load_from_str_or_bytes(src):
    if isinstance(src, str):
        if os.path.isfile(src):
            data = open(src, "rb").read()
        else:
            data = _b64_to_bytes(src)
    else:
        data = src
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _b64_to_bytes(b64):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    b64 = b64.strip().replace(" ", "+")
    return base64.b64decode(b64)


def _ensure_color_mode(img, mode):
    if mode == "unchanged":
        return img
    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if mode == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # bgr
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return img
    raise ValueError(f"未知通道数: {img.shape}")


# ---------------------------------------------------------------------------
# 2. 基础变换
# ---------------------------------------------------------------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def resize(img, width=None, height=None, fx=None, fy=None, interpolation=cv2.INTER_CUBIC):
    if width is not None or height is not None:
        if width is None:
            width = int(img.shape[1] * height / img.shape[0])
        if height is None:
            height = int(img.shape[0] * width / img.shape[1])
        return cv2.resize(img, (width, height), interpolation=interpolation)
    if fx is not None or fy is not None:
        return cv2.resize(img, None, fx=fx or 1, fy=fy or 1, interpolation=interpolation)
    return img


def crop(img, x, y, w, h):
    return img[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# 3. 增强 / 去噪 / 二值化
# ---------------------------------------------------------------------------
def denoise(img, strength=10):
    gray = to_gray(img)
    denoised = cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
    return to_bgr(denoised)


def normalize_contrast(img, clip_limit=3.0, grid=(8, 8)):
    gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return to_bgr(clahe.apply(gray))


def sharpen(img, amount=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


def binarize(img, method="otsu", block=31, c=12, thresh=128):
    gray = to_gray(img)
    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)
    elif method == "binary":
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    elif method == "inv":
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(f"未知二值化方法: {method}")
    return to_bgr(bw)


def invert(img):
    return cv2.bitwise_not(img)


# ---------------------------------------------------------------------------
# 4. 常用流水线
# ---------------------------------------------------------------------------
def pipeline(src, steps):
    """按步骤依次执行预处理。

    Args:
        src: 任意支持的输入
        steps: list[dict]，例如：
            [{"op": "load", "mode": "bgr"},
             {"op": "resize", "fx": 2, "fy": 2},
             {"op": "denoise", "strength": 10},
             {"op": "binarize", "method": "otsu"}]
    """
    result = None
    for step in steps:
        op = step.get("op")
        if op == "load":
            result = load(src, step.get("mode", "bgr"))
        elif op == "to_bgr":
            result = to_bgr(result)
        elif op == "to_gray":
            result = to_gray(result)
        elif op == "to_rgb":
            result = to_rgb(result)
        elif op == "resize":
            result = resize(result, width=step.get("width"), height=step.get("height"),
                            fx=step.get("fx"), fy=step.get("fy"), interpolation=step.get("interpolation", cv2.INTER_CUBIC))
        elif op == "denoise":
            result = denoise(result, step.get("strength", 10))
        elif op == "normalize_contrast":
            result = normalize_contrast(result, step.get("clip_limit", 3.0), step.get("grid", (8, 8)))
        elif op == "sharpen":
            result = sharpen(result, step.get("amount", 1.0))
        elif op == "binarize":
            result = binarize(result, step.get("method", "otsu"), step.get("block", 31), step.get("c", 12), step.get("thresh", 128))
        elif op == "invert":
            result = invert(result)
        elif op == "crop":
            result = crop(result, step["x"], step["y"], step["w"], step["h"])
        else:
            raise ValueError(f"未知预处理操作: {op}")
    return result


# ---------------------------------------------------------------------------
# 5. 兼容旧接口
# ---------------------------------------------------------------------------
def b64_to_bytes(b64):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    b64 = b64.strip().replace(" ", "+")
    return base64.b64decode(b64)


def bytes_to_cv(b):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败")
    return img


def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def load_to_cv(src):
    return load(src, "bgr")


def grayscale_binarize(img, adaptive=True):
    gray = to_gray(img)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if adaptive:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


# ---------------------------------------------------------------------------
# 6. 常用题型预设
# ---------------------------------------------------------------------------
def preprocess_for_ocr(src):
    """OCR 题型：返回多种预处理变体，供投票使用。"""
    img = load(src, "bgr")
    return [("orig", to_bgr(img))] + _ocr_variants(to_gray(img))


def preprocess_for_ocr_type(src, type_code):
    """按 1001/1002/1003 题型返回定制化预处理变体。"""
    img = load(src, "bgr")
    gray = to_gray(img)
    variants = [("orig", to_bgr(img))]

    if type_code == 1002:
         # 数字：强调笔画分离、去粘连、二值化
         variants.extend(_ocr_variants(gray, adapt_block=21, denoise_strength=15, sharpen=True, morphology=True))
    elif type_code == 1003:
         # 字母：平滑+锐化，减少断裂
         variants.extend(_ocr_variants(gray, adapt_block=31, denoise_strength=10, sharpen=True, morphology=False))
    else:
         # 1001 英数混合：平衡策略
         variants.extend(_ocr_variants(gray, adapt_block=25, denoise_strength=12, sharpen=True, morphology=False))
    return variants


def _ocr_variants(gray, adapt_block=31, denoise_strength=10, sharpen=False, morphology=False):
    """生成 OCR 预处理变体列表。"""
    variants = []
    up2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("up2", to_bgr(up2)))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu2 = cv2.resize(otsu, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("otsu2x", to_bgr(otsu2)))

    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, adapt_block, 12)
    variants.append(("adapt", to_bgr(adapt)))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", to_bgr(clahe)))

    den = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)
    den2 = cv2.resize(den, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("den2x", to_bgr(den2)))

    inv = cv2.bitwise_not(gray)
    inv2 = cv2.resize(inv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("inv2x", to_bgr(inv2)))

    if sharpen:
         sharpened = cv2.filter2D(up2, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
         variants.append(("sharp2x", to_bgr(sharpened)))

    if morphology:
         # 开运算：去噪 + 分离粘连
         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
         opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
         opened2 = cv2.resize(opened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
         variants.append(("morph_open2x", to_bgr(opened2)))

    return variants


def preprocess_for_slide(src):
    """滑动题型：标准化加载并返回 BGR。"""
    return load(src, "bgr")


def preprocess_for_click(src):
    """点选题型：标准化加载并返回 BGR。"""
    return load(src, "bgr")
