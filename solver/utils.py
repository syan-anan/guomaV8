# -*- coding: utf-8 -*-
"""共享小工具：图像解码/预处理/Base64。
现在预处理逻辑已迁移到 solver.preprocess，本文件保留兼容入口。"""
from solver.preprocess import (
    b64_to_bytes,
    bytes_to_cv,
    cv_to_pil,
    load_to_cv,
    grayscale_binarize,
)
