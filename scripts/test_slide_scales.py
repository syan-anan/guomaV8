import os, sys, json, base64, io
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.utils import load_to_cv
from solver.slide import detect_gap
import numpy as np, cv2

with open("H:/qinglong/syandaV8/data/yunda_shade.png", "rb") as f:
    bg_bytes = f.read()
with open("H:/qinglong/syandaV8/data/yunda_cutout.png", "rb") as f:
    gap_bytes = f.read()

bg_b64 = base64.b64encode(bg_bytes).decode()
gap_b64 = base64.b64encode(gap_bytes).decode()

bg = load_to_cv(bg_b64)
gap = load_to_cv(gap_b64)
print("背景图尺寸:", bg.shape[1], "x", bg.shape[0])
print("滑块图尺寸:", gap.shape[1], "x", gap.shape[0])
print("用户提交的 slideImageWidth: 318")
print("预期距离: 193")

# 用不同scale测试模板匹配
for scale in [1.0, 0.9, 0.8, 1.1, 1.2]:
    result = detect_gap(bg_b64, gap_b64, scale=scale)
    print(f"scale={scale}: {json.dumps(result)}")
