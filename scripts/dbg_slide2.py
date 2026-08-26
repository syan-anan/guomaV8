import os, sys, json, base64
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.utils import load_to_cv
import numpy as np, cv2

with open("H:/qinglong/syandaV8/data/yunda_shade.png", "rb") as f:
    bg_bytes = f.read()
with open("H:/qinglong/syandaV8/data/yunda_cutout.png", "rb") as f:
    gap_bytes = f.read()

bg_b64 = base64.b64encode(bg_bytes).decode()
gap_b64 = base64.b64encode(gap_bytes).decode()

bg = load_to_cv(bg_b64)
gap = load_to_cv(gap_b64)

print("bg type:", type(bg), "shape:", bg.shape if bg is not None else "None")
print("gap type:", type(gap), "shape:", gap.shape if gap is not None else "None")

bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)

print("bg_gray shape:", bg_gray.shape)
print("gap_gray shape:", gap_gray.shape)

# 直接模板匹配
res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
minv, maxv, minl, maxl = cv2.minMaxLoc(res)
print("tmpl match: max_val=", round(maxv, 4), "max_loc=", maxl)

# 检查gap图片是否有alpha通道
print("gap channels:", gap.shape[2] if len(gap.shape) > 2 else "gray")
