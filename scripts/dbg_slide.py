import os, sys, base64, numpy as np, cv2
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.utils import load_to_cv

with open("H:/qinglong/syandaV8/data/yunda_shade.png", "rb") as f:
    bg_bytes = f.read()
with open("H:/qinglong/syandaV8/data/yunda_cutout.png", "rb") as f:
    gap_bytes = f.read()

bg = load_to_cv(bg_bytes)
gap = load_to_cv(gap_bytes)

bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)

print("背景:", bg_gray.shape, "滑块:", gap_gray.shape)

# 直接模板匹配
res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
minv, maxv, minl, maxl = cv2.minMaxLoc(res)
print("模板匹配 max_val:", round(maxv, 4), "max_loc:", maxl)
# 缺口中心x
if maxv >= 0.4:
    cx = maxl[0] + gap_gray.shape[1] // 2
    print("缺口中心x(模板匹配):", cx)

# 尝试用边缘增强的模板匹配
import numpy as np, cv2

def tmpl_match(bg, gap, vis_gap):
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
    # 用 alpha 通道
    if gap.shape[2] == 4:
        alpha = gap[:, :, 3]
        print("alpha 通道存在, 均值:", alpha.mean())
        # 掩码
        mask = alpha
        res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxl = cv2.minMaxLoc(res)
    return maxv, maxl

# 打印滑块图通道
print("滑块通道数:", gap.shape[2])
