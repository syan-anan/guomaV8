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

# 模拟 detect_gap 内部逻辑
bg = load_to_cv(bg_b64)
print("bg loaded:", bg.shape)
gap = load_to_cv(gap_b64)
print("gap loaded:", gap.shape)
print("gap > 20:", gap.shape[0] > 20, gap.shape[1] > 20)

if gap is not None and gap.shape[0] > 20 and gap.shape[1] > 20:
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
    print("channels:", gap.shape[2] if len(gap.shape) > 2 else "gray")
    if gap.shape[2] == 4:
        print("has alpha")
        alpha = gap[:, :, 3]
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        print("no alpha - standard tmpl")
        res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print("max_val:", round(max_val, 4), "max_loc:", max_loc)
    if max_val >= 0.3:
        offset = gap_gray.shape[1] // 6 if gap_gray.shape[1] <= 40 else gap_gray.shape[1] // 4
        dist = max_loc[0] - offset
        print("tmpl method triggered! distance:", round(dist), "offset:", offset)
    else:
        print("tmpl threshold not met")
else:
    print("gap condition failed")
