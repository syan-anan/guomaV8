import os, sys, json, time, base64, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.slide import detect_gap, detect_gap_multiscale
import numpy as np, cv2

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 MicroMessenger/7.0.20.1781(0x6700143B) NetType/WIFI MiniProgramEnv/Windows"
REFERER = "https://servicewechat.com/wxdeb5309aa3e93fd1/695/page-frame.html"
BASE = "https://mbpxapi.yundasys.com:38861/gateway/interface"
ACCOUNT = "oPJUI0diLY2l-xkwBOCTxXj54fd0"
APPID = "wjvxmno358lze827"

c = httpx.Client(timeout=20, verify=False, headers={
    "User-Agent": UA, "Referer": REFERER, "Content-Type": "application/json"
})

now = int(time.time() * 1000)
payload = {
    "version": "V1.0", "action": "ydmbaccount.ydaccount.getImageVerifyCode",
    "data": {"client": "mobile", "slideImageWidth": 318, "type": "slide",
             "accountId": ACCOUNT, "accountSrc": "wxapp", "reqTime": now},
    "appid": APPID, "req_time": now, "options": False,
}
r = c.post(BASE + "?ydmbaccount.ydaccount.getImageVerifyCode", json=payload)
data = r.json()
body = data.get("body", {})
if isinstance(body, str):
    body = json.loads(body)
dd = body.get("data", {})

shade = base64.b64decode(dd["shadeImage"])
cutout = base64.b64decode(dd["cutoutImage"]) if dd.get("cutoutImage") else None

# 保存新图片
os.makedirs("H:/qinglong/syandaV8/data", exist_ok=True)
with open("H:/qinglong/syandaV8/data/yunda_live_shade.png", "wb") as f:
    f.write(shade)
with open("H:/qinglong/syandaV8/data/yunda_live_cutout.png", "wb") as f:
    f.write(cutout)

# 手动检测
from solver.utils import load_to_cv
shade_b64 = base64.b64encode(shade).decode()
cutout_b64 = base64.b64encode(cutout).decode()
bg = load_to_cv(shade_b64)
gap = load_to_cv(cutout_b64)
print("新图片尺寸:", bg.shape[1], "x", bg.shape[0], "滑块:", gap.shape[1], "x", gap.shape[0])

# 模板匹配
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
_, maxv, _, maxl = cv2.minMaxLoc(res)
print("模板匹配: max_val=", round(maxv, 4), "max_loc=", maxl)

# 引擎检测
result = detect_gap_multiscale(shade_b64, cutout_b64)
print("引擎检测:", result)
