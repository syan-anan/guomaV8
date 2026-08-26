import os, sys, json, time, base64, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
import numpy as np, cv2
from solver.utils import load_to_cv

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ..."
BASE = "https://mbpxapi.yundasys.com:38861/gateway/interface"
ACCOUNT = "oPJUI0diLY2l-xkwBOCTxXj54fd0"
APPID = "wjvxmno358lze827"

c = httpx.Client(timeout=20, verify=False, headers={"User-Agent": UA, "Content-Type": "application/json"})

def submit(dist, y, flag):
    now2 = int(time.time() * 1000)
    p = {"version": "V1.0", "action": "ydmbintegral.ydintegral.obtain.event.integral",
         "data": {"channelId": "wxapp", "itgType": "browse", "imageCode": f"{dist}|{y}",
                  "flag": flag, "accountId": ACCOUNT, "accountSrc": "wxapp", "reqTime": now2},
         "appid": APPID, "req_time": now2, "options": False}
    rr = c.post(BASE + "?ydmbintegral.ydintegral.obtain.event.integral", json=p)
    return rr.json()

# 请求验证码
now = int(time.time() * 1000)
payload = {"version": "V1.0", "action": "ydmbaccount.ydaccount.getImageVerifyCode",
    "data": {"client": "mobile", "slideImageWidth": 318, "type": "slide",
             "accountId": ACCOUNT, "accountSrc": "wxapp", "reqTime": now},
    "appid": APPID, "req_time": now, "options": False}
r = c.post(BASE + "?ydmbaccount.ydaccount.getImageVerifyCode", json=payload)
data = r.json()
body = data.get("body", {})
if isinstance(body, str): body = json.loads(body)
dd = body.get("data", {})
flag = dd["flag"]
shade_b64 = dd["shadeImage"]
cutout_b64 = dd.get("cutoutImage", "")
print("flag:", flag)

bg = load_to_cv(shade_b64)
gap = load_to_cv(cutout_b64)
print("bg:", bg.shape, "gap:", gap.shape, "gap channels:", gap.shape[2] if len(gap.shape)>2 else 1)

bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)

# 方法1: 标准模板匹配
res1 = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
_, v1, _, l1 = cv2.minMaxLoc(res1)
print("方法1 CCOEFF_NORMED:", round(v1,4), l1)

# 方法2: Canny边缘模板匹配
bg_edge = cv2.Canny(bg_gray, 50, 150)
gap_edge = cv2.Canny(gap_gray, 50, 150)
res2 = cv2.matchTemplate(bg_edge, gap_edge, cv2.TM_CCOEFF_NORMED)
_, v2, _, l2 = cv2.minMaxLoc(res2)
print("方法2 Canny边缘:", round(v2,4), l2)

# 方法3: TM_SQDIFF_NORMED
res3 = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_SQDIFF_NORMED)
mv3, v3, ml3, l3 = cv2.minMaxLoc(res3)  # 最小值为最佳
print("方法3 SQDIFF_NORMED (min):", round(mv3,4), ml3)

# 方法4: 自适应二值化模板匹配
_, bg_th = cv2.threshold(bg_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, gap_th = cv2.threshold(gap_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
res4 = cv2.matchTemplate(bg_th, gap_th, cv2.TM_CCOEFF_NORMED)
_, v4, _, l4 = cv2.minMaxLoc(res4)
print("方法4 二值化:", round(v4,4), l4)

# 方法5: gap透明度mask
if len(gap.shape) == 3 and gap.shape[2] >= 3:
    # 如果gap有alpha就用mask，没有就用所有通道
    pass

# 提交各个方法的结果
candidates = [l1, l2, ml3, l4]
names = ["CCOEFF", "Canny", "SQDIFF", "Bin"]
for loc, name in zip(candidates, names):
    resp = submit(loc[0], loc[1], flag)
    code = resp.get("body", {}).get("code", "")
    msg = resp.get("body", {}).get("message", "")
    ok = "通过" if code == 200 else "失败"
    print(f"  {name} ({loc[0]},{loc[1]}): code={code} {ok}")
