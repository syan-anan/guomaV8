import os, sys, json, time, base64, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
import numpy as np, cv2

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ..."
BASE = "https://mbpxapi.yundasys.com:38861/gateway/interface"
ACCOUNT = "oPJUI0diLY2l-xkwBOCTxXj54fd0"
APPID = "wjvxmno358lze827"

c = httpx.Client(timeout=20, verify=False, headers={"User-Agent": UA, "Content-Type": "application/json"})
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
shade_b64 = dd["shadeImage"]
cutout_b64 = dd.get("cutoutImage", "")
flag = dd["flag"]

from solver.utils import load_to_cv
bg = load_to_cv(shade_b64)
gap = load_to_cv(cutout_b64)
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)

# 尝试多种模板匹配方法
methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF]
names = ["CCOEFF_NORMED", "CCORR_NORMED", "CCOEFF"]
dists = []
for m, n in zip(methods, names):
    res = cv2.matchTemplate(bg_gray, gap_gray, m)
    _, maxv, _, maxl = cv2.minMaxLoc(res)
    dist = maxl[0]
    dists.append(dist)
    print(f"  {n}: max_val={maxv:.4f} max_loc={maxl} dist={dist}")

# 用所有距离提交看哪个能过
def submit(dist, flag):
    now2 = int(time.time() * 1000)
    p = {"version": "V1.0", "action": "ydmbintegral.ydintegral.obtain.event.integral",
         "data": {"channelId": "wxapp", "itgType": "browse", "imageCode": f"{dist}|124",
                  "flag": flag, "accountId": ACCOUNT, "accountSrc": "wxapp", "reqTime": now2},
         "appid": APPID, "req_time": now2, "options": False}
    r = c.post(BASE + "?ydmbintegral.ydintegral.obtain.event.integral", json=p)
    return r.json()

print("\n提交不同距离验证:")
for d in set(dists):
    resp = submit(d, flag)
    code = resp.get("body", {}).get("code", "")
    msg = resp.get("body", {}).get("message", "")
    print(f"  dist={d}: code={code} msg={msg}")
