import os, sys, json, base64, time, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
import numpy as np, cv2

UA = "Mozilla/5.0 ..."
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
gap = cv2.imdecode(np.frombuffer(base64.b64decode(dd["cutoutImage"]), np.uint8), cv2.IMREAD_UNCHANGED)
print("cutout shape:", gap.shape)
if len(gap.shape) == 3:
    print("channels:", gap.shape[2])
    if gap.shape[2] == 4:
        a = gap[:,:,3]
        print("alpha min/max/mean:", a.min(), a.max(), a.mean())
        print("alpha unique count:", len(np.unique(a)))
        # 透明区比例
        print("alpha<255比例:", np.mean(a<255))
    else:
        # 3通道，检查是否有接近白色的边框
        gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
        print("gray mean:", gray.mean())
        print("白角像素:", gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1])
