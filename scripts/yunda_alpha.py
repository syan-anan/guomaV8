import os, sys, json, time, base64, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
import numpy as np, cv2

UA = "Mozilla/5.0 ..."
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

def detect_alpha(shade_b64, cutout_b64):
    """用alpha mask做模板匹配。"""
    bg = cv2.imdecode(np.frombuffer(base64.b64decode(shade_b64), np.uint8), cv2.IMREAD_COLOR)
    gap = cv2.imdecode(np.frombuffer(base64.b64decode(cutout_b64), np.uint8), cv2.IMREAD_UNCHANGED)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    if gap.shape[2] == 4:
        alpha = gap[:,:,3]
        gap_rgb = gap[:,:,:3]
        gap_bgr = cv2.cvtColor(gap_rgb, cv2.COLOR_RGB2BGR)
        gap_gray = cv2.cvtColor(gap_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        # mask 必须与gap同尺寸单通道
        res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCORR_NORMED, mask=mask)
        _, maxv, _, maxl = cv2.minMaxLoc(res)
        return maxl[0], maxl[1], maxv
    else:
        gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(bg_gray, gap_gray, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv2.minMaxLoc(res)
        return maxl[0], maxl[1], maxv

# 循环测试
passed = 0
total = 10
for i in range(total):
    try:
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

        x, y, conf = detect_alpha(shade_b64, cutout_b64)
        resp = submit(x, y, flag)
        code = resp.get("body", {}).get("code", "")
        ok = code == 200
        if ok: passed += 1
        print(f"[{i}] x={x:3d} y={y:3d} conf={conf:.4f} -> {'OK' if ok else 'FAIL'} code={code}")
    except Exception as e:
        print(f"[{i}] 出错: {e}")

print(f"\n通过率: {passed}/{total} = {100*passed/total:.1f}%")
