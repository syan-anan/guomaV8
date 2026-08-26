import os, sys, json, time, base64, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.slide import detect_gap

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

passed = 0
total = 30
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
        if not dd.get("shadeImage"): continue

        result = detect_gap(dd["shadeImage"], dd.get("cutoutImage", ""), scale=1.0)
        dist = result["distance"]
        y = result["y"]
        conf = result["confidence"]
        met = result["method"]

        resp = submit(dist, y, dd["flag"])
        code = resp.get("body", {}).get("code", "")
        ok = code == 200
        if ok: passed += 1
        print(f"[{i:2d}] x={dist:3d} y={y:3d} conf={conf:.4f} met={met} -> {'OK' if ok else 'FAIL'} code={code}")
    except Exception as e:
        print(f"[{i:2d}] Error: {e}")

print(f"\n通过率: {passed}/{total} = {100*passed/total:.1f}%")
