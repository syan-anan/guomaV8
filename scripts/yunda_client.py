# -*- coding: utf-8 -*-
"""韵达滑块生产示例 - 本地或服务器 API + 客户端重试到 98%+.\n依赖: pip install httpx\n"""
import httpx, time, json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_retry import solve_with_retry

API = "http://localhost:8000/solve"
BASE = "https://mbpxapi.yundasys.com:38861/gateway/interface"
ACCOUNT = "oPJUI0diLY2l-xkwBOCTxXj54fd0"
APPID = "wjvxmno358lze827"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

def api_solve(bg_b64, gap_b64):
    r = httpx.post(API + "/solve", json={"type": 1004, "image": bg_b64, "gap_image": gap_b64}, timeout=30)
    d = r.json()
    return d["data"]

def fetch():
    now = int(time.time() * 1000)
    payload = {"version": "V1.0", "action": "ydmbaccount.ydaccount.getImageVerifyCode",
        "data": {"client": "mobile", "slideImageWidth": 318, "type": "slide",
                 "accountId": ACCOUNT, "accountSrc": "wxapp", "reqTime": now},
        "appid": APPID, "req_time": now, "options": False}
    c = httpx.post(BASE + "?ydmbaccount.ydaccount.getImageVerifyCode",
                   json=payload, headers={"User-Agent": UA, "Content-Type": "application/json"}, timeout=20, verify=False)
    dd = json.loads(c.json().get("body", "{}")).get("data", {}) if isinstance(c.json().get("body"), str) else c.json().get("body", {}).get("data", {})
    return dd

def submit(ans, cap):
    now = int(time.time() * 1000)
    payload = {"version": "V1.0", "action": "ydmbintegral.ydintegral.obtain.event.integral",
            "data": {"channelId": "wxapp", "itgType": "browse", "imageCode": str(ans["distance"]) + "|" + str(ans.get("y", 0)),
                     "flag": cap["flag"], "accountId": ACCOUNT, "accountSrc": "wxapp", "reqTime": now},
            "appid": APPID, "req_time": now, "options": False}
    r = httpx.post(BASE + "?ydmbintegral.ydintegral.obtain.event.integral",
               json=payload, headers={"User-Agent": UA, "Content-Type": "application/json"}, timeout=20, verify=False)
    body = r.json().get("body", {});
    return body.get("code", body) == 200

def solve_one(cap):
    return api_solve(cap["shadeImage"], cap.get("cutoutImage", ""))

if __name__ == "__main__":
    ok, tries = solve_with_retry(fetch, submit, solve_one, max_retry=3)
    print("PASS" if ok else "FAIL", "attempts=", tries)
