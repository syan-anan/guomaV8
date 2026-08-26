# -*- coding: utf-8 -*-
"""复现韵达滑块验证完整流程：请求验证码->模板匹配->提交验证。"""
import os, sys, json, time, base64, httpx
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.slide import detect_gap_multiscale
from solver.trajectory import generate_track

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 MicroMessenger/7.0.20.1781(0x6700143B) NetType/WIFI MiniProgramEnv/Windows WindowsWechat/WMPF WindowsWechat(0x63090a13) UnifiedPCWindowsWechat(0xf2541b37) XWEB/25297"
REFERER = "https://servicewechat.com/wxdeb5309aa3e93fd1/695/page-frame.html"
BASE = "https://mbpxapi.yundasys.com:38861/gateway/interface"
ACCOUNT = "oPJUI0diLY2l-xkwBOCTxXj54fd0"
APPID = "wjvxmno358lze827"

c = httpx.Client(timeout=20, verify=False, headers={
    "User-Agent": UA,
    "Referer": REFERER,
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
})

def get_verify_code():
    now = int(time.time() * 1000)
    payload = {
        "version": "V1.0",
        "action": "ydmbaccount.ydaccount.getImageVerifyCode",
        "data": {
            "client": "mobile",
            "slideImageWidth": 318,
            "type": "slide",
            "accountId": ACCOUNT,
            "accountSrc": "wxapp",
            "reqTime": now,
        },
        "appid": APPID,
        "req_time": now,
        "options": False,
    }
    r = c.post(BASE + "?ydmbaccount.ydaccount.getImageVerifyCode", json=payload)
    return r.json()

def submit_code(flag, dist):
    now = int(time.time() * 1000)
    payload = {
        "version": "V1.0",
        "action": "ydmbintegral.ydintegral.obtain.event.integral",
        "data": {
            "channelId": "wxapp",
            "itgType": "browse",
            "imageCode": f"{dist}|124",
            "flag": flag,
            "accountId": ACCOUNT,
            "accountSrc": "wxapp",
            "reqTime": now,
        },
        "appid": APPID,
        "req_time": now,
        "options": False,
    }
    r = c.post(BASE + "?ydmbintegral.ydintegral.obtain.event.integral", json=payload)
    return r.json()

import base64 as b
# 步骤1: 获取验证码
print("步骤1: 请求滑块验证码...")
try:
    data = get_verify_code()
    body = data.get("body", {})
    if isinstance(body, str):
        body = json.loads(body)
    dd = body.get("data", {})

    if not dd or not dd.get("shadeImage"):
        print("获取验证码失败:", json.dumps(data, ensure_ascii=False)[:300])
    else:
        flag = dd["flag"]
        shade = b.b64decode(dd["shadeImage"])
        cutout = b.b64decode(dd["cutoutImage"]) if dd.get("cutoutImage") else None
        print(f"  flag={flag}")
        print(f"  shadeImage: {len(shade)} bytes, cutout: {len(cutout) if cutout else 0} bytes")

        # 步骤2: 引擎检测距离
        print("\n步骤 2: 引擎检测滑块距离...")
        shade_b64 = b.b64encode(shade).decode()
        cutout_b64 = b.b64encode(cutout).decode() if cutout else None
        result = detect_gap_multiscale(shade_b64, cutout_b64)
        dist = result["distance"]
        print(f"  检测距离: {dist}, 方法: {result['method']}, 置信度: {result['confidence']}")

        # 步骤3: 提交验证
        print("\n步骤 3: 提交验证...")
        # 尝试几个附近的距离提高通过率
        for d in [dist, dist-1, dist+1, dist-2, dist+2]:
            resp = submit_code(flag, d)
            body2 = resp.get("body", {})
            success = resp.get("success", False)
            print(f"  距离={d}: success={success} {json.dumps(resp, ensure_ascii=False)[:150]}")
            if success:
                print("\n*** 验证通过! ***")
                break
except Exception as e:
    print("出错:", e)
