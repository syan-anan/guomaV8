import httpx, os, sys, json, io, base64, random
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False, follow_redirects=True)

# 尝试更多验证码图片来源
endpoints = [
    # 大学/机构验证码（很多大学用简单验证码）
    ("https://sso.ustc.edu.cn/captcha", "ustc"),
    ("https://cas.bit.edu.cn/captcha", "bit"),
    ("https://cas.hust.edu.cn/captcha", "hust"),
    # 一些论坛/社区验证码
    ("https://www.52pojie.cn/misc.php?mod=seccode&action=update&idhash=", "52pojie"),
    ("https://www.right.com.cn/forum/misc.php?mod=seccode&action=update", "right"),
    # 一些开发平台
    ("https://www.oschina.net/action/user/captcha", "oschina"),
    ("https://passport.csdn.net/account/verify/randi", "csdn"),
    # 一些验证码样本
    ("https://captcha.com/demos/features/captcha-demo.aspx", "captcha.com"),
]

for url, name in endpoints:
    try:
        r = c.get(url, headers={"User-Agent": "Mozilla/5.0"})
        ct = r.headers.get("content-type", "")
        print(f"[{name}] {url} -> {r.status_code} {len(r.content)}B {ct[:50]}")
        if ("image" in ct or ct.startswith("application/octet")) and len(r.content) > 200:
            os.makedirs("H:/qinglong/syandaV8/data", exist_ok=True)
            ext = "png" if "png" in ct else "jpg"
            fname = f"H:/qinglong/syandaV8/data/captcha_{name}.{ext}"
            with open(fname, "wb") as f:
                f.write(r.content)
            print(f"  Saved: {fname}")
            res = solve_ocr(r.content, 1001)
            print(f"  OCR: {json.dumps(res, ensure_ascii=False)[:100]}")
    except Exception as e:
        print(f"[{name}] {url} -> {str(e)[:60]}")
