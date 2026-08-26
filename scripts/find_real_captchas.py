import httpx, os, sys, json, io, base64
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False, follow_redirects=True)

# 一些服务端生成验证码图的URL（常见形式）
captcha_endpoints = [
    ("http://www.jscodes.com/captcha.php", "jscodes"),
    ("https://www.anquanke.com/captcha", "anquanke"),
    ("https://www.tianyancha.com/verifyimage", "tianyancha"),
    ("https://www.qcc.com/web/captcha", "qcc"),
    ("https://login.51job.com/login/captcha", "51job"),
    ("https://passport.jd.com/verify/image", "jd"),
    ("https://tcredit.com/captcha/image", "tcredit"),
    ("https://www.sogou.com/captcha", "sogou"),
    ("https://user.qunar.com/captcha", "qunar"),
    ("https://passport.weibo.com/sso/captcha", "weibo"),
]

found = 0
for url, name in captcha_endpoints:
    try:
        r = c.get(url)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and len(r.content) > 500 and ("image" in ct or len(r.content) < 50000):
            print(f"[OK/{name}] {url} -> {r.status_code} {len(r.content)}B {ct}")
            found += 1
            # 保存测试
            os.makedirs("H:/qinglong/syandaV8/data", exist_ok=True)
            ext = "png" if "png" in ct else "jpg"
            with open(f"H:/qinglong/syandaV8/data/captcha_{name}.{ext}", "wb") as f:
                f.write(r.content)
    except Exception as e:
        print(f"[SKIP/{name}] {url} -> {str(e)[:50]}")

print(f"\n找到 {found} 个可用的验证码图片源")
