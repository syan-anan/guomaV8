import httpx, os, sys, json
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False)

# 尝试从不同来源获取验证码图片
urls = [
    # 一些有验证码图片的网站
    "https://www.zhihu.com/captcha?lang=en&type=login",
    "https://www.google.com/recaptcha/api2/logo.png",
    "https://httpbin.org/image/jpeg",
]

for url in urls:
    try:
        r = c.get(url)
        print(f"[{url}] -> {r.status_code} {len(r.content)}B {r.headers.get('content-type','')}")
        if len(r.content) > 500:
            result = solve_ocr(r.content, 1001)
            print(f"  OCR: {json.dumps(result, ensure_ascii=False)}")
    except Exception as e:
        print(f"[{url}] -> FAIL: {e}")

# 尝试从验证码识别平台获取测试图片
print("\n--- 尝试访问免费验证码 API ---")
try:
    r = c.get("https://captcha.com/demos/features/captcha-demo.aspx", timeout=10)
    print(f"captcha.com demo: {r.status_code} {len(r.content)}B")
except Exception as e:
    print(f"captcha.com: {e}")
