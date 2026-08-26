import httpx, os, json, sys
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False)

# 测试真实验证码图片
test_urls = [
    "https://www.gstatic.com/recaptcha/api2/logo.png",
    "https://httpbin.org/image/png",
]

for url in test_urls:
    try:
        r = c.get(url)
        ct = r.headers.get("content-type", "")
        print(f"[{url}] -> {r.status_code} {len(r.content)}B type={ct}")
        if "image" in ct or len(r.content) > 100:
            r2 = solve_ocr(r.content, 1001)
            print(f"  OCR: {json.dumps(r2, ensure_ascii=False)}")
    except Exception as e:
        print(f"[{url}] -> FAIL: {e}")
