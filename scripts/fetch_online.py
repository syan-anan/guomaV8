import os, sys, json, re, httpx
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
from solver.ocr import solve_ocr

# 尝试访问 Geetest 公开 API
try:
    r = httpx.post("https://api.geetest.com/register.php", 
        data={"gt": "f2b8c7e8f9d4a6b2c0e1d3f5a7b9c0d1"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10)
    print("Register:", r.status_code, r.text[:200])
except Exception as e:
    print("Register error:", e)

# 从易盾 demo 页面找图片
try:
    r = httpx.get("https://dun.163.com/trial/sense", timeout=15, follow_redirects=True)
    print("Yidun page:", r.status_code, len(r.text))
    imgs = re.findall(r"https?://[^\"\'\s]+\.(?:png|jpg|jpeg|gif)", r.text)
    for u in imgs[:10]:
        print("  img:", u)
        # 尝试下载
        try:
            r2 = httpx.get(u, timeout=10, follow_redirects=True)
            if r2.status_code == 200 and len(r2.content) > 500:
                result = solve_ocr(r2.content, 1001)
                print("    OCR:", result["text"])
        except Exception as e2:
            print("    download error:", e2)
except Exception as e:
    print("Yidun error:", e)
