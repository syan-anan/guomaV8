import os, sys, io, re
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
import httpx
from PIL import Image
from solver.ocr import solve_ocr

r = httpx.get("https://www.geetest.com/en/demo", timeout=15, follow_redirects=True)
html = r.text
imgs = re.findall(r'<img[^>]+src="([^"]+)"', html)
print("Geetest images found:", len(imgs))
for img_url in imgs[:5]:
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = "https://www.geetest.com" + img_url
    print(" ", img_url)
    try:
        r2 = httpx.get(img_url, timeout=10, follow_redirects=True)
        if r2.status_code == 200 and len(r2.content) > 500:
            img = Image.open(io.BytesIO(r2.content))
            print("   Size: %dx%d" % (img.width, img.height))
            result = solve_ocr(r2.content, 1001)
            print("   OCR: %s" % result["text"])
    except Exception as e:
        print("   Error: %s" % e)

# 找 captcha js/api
api_urls = re.findall(r"https?://[^\'\" ]*(?:captcha|gt|geetest|sense)[^\'\" ]*", html)
print("\nCaptcha API URLs:", api_urls[:10])
