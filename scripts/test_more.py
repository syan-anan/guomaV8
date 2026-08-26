import httpx, os, sys, json, io, base64, random
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False)

# 尝试从不同来源获取真实图片来测试OCR
test_cases = []

# 1. 从已知的验证码样本网站获取
captcha_urls = [
    "https://raw.githubusercontent.com/nickliqian/cnn_captcha/master/samples/1e7f.jpg",
    "https://raw.githubusercontent.com/nickliqian/cnn_captcha/master/samples/2b3g.jpg",
    "https://raw.githubusercontent.com/nickliqian/cnn_captcha/master/samples/3c4d.jpg",
]

for url in captcha_urls:
    try:
        r = c.get(url)
        if r.status_code == 200 and len(r.content) > 100:
            test_cases.append((f"github_{url.split('/')[-1]}", r.content))
            print(f"  Downloaded: {url.split('/')[-1]} ({len(r.content)}B)")
    except Exception as e:
        pass

# 2. 生成不同噪声级别的验证码
from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

fonts = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
gen = ImageCaptcha(width=200, height=70, fonts=fonts)

# 生成不同配置的验证码
for t in ["ABCD", "1234", "A1B2", "TEST", "CODE", "DATA", "HELL", "WORD", "EXAM", "USER"]:
    buf = io.BytesIO()
    gen.write(t, buf)
    test_cases.append((f"captcha_{t}", buf.getvalue()))

# 3. 测试
correct = 0
total = 0
for name, data in test_cases:
    try:
        r = solve_ocr(data, 1001)
        total += 1
        if name.startswith("captcha_"):
            expected = name.replace("captcha_", "").upper()
            got = r["text"].upper()
            ok = got == expected
            if ok: correct += 1
            prefix = "OK" if ok else "XX"
            if not ok:
                print(f"  [{prefix}] {expected} -> {got} (conf={r['confidence']})")
    except Exception as e:
        print(f"  [ERR] {name}: {e}")

if total > 0:
    print(f"\n测试结果: {correct}/{total} = {100*correct/total:.1f}%")
