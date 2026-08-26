import os, sys, io, json, httpx, base64, cv2, numpy as np
from PIL import Image
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")

from solver.ocr import solve_ocr

# 从网络收集验证码样本
samples = []
output_dir = "H:\\qinglong\\syandaV8\\__cache\\test_samples"
os.makedirs(output_dir, exist_ok=True)

# 1. 从 syandaV8 的 GitHub 仓库下载测试图片
test_urls = [
    # 经典验证码样本
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/1.png", "dddd_1.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/2.png", "dddd_2.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/3.png", "dddd_3.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/4.png", "dddd_4.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/5.png", "dddd_5.png"),
    # 更多验证码样本
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/f0e9e1a0.png", "dddd_a.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/f1b2c3d4.png", "dddd_b.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/f2e3f4a5.png", "dddd_c.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/f3a4b5c6.png", "dddd_d.png"),
    ("https://raw.githubusercontent.com/sml2h3/syandaV8/master/tests/test_img/b1c2d3e4.png", "dddd_e.png"),
    # 通用验证码样本
    ("https://captcha.com/demos/features/captcha-demo-images/simple-captcha-1.png", "captcha_1.png"),
    ("https://captcha.com/demos/features/captcha-demo-images/simple-captcha-2.png", "captcha_2.png"),
]

downloaded = 0
for url, fname in test_urls:
    try:
        r = httpx.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 100:
            path = os.path.join(output_dir, fname)
            with open(path, "wb") as f:
                f.write(r.content)
            samples.append({"path": path, "url": url, "size": len(r.content)})
            downloaded += 1
            print("  OK: %s (%d bytes)" % (fname, len(r.content)))
        else:
            print("  FAIL: %s (status=%d)" % (fname, r.status_code))
    except Exception as e:
        print("  ERROR: %s -> %s" % (fname, e))

print("Downloaded: %d / %d" % (downloaded, len(test_urls)))

# 如果下载到样本，跑 OCR
if samples:
    print("\n=== OCR Testing on %d samples ===" % len(samples))
    for s in samples:
        with open(s["path"], "rb") as f:
            data = f.read()
        result = solve_ocr(data, 1001)
        print("  %s -> %s (conf=%.2f)" % (os.path.basename(s["path"]), result["text"], result["confidence"]))
else:
    print("\nNo samples downloaded, creating synthetic test images")
    # 生成更真实的验证码（带噪点、线条、扭曲）
    from PIL import ImageDraw, ImageFilter
    for i, text in enumerate(["A7K3", "4B7K", "ABCD", "1234", "WXYZ", "3F9P", "J8K2", "MV6N"]):
        img = Image.new("RGB", (150, 50), 255)
        d = ImageDraw.Draw(img)
        for _ in range(5):
            d.line([(np.random.randint(0,50), np.random.randint(0,50)),
                    (np.random.randint(50,150), np.random.randint(0,50))], fill=(160,)*3, width=1)
        for _ in range(20):
            x = np.random.randint(0, 150)
            y = np.random.randint(0, 50)
            d.point((x, y), fill=(100, 100, 100))
        d.text((10, 10), text, fill=(0, 0, 0))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        path = os.path.join(output_dir, "synth_%d.png" % i)
        img.save(path)
        with open(path, "rb") as f:
            data = f.read()
        result = solve_ocr(data, 1001)
        print("  synth_%d(%s) -> %s" % (i, text, result["text"]))

