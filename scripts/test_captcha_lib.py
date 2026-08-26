import os, sys, io, random, collections
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
from captcha.image import ImageCaptcha
from solver.ocr import solve_ocr
import cv2, numpy as np

# 用 captcha 库生成标准验证码
generator = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])

test_count = 100
charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
correct = 0
results = []

for i in range(test_count):
    text = "".join(random.choices(charset, k=4))
    data = generator.generate(text).read()
    # data 是 PNG bytes
    result = solve_ocr(data, 1001)
    ok = result["text"].upper() == text
    if ok:
        correct += 1
    else:
        results.append({"text": text, "got": result["text"]})

print("Captcha library test: %d/%d = %.1f%%" % (correct, test_count, 100*correct/test_count))
if results:
    print("Failures:")
    for r in results[:10]:
        print("  wanted: %s, got: %s" % (r["text"], r["got"]))
