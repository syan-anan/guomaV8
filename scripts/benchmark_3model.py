import os, sys, random, io
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from captcha.image import ImageCaptcha
from solver.ocr import solve_ocr
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
cs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
correct = 0
n = 500
for i in range(n):
    t = "".join(random.choices(cs, k=4))
    r = solve_ocr(gen.generate(t).read(), 1001)
    if r["text"].upper() == t:
        correct += 1
print("OCR 3-model 500: %.1f%% (%d/%d)" % (100 * correct / n, correct, n))
