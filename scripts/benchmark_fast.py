import os, sys, random, io, time
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from captcha.image import ImageCaptcha
from solver.ocr import solve_ocr
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
cs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
correct = 0
n = 100
t0 = time.time()
for i in range(n):
    t = "".join(random.choices(cs, k=4))
    r = solve_ocr(gen.generate(t).read(), 1001)
    if r["text"].upper() == t:
        correct += 1
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{n}  current: {100*correct/(i+1):.1f}%")
elapsed = time.time() - t0
print("OCR 3-model 100: %.1f%% (%d/%d)  [%.1fs, %.1fms/img]" % (
    100 * correct / n, correct, n, elapsed, 1000 * elapsed / n))
