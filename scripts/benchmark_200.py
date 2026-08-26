import os, sys, random, io, time
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from captcha.image import ImageCaptcha
from solver.ocr import solve_ocr
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
cs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
correct = 0
n = 200
t0 = time.time()
errors = []
for i in range(n):
    t = "".join(random.choices(cs, k=4))
    r = solve_ocr(gen.generate(t).read(), 1001)
    if r["text"].upper() == t:
        correct += 1
    else:
        errors.append((t, r["text"], r["confidence"]))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{n}  {100*correct/(i+1):.1f}%")
elapsed = time.time() - t0
print(f"\nOCR 200: {100*correct/n:.1f}% ({correct}/{n})  [{elapsed:.0f}s, {1000*elapsed/n:.0f}ms/img]")
if errors:
    print(f"\n误读 {len(errors)} 张:")
    for exp, got, conf in errors[:15]:
        print(f"  expect={exp} got={got} conf={conf}")
