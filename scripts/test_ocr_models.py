import os, sys, syandaV8, io, cv2, numpy as np, random
from PIL import Image, ImageDraw, ImageFilter
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")

models = {
    "default": syandaV8.DdddOcr(show_ad=False),
    "beta": syandaV8.DdddOcr(show_ad=False, beta=True),
    "old": syandaV8.DdddOcr(show_ad=False, old=True),
}

def gen_captcha(text):
    img = Image.new("RGB", (150, 50), 255)
    d = ImageDraw.Draw(img)
    for _ in range(3):
        d.line([(random.randint(0,50), random.randint(0,50)),
                (random.randint(50,150), random.randint(0,50))], fill=(180,)*3, width=1)
    d.text((10, 10), text, fill=(0, 0, 0))
    return img

tests = ["A7K3", "4B7K", "ABCD", "1234", "WXYZ"]
for name, ocr in models.items():
    correct = 0
    for t in tests:
        img = gen_captcha(t)
        b = io.BytesIO(); img.save(b, "PNG")
        r = ocr.classification(b.getvalue())
        ok = r.upper() == t
        if ok: correct += 1
        print("  %s: %s -> %s %s" % (name, t, r, "OK" if ok else "WRONG"))
    print("  %s accuracy: %d/%d" % (name, correct, len(tests)))
    print()
