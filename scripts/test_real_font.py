import ddddocr
import os, sys, io, random
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def get_font(size=30):
    for fp in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\consola.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except:
            continue
    return ImageFont.load_default()

def gen_real(text, seed=None):
    if seed is not None: random.seed(seed)
    w, h = 200, 70
    img = Image.new("RGB", (w, h), (random.randint(230,255),)*3)
    d = ImageDraw.Draw(img)
    for _ in range(120):
        d.point((random.randint(0,w-1), random.randint(0,h-1)), fill=(random.randint(180,230),)*3)
    for _ in range(3):
        x1, y1 = random.randint(0, w//2), random.randint(0, h)
        x2, y2 = random.randint(w//2, w), random.randint(0, h)
        d.line([(x1, y1), (x2, y2)], fill=(random.randint(100,180),)*3, width=1)
    font = get_font(30)
    x = random.randint(5, 15)
    for ch in text:
        ch_img = Image.new("RGBA", (40, 50), (0,0,0,0))
        cd = ImageDraw.Draw(ch_img)
        cd.text((0, 0), ch, fill=(random.randint(20,80),)*3, font=font)
        angle = random.uniform(-20, 20)
        ch_img = ch_img.rotate(angle, expand=True, resample=Image.BICUBIC)
        img.paste(ch_img, (x, 8), ch_img)
        x += ch_img.width + random.randint(2, 6)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=70)
    return buf.getvalue()

ocr = ddddocr.DdddOcr(show_ad=False)
tests = ["A7K3", "4B7K", "ABCD", "1234", "WXYZ", "3F9P", "J8K2", "MV6N", "5QXH", "H2P9"]
correct = 0
for i, t in enumerate(tests):
    data = gen_real(t, seed=i*3)
    r = ocr.classification(data)
    ok = r.upper() == t
    if ok: correct += 1
    print("  %s -> %s %s" % (t, r, "OK" if ok else "WRONG"))
print("")
print("Accuracy: %d/%d = %.1f%%" % (correct, len(tests), 100*correct/len(tests)))
