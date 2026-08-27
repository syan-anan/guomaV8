import ddddocr
import os
os.chdir("H:/qinglong/syandaV8")
f = open("scripts/test_accuracy2.py", "w", encoding="utf-8")
f.write("""import os, sys, io, random, collections
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def get_font(size=32):
    return ImageFont.truetype(r"C:\\Windows\\Fonts\\arial.ttf", size)

def gen(text, seed):
    random.seed(seed)
    w, h = 200, 70
    img = Image.new("RGB", (w, h), (random.randint(240,255),)*3)
    d = ImageDraw.Draw(img)
    for _ in range(60):
        d.point((random.randint(0,w-1), random.randint(0,h-1)), fill=(random.randint(180,220),)*3)
    for _ in range(2):
        d.line([(random.randint(0,w//2), random.randint(10,h-10)), (random.randint(w//2,w), random.randint(10,h-10))], fill=(random.randint(140,190),)*3, width=1)
    font = get_font(36)
    x = random.randint(5, 15)
    for ch in text:
        ch_img = Image.new("RGBA", (45, 55), (0,0,0,0))
        cd = ImageDraw.Draw(ch_img)
        cd.text((5, 20), ch, fill=(random.randint(30,90),)*3, font=font)
        ch_img = ch_img.rotate(random.uniform(-10, 10), expand=True, resample=Image.BICUBIC)
        img.paste(ch_img, (x, 8), ch_img)
        x += 42
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=75)
    return buf.getvalue()

ocr = ddddocr.DdddOcr(show_ad=False)
correct = 0
for i in range(100):
    text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))
    data = gen(text, i*17)
    r = ocr.classification(data).upper()
    if r == text: correct += 1
print("Single-pass 英数: %d/100 = %.1f%%" % (correct, correct))
""")
f.close()
print("done")
