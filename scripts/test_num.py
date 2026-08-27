import ddddocr
import os, sys, io, random
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def get_font(size=30):
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)

def gen(text, seed):
    random.seed(seed)
    w, h = 200, 70
    img = Image.new("RGB", (w, h), (random.randint(235,255),)*3)
    d = ImageDraw.Draw(img)
    for _ in range(60):
        d.point((random.randint(0,w-1), random.randint(0,h-1)), fill=(random.randint(150,220),)*3)
    for _ in range(2):
        d.line([(random.randint(0,w//2), random.randint(0,h)), (random.randint(w//2,w), random.randint(0,h))], fill=(random.randint(120,190),)*3, width=1)
    font = get_font(32)
    x = random.randint(5, 15)
    for ch in text:
        ch_img = Image.new("RGBA", (40, 50), (0,0,0,0))
        cd = ImageDraw.Draw(ch_img)
        cd.text((0, 30), ch, fill=(random.randint(30,90),)*3, font=font)
        ch_img = ch_img.rotate(random.uniform(-10, 10), expand=True, resample=Image.BICUBIC)
        img.paste(ch_img, (x, 10), ch_img)
        x += ch_img.width + random.randint(2, 5)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=75)
    return buf.getvalue()

# 数字专用模型（beta + 纯数字字符集）
num_ocr = ddddocr.DdddOcr(show_ad=False, beta=True)
def solve_num(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cands = []
    for b in [img, cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]]:
        for _ in range(3):
            r = num_ocr.classification(cv2.imencode(".png", b)[1].tobytes())
            if r: cands.append(r)
    digit_only = ["".join(c for c in x if c.isdigit()) for x in cands]
    digit_only = [x for x in digit_only if len(x) >= 4]
    if not digit_only:
        # 兜底
        return ""
    return collections.Counter(digit_only).most_common(1)[0][0] if digit_only else ""

import collections
correct = 0
for i in range(50):
    text = "".join(random.choices("0123456789", k=4))
    data = gen(text, i*13)
    r = solve_num(data)
    if r == text: correct += 1
print("纯数字(beta模型): %d/50 = %.1f%%" % (correct, 100*correct/50))
