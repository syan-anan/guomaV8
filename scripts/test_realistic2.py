import os, sys, io, random, collections
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from solver.ocr import solve_ocr

# 真实验证码风格测试
# 使用更真实的干扰程度（类似真实网站的验证码）
def gen_realistic(text, seed):
    random.seed(seed)
    w, h = 200, 70
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34)
    img = Image.new("RGB", (w, h), (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255)))
    d = ImageDraw.Draw(img)
    # 轻量干扰线
    for _ in range(2):
        x1 = random.randint(0, w//2)
        y1 = random.randint(10, h-10)
        x2 = random.randint(w//2, w)
        y2 = random.randint(10, h-10)
        d.line([(x1, y1), (x2, y2)], fill=(random.randint(160, 200),)*3, width=1)
    # 轻量噪点
    for _ in range(30):
        d.point((random.randint(0, w-1), random.randint(0, h-1)), fill=(random.randint(160, 210),)*3)
    # 字符绘制（轻微旋转）
    x = random.randint(5, 15)
    for ch in text:
        ch_img = Image.new("RGBA", (40, 55), (0,0,0,0))
        cd = ImageDraw.Draw(ch_img)
        cd.text((5, 25), ch, fill=(random.randint(30, 100),)*3, font=font)
        ch_img = ch_img.rotate(random.uniform(-5, 5), expand=True, resample=Image.BICUBIC)
        img.paste(ch_img, (x, 8), ch_img)
        x += 40 + random.randint(0, 3)
    # 轻微模糊
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
    return buf.getvalue()

# 测试 10 轮，每轮 100 个
total_correct = 0
total_count = 0
for rnd in range(5):
    random.seed(rnd * 137)
    correct = 0
    for i in range(100):
        text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))
        data = gen_realistic(text, rnd * 1000 + i * 7)
        result = solve_ocr(data, 1001)
        if result["text"].upper() == text:
            correct += 1
    total_correct += correct
    total_count += 100
    print("Round %d: %d/100 = %.1f%%" % (rnd+1, correct, 100*correct/100))
print("Total: %d/%d = %.1f%%" % (total_correct, total_count, 100*total_correct/total_count))
