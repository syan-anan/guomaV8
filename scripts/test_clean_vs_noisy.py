import os, sys, io, base64, random
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# 生成不同清晰度的验证码
def gen_clean(text, font_path=r"C:\Windows\Fonts\arial.ttf", size=60):
    """生成干净验证码，无噪点线条"""
    img = Image.new("RGB", (200, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (200 - tw) // 2
    y = (70 - (bbox[3] - bbox[1])) // 2 - bbox[1]
    for i, ch in enumerate(text):
        ch_img = Image.new("RGB", (50, 70), (255, 255, 255))
        ch_draw = ImageDraw.Draw(ch_img)
        ch_draw.text((5, y), ch, fill=(0, 0, 0), font=font)
        # 随机轻微旋转
        angle = random.uniform(-10, 10)
        ch_img = ch_img.rotate(angle, expand=1, fillcolor=(255, 255, 255))
        img.paste(ch_img, (int(i * 50 + 5), 0))
    return img

def gen_medium(text):
    """中等噪声验证码"""
    from captcha.image import ImageCaptcha
    gen = ImageCaptcha(width=200, height=70,
                       fonts=[r"C:\Windows\Fonts\arial.ttf"],
                       font_sizes=(55, 65))
    return gen.generate(text)

# 测试数据
tests = []
# 干净验证码
for t in ["ABCD", "1234", "A1B2", "TEST", "CODE", "DATA", "HELL", "WORD", "EXAM", "USER",
          "PASS", "GOOD", "HOME", "WORK", "PLAY", "READ", "WALK", "TALK", "CALL", "BABY"]:
    img = gen_clean(t)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    tests.append(("clean", t, buf.getvalue()))

# 中等噪声
for t in ["ABCD", "1234", "A1B2", "TEST", "CODE", "DATA", "HELL", "WORD", "EXAM", "USER"]:
    buf = gen_medium(t)
    tests.append(("medium", t, buf.read()))

# 测试
for level, expected, data in tests:
    r = solve_ocr(data, 1001)
    got = r["text"].upper()
    ok = got == expected
    if not ok:
        print(f"  [{level}] {expected} -> {got} (conf={r['confidence']})")

# 统计
for level in ["clean", "medium"]:
    level_tests = [t for t in tests if t[0] == level]
    correct = sum(1 for t in level_tests if solve_ocr(t[2], 1001)["text"].upper() == t[1])
    print(f"\n{level}: {correct}/{len(level_tests)} = {100*correct/len(level_tests):.1f}%")
