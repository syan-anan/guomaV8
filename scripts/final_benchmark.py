import httpx, os, sys, json, io, base64
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False, follow_redirects=True)

# 从oschina获取多张验证码测试
print("=== oschina 验证码识别测试 ===")
correct = 0
total = 5
for i in range(total):
    try:
        r = c.get("https://www.oschina.net/action/user/captcha", headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 200:
            res = solve_ocr(r.content, 1001)
            # 保存图片
            os.makedirs("H:/qinglong/syandaV8/data", exist_ok=True)
            with open(f"H:/qinglong/syandaV8/data/oschina_{i}.gif", "wb") as f:
                f.write(r.content)
            print(f"  [{i}] OCR: {res['text']} (conf={res['confidence']}, votes={res['votes']})")
    except Exception as e:
        print(f"  [{i}] Error: {e}")

# 也测试一下英文字母验证码的常见模式
print("\n=== 干净验证码批量测试 ===")
from PIL import Image, ImageDraw, ImageFont
def gen_clean(text):
    img = Image.new("RGB", (200, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 55)
    for i, ch in enumerate(text):
        ch_img = Image.new("RGB", (50, 70), (255, 255, 255))
        ch_draw = ImageDraw.Draw(ch_img)
        ch_draw.text((5, 5), ch, fill=(0, 0, 0), font=font)
        import random
        angle = random.uniform(-8, 8)
        ch_img = ch_img.rotate(angle, expand=1, fillcolor=(255, 255, 255))
        img.paste(ch_img, (int(i * 50 + 5), 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

import random
tests = ["".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4)) for _ in range(50)]
correct = 0
for t in tests:
    data = gen_clean(t)
    res = solve_ocr(data, 1001)
    if res["text"].upper() == t:
        correct += 1
print(f"干净验证码 50 张: {correct}/50 = {100*correct/50:.1f}%")

# 带噪点验证码（模拟真实场景）
print("\n=== 模拟真实条件验证码测试 ===")
from captcha.image import ImageCaptcha
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"], font_sizes=(55, 65))
tests2 = ["".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4)) for _ in range(50)]
correct2 = 0
for t in tests2:
    data = gen.generate(t).read()
    res = solve_ocr(data, 1001)
    if res["text"].upper() == t:
        correct2 += 1
print(f"captcha-lib 50 张: {correct2}/50 = {100*correct2/50:.1f}%")
