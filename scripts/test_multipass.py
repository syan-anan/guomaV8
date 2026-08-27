import ddddocr
import os, sys, io, random, collections
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

# 多策略 OCR
def multi_pass_ocr(data, ocr):
    CHARSET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    candidates = []
    raw = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    # 策略1: 原图
    for _ in range(3):
        r = ocr.classification(data)
        if r: candidates.append(r)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 策略2: 自适应阈值
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    for _ in range(2):
        r = ocr.classification(cv2.imencode(".png", bw)[1].tobytes())
        if r: candidates.append(r)
    # 策略3: OTSU
    _, bw2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for _ in range(2):
        r = ocr.classification(cv2.imencode(".png", bw2)[1].tobytes())
        if r: candidates.append(r)
    # 策略4: 去噪+放大
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    scaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw3 = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for _ in range(2):
        r = ocr.classification(cv2.imencode(".png", bw3)[1].tobytes())
        if r: candidates.append(r)
    # 策略5: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    for _ in range(2):
        r = ocr.classification(cv2.imencode(".png", enhanced)[1].tobytes())
        if r: candidates.append(r)
    # 投票
    filtered = []
    for c in candidates:
        filtered.append("".join(ch.upper() for ch in c if ch.upper() in CHARSET))
    # 按长度降序，取最长的
    filtered.sort(key=lambda x: len(x), reverse=True)
    if filtered:
        # 选出最多出现的
        counter = collections.Counter(filtered)
        return counter.most_common(1)[0][0]
    return ""

ocr = ddddocr.DdddOcr(show_ad=False)
tests = ["A7K3", "4B7K", "ABCD", "1234", "WXYZ", "3F9P", "J8K2", "MV6N", "5QXH", "H2P9"]
correct = 0
for i, t in enumerate(tests):
    data = gen_real(t, seed=i*3)
    r = multi_pass_ocr(data, ocr)
    ok = r.upper() == t
    if ok: correct += 1
    print("  %s -> %s %s" % (t, r, "OK" if ok else "WRONG"))
print("")
print("Multi-pass Accuracy: %d/%d = %.1f%%" % (correct, len(tests), 100*correct/len(tests)))
