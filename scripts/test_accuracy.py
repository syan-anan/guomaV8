import os, sys, io, random, collections
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")
import numpy as np, cv2, syandaV8
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def get_font(size=30):
    for fp in [r"C:\Windows\Fonts\arial.ttf"]:
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

def solve(data, ocr):
    CHARSET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    candidates = []
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # collect results from many variants
    def run(b):
        r = ocr.classification(cv2.imencode(".png", b)[1].tobytes())
        if r: candidates.append(r.upper())
    versions = {
        "orig": img,
        "adaptive": cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,15),
        "otsu": cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
        "clahe": cv2.createCLAHE(2.0,(8,8)).apply(gray),
        "denoise_scale": cv2.resize(cv2.fastNlMeansDenoising(gray,None,10,7,21),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC),
    }
    for k, b in versions.items():
        for _ in range(3):
            run(b)
    # filter
    filtered = ["".join(c for c in x if c in CHARSET) for x in candidates]
    filtered2 = [x for x in filtered if len(x) >= 3]
    if not filtered2:
        return ""
    c = collections.Counter(filtered2)
    # prefer longer
    best_len = max(len(x) for x in filtered2)
    longs = [x for x in filtered2 if len(x) == best_len]
    result = collections.Counter(longs).most_common(1)[0][0]
    return result

ocr = syandaV8.DdddOcr(show_ad=False)
all_correct, all_total = 0, 0
for rnd in range(5):
    random.seed(rnd*100)
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    correct = 0
    for i in range(50):
        text = "".join(random.choices(charset, k=4))
        data = gen_real(text, seed=rnd*100 + i*7)
        r = solve(data, ocr)
        if r == text: correct += 1
    all_correct += correct
    all_total += 50
    print("Round %d: %d/50 = %.1f%%" % (rnd+1, correct, 100*correct/50))
print("")
print("Total: %d/%d = %.1f%%" % (all_correct, all_total, 100*all_correct/all_total))

