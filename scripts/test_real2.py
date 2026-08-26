import httpx, os, sys, json, base64, io
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.ocr import solve_ocr

c = httpx.Client(timeout=15, verify=False)

# 尝试从各种验证码生成服务获取图片
sources = []

# 1. 字符验证码生成服务
for i in range(5):
    try:
        # 模仿一些常见验证码
        w, h = 200, 70
        r = c.get(f"https://picsum.photos/{w}/{h}?random={i}")
        if r.status_code == 200 and len(r.content) > 500:
            sources.append(("picsum", r.content))
    except:
        pass

# 2. 用 Python 生成一些更真实的验证码（带噪点、线条）
from captcha.image import ImageCaptcha
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
for t in ["AB12", "CD34", "EF56", "GH78", "IJ90", "KLAB", "MNOP", "QRST", "UVWX", "YZ01",
          "A7K3", "TEST", "CODE", "DATA", "USER", "PASS", "HELL", "WORD", "EXAM", "BEST"]:
    buf = io.BytesIO()
    gen.write(t, buf)
    sources.append((f"captcha_{t}", buf.getvalue()))

# 测试
correct = 0
total = 0
for name, data in sources:
    try:
        r = solve_ocr(data, 1001)
        total += 1
        expected = name.replace("captcha_", "").upper() if name.startswith("captcha_") else "?"
        got = r["text"].upper()
        if expected != "?":
            ok = got == expected
            if ok: correct += 1
            prefix = "OK" if ok else "XX"
            if not ok:
                print(f"  [{prefix}] {expected} -> {got} (conf={r['confidence']})")
        else:
            print(f"  [??] {name} -> {got}")
    except Exception as e:
        print(f"  [ERR] {name}: {e}")

if total > 0 and correct > 0:
    print(f"\n真实测试: {correct}/{total} = {100*correct/total:.1f}%")
