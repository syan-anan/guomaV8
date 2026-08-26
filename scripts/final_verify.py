import httpx, json, os, sys, io, base64, random
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")

c = httpx.Client(timeout=10)
r = c.get("http://localhost:8000/")
print("API状态:", json.dumps(r.json(), ensure_ascii=False, indent=2))

r2 = c.get("http://localhost:8000/types")
types = r2.json()["types"]
print("\n已注册题型:", len(types))
for t in types:
    print(f"  [{t['code']}] {t['name']}")

# 测试OCR
from captcha.image import ImageCaptcha
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
buf = io.BytesIO()
gen.write("AB12", buf)
b64 = base64.b64encode(buf.getvalue()).decode()

print("\n--- OCR测试 ---")
for code, name in [(1001, "英数"), (1002, "数字"), (1003, "字母")]:
    r = c.post("http://localhost:8000/solve", json={"type": code, "image": b64})
    d = r.json()
    print(f"  [{code}] {name}: text={d['data'].get('text','?')} conf={d['data'].get('confidence',0)}")
