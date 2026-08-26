# 验证滑块、点选、九宫格等题型
import httpx, json, os, sys, base64, io
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from captcha.image import ImageCaptcha
from solver.ocr import solve_ocr

c = httpx.Client(timeout=10)
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
buf = io.BytesIO()
gen.write("AB12", buf)
b64 = base64.b64encode(buf.getvalue()).decode()

print("=== 滑块测试 ===")
for code in [1004, 1010, 1012, 1020]:
    r = c.post("http://localhost:8000/solve", json={"type": code, "image": b64})
    d = r.json()
    dist = d["data"].get("distance", "?")
    conf = d["data"].get("confidence", d["data"].get("method","?"))
    track_len = len(d["data"].get("track", []))
    print(f"  [{code}] dist={dist} conf={conf} track={track_len}pts")

print("\n=== 点选-图标测试 ===")
for code in [1006, 1017, 1023]:
    r = c.post("http://localhost:8000/solve", json={"type": code, "image": b64, "extra": {"icons": [b64]}})
    d = r.json()
    print(f"  [{code}] points={d['data'].get('count',0)}")

print("\n=== 九宫格测试 ===")
for code in [1018, 1019]:
    r = c.post("http://localhost:8000/solve", json={"type": code, "image": b64, "extra": {"positions": [1,5,9]}})
    d = r.json()
    pts = d["data"].get("points", [])
    print(f"  [{code}] points={len(pts)} first={pts[0] if pts else 'none'}")

print("\n=== 点过测试 ===")
for code in [1016, 1021]:
    r = c.post("http://localhost:8000/solve", json={"type": code, "image": b64, "extra": {"count": 3}})
    d = r.json()
    print(f"  [{code}] points={d['data'].get('count',0)}")

print("\n所有题型测试完成!")
