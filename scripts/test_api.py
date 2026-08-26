import os, sys, io, base64
os.chdir("H:\\qinglong\\syandaV8")
sys.path.insert(0, ".")
from PIL import Image, ImageDraw
import numpy as np, cv2, httpx

# 启动服务
import threading, uvicorn
from api.server import app
t = threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error"), daemon=True)
t.start()
import time; time.sleep(2.5)

c = httpx.Client(base_url="http://127.0.0.1:8001", timeout=60)

# 1. OCR 测试
img = Image.new("RGB", (150, 50), 255)
ImageDraw.Draw(img).text((10, 10), "A7K3", fill=(0, 0, 0))
b = io.BytesIO(); img.save(b, "PNG")
b64 = base64.b64encode(b.getvalue()).decode()
r = c.post("/solve", json={"type": 1001, "image": b64}).json()
print("OCR 1001:", r)

# 2. 滑块测试
bg = np.ones((400, 600, 3), dtype=np.uint8) * 200
cv2.rectangle(bg, (200, 100), (260, 300), (120, 120, 120), -1)
_, bb = cv2.imencode(".png", bg)
gap = np.ones((200, 60, 3), dtype=np.uint8) * 120
_, gb = cv2.imencode(".png", gap)
r2 = c.post("/solve", json={"type": 1004, "image": base64.b64encode(bb.tobytes()).decode(), "gap_image": base64.b64encode(gb.tobytes()).decode()}).json()
print("Slide 1004:", r2)

# 3. 九宫格
grid = np.ones((300, 300, 3), dtype=np.uint8) * 255
r3 = c.post("/solve", json={"type": 1018, "image": base64.b64encode(cv2.imencode(".png", grid)[1].tobytes()).decode(), "extra": {"positions": [1, 5, 9]}}).json()
print("NineGrid 1018:", r3)

# 4. 无效题型
r4 = c.post("/solve", json={"type": 9999, "image": b64}).json()
print("Invalid:", r4)

print("=== API TEST DONE ===")
