import os, sys
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")
from solver.ocr import solve_ocr
from solver.slide import detect_gap_multiscale
from solver.trajectory import generate_track
from solver.registry import REGISTRY
import io, cv2, numpy as np
from PIL import Image, ImageDraw

print("=== OCR Test ===")
img = Image.new("RGB", (150, 50), 255)
d = ImageDraw.Draw(img)
d.text((10, 10), "A7K3", fill=(0, 0, 0))
buf = io.BytesIO(); img.save(buf, format="PNG")
r = solve_ocr(buf.getvalue(), 1001)
print("OCR 1001:", r)

print("=== Slide Test ===")
bg = np.ones((400, 600, 3), dtype=np.uint8) * 200
cv2.rectangle(bg, (200, 100), (260, 300), (120, 120, 120), -1)
_, bgbuf = cv2.imencode(".png", bg)
gap = np.ones((200, 60, 3), dtype=np.uint8) * 120
_, gapbuf = cv2.imencode(".png", gap)
r2 = detect_gap_multiscale(bgbuf.tobytes(), gapbuf.tobytes())
print("Slide:", r2)

print("=== Track Test ===")
track = generate_track(r2.get("distance", 200))
print("Track samples:", track[:3])
print("Track last:", track[-3:])

print("=== Registry ===")
for k, v in sorted(REGISTRY.items()):
    print("  %d: %s" % (k, v["name"]))
print("=== ALL ENGINES OK ===")
