import os, sys, json, base64
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from solver.slide import detect_gap_multiscale, detect_gap
from solver.trajectory import generate_track

with open("H:/qinglong/syandaV8/data/yunda_shade.png", "rb") as f:
    bg = f.read()
with open("H:/qinglong/syandaV8/data/yunda_cutout.png", "rb") as f:
    gap = f.read()

bg_b64 = base64.b64encode(bg).decode()
gap_b64 = base64.b64encode(gap).decode()

# 测试各个scale
for scale in [1.0, 0.9, 0.8]:
    r = detect_gap(bg_b64, gap_b64, scale=scale)
    print(f"scale={scale}: {json.dumps(r)}")

# 多尺度
r = detect_gap_multiscale(bg_b64, gap_b64)
print(f"\nmultiscale: {json.dumps(r)}")
print(f"用户手动距离: 193")

if r.get("distance", 0) > 0:
    track = generate_track(r["distance"])
    print(f"轨迹点数: {len(track)}")
    print(f"终点: x={track[-1]['x']} y={track[-1]['y']}")
