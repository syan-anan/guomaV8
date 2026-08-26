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

print("测试滑块缺口检测...")
result = detect_gap_multiscale(bg_b64, gap_b64)
print("检测结果:", json.dumps(result, ensure_ascii=False))
print("用户手动距离: 193")

if result.get("distance", 0) > 0:
    track = generate_track(result["distance"])
    print("轨迹点数:", len(track))
    print("轨迹起点:", track[0]["x"], track[0]["y"])
    print("轨迹终点:", track[-1]["x"], track[-1]["y"])
    print("轨迹总时长:", track[-1]["t"], "ms")
