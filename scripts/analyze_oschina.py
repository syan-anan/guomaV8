import os, sys
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
import numpy as np, cv2
from PIL import Image

for i in range(5):
    img = Image.open(f"H:/qinglong/syandaV8/data/oschina_{i}.gif").convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # 分析图像特征
    print(f"\n--- oschina_{i}.gif ---")
    print(f"  尺寸: {img.size}, 灰度均值: {gray.mean():.1f}, 标准差: {gray.std():.1f}")
    # 二值化分析文字区域
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 检查笔画密度
    black = np.sum(bw < 128)
    white = np.sum(bw >= 128)
    print(f"  黑色像素: {black}, 白色像素: {white}, 比值: {black/(black+white):.3f}")
    # 检测非白(文字)连通区域
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = [c for c in contours if cv2.contourArea(c) > 20]
    print(f"  连通区域: {len(chars)}")
    for ci, c in enumerate(chars):
        x, y, w, h = cv2.boundingRect(c)
        if w > 3 and h > 5:
            print(f"    区域{ci}: x={x} y={y} w={w} h={h}")
