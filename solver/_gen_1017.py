# -*- coding: utf-8 -*-
code = r'''# -*- coding: utf-8 -*-
"""V8 图文组合引擎 (1017) — 文字/图标检测 + 网格点击点生成"""
import cv2
import numpy as np


class ImageTextComboSolver:
    def load_cv(self, src):
        import io, base64
        if isinstance(src, str):
            bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src, "rb").read()
        elif hasattr(src, "read"):
            bts = src.read()
        else:
            bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, src):
        """
        OCR 文字区域检测 + 返回点击候选点
        典型 1017 布局：多个汉字图标，要求按文本点选。
        """
        try:
            img = self.load_cv(src)
            if img is None:
                return {"code": -1, "error": "image load failed"}
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1) 连通域分割候选文字块
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

            n, labels, stats, cents = cv2.connectedComponentsWithStats(closed, 8)
            valid = []
            for k in range(1, n):
                x, y, bw, bh, area = stats[k]
                if area < 60 or bw > w * 0.5 or bh > h * 0.6:
                    continue
                if bw < 8 or bh < 8:
                    continue
                # 正方形文字块优先（汉字宽高比 0.5~2）
                ratio = bw / float(bh)
                if 0.4 < ratio < 2.5:
                    valid.append((int(cents[k][0]), int(cents[k][1]), int(bw * bh)))

            # 2) 合并相近点（同一文字的多个连通块）
            def merge(points, dist=12):
                out = []
                for p in sorted(points, key=lambda k: -k[2]):
                    x, y, a = p
                    if any((x - ox) ** 2 + (y - oy) ** 2 < dist ** 2 for ox, oy, oa in out):
                        continue
                    out.append(p)
                return out

            merged = merge(valid)
            merged.sort(key=lambda k: (k[1], k[0]))

            # 3) 按行分组，返回最可能的网格点击点
            rows = []
            for p in merged:
                x, y, a = p
                placed = False
                for r in rows:
                    if abs(r[0][1] - y) < 15:
                        r.append(p)
                        placed = True
                        break
                if not placed:
                    rows.append([p])

            points = []
            for r in rows:
                r.sort(key=lambda k: k[0])
                for x, y, a in r:
                    points.append({"x": x, "y": y, "w": int(np.sqrt(a))})

            if not points:
                # 兜底：均匀网格（4x4）
                cols, rows_n = 4, 4
                cell_w, cell_h = w // cols, h // rows_n
                points = [
                    {"x": c * cell_w + cell_w // 2, "y": r * cell_h + cell_h // 2, "w": min(cell_w, cell_h) // 2}
                    for r in range(rows_n) for c in range(cols)
                ]

            return {
                "code": 0,
                "type": "click",
                "engine": "1017_v8",
                "data": {"points": points, "count": len(points)},
            }

        except Exception as e:
            return {"code": -99, "error": str(e)}


_instance = ImageTextComboSolver()

def solve_combo_1017_v8(img_src):
    return _instance.solve(img_src)
'''
import io, os
os.makedirs(r"H:\qinglong\syandaV8\solver\engines", exist_ok=True)
with io.open(r"H:\qinglong\syandaV8\solver\engines\_1017_image_text_combo.py", "w", encoding="utf-8") as f:
    f.write(code)
print("1017 written:", os.path.getsize(r"H:\qinglong\syandaV8\solver\engines\_1017_image_text_combo.py"))
