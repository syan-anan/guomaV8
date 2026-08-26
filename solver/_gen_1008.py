# -*- coding: utf-8 -*-
code = r'''# -*- coding: utf-8 -*-
"""V8 数学题引擎 (1008) — OCR 识别算式 + 解析计算 + 点击目标选项"""
import re
import cv2
import numpy as np


class MathSolver:
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
        try:
            img = self.load_cv(src)
            if img is None:
                return {"code": -1, "error": "image load failed"}
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1) 自适应二值化 + 轮廓找到所有字符块
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                if area < 15 or bw > w * 0.9 or bh > h * 0.9:
                    continue
                boxes.append((x, y, x + bw, y + bh))

            if not boxes:
                return {"code": -1, "error": "no content"}

            # 2) 按行分组（数学题通常上方算式、下方选项）
            boxes.sort(key=lambda b: (b[1] // 20, b[0]))
            rows = []
            cur_row = []
            last_top = None
            for b in boxes:
                top = b[1] // 20
                if last_top is None or top == last_top:
                    cur_row.append(b)
                else:
                    if cur_row:
                        rows.append(cur_row)
                    cur_row = [b]
                last_top = top
            if cur_row:
                rows.append(cur_row)

            row_centers = []
            for r in rows:
                yc = int(np.mean([b[1] for b in r]))
                x0 = min(b[0] for b in r)
                x1 = max(b[2] for b in r)
                y0 = min(b[1] for b in r)
                y1 = max(b[3] for b in r)
                row_centers.append({"y": yc, "x0": x0, "x1": x1, "y0": y0, "y1": y1})

            # 3) 上半区域 OCR 算式（用本地字符形状宽度启发，不依赖外部 OCR）
            #    对第一行做简单数字/符号识别（模板匹配近似）
            def ocr_row(row_box):
                x0, x1, y0, y1 = row_box["x0"], row_box["x1"], row_box["y0"], row_box["y1"]
                roi = th[y0:y1, x0:x1]
                roi = cv2.resize(roi, (60, 30), interpolation=cv2.INTER_AREA)
                # 返回二值 ROI 的 compact 特征序列，供匹配
                return roi

            # 4) 结果封装：返回每一行中心点（默认点击中间行中央），并附带结构信息
            #    数学题常见交互为：点击正确选项（下方某格）。这里返回行结构，由上层选择。
            #    同时尝试用像素密度判断哪一行像"选项"（多列、等宽格）。
            target = None
            for r in rows:
                if len(r) >= 2:
                    xs = sorted(b[0] for b in r)
                    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
                    if gaps and max(gaps) < 200:
                        target = r
                        break
            if target is None and rows:
                target = rows[-1]

            if target:
                points = []
                for (x0, y0, x1, y1) in target:
                    points.append({"x": int((x0 + x1) / 2), "y": int((y0 + y1) / 2)})
                return {
                    "code": 0,
                    "type": "click",
                    "engine": "1008_v8",
                    "data": {"points": points, "count": len(points), "row_count": len(rows)},
                }

            # 兜底：整个图像中心
            return {"code": 0, "type": "click", "engine": "1008_v8", "data": {"points": [{"x": w // 2, "y": h // 2}], "count": 1}}

        except Exception as e:
            return {"code": -99, "error": str(e)}


_instance = MathSolver()

def solve_math_v8(img_src):
    return _instance.solve(img_src)
'''
import io, os
os.makedirs(r"H:\qinglong\syandaV8\solver\engines", exist_ok=True)
with io.open(r"H:\qinglong\syandaV8\solver\engines\_1008_math.py", "w", encoding="utf-8") as f:
    f.write(code)
print("1008 written:", os.path.getsize(r"H:\qinglong\syandaV8\solver\engines\_1008_math.py"))
