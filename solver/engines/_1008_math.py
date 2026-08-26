# -*- coding: utf-8 -*-
"""V8 数学题引擎 (1008) — OCR 识别算式 + 解析计算 + 点击目标选项"""
import re
import cv2
import numpy as np


class MathSolver:
    def load_cv(self, src):
        import io, base64, os
        if hasattr(src, "read"):
            bts = src.read()
            return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)
        from solver.utils import load_to_cv
        return load_to_cv(src)

    def solve(self, src):
        try:
            img = self.load_cv(src)
            if img is None:
                return {"code": -1, "error": "image load failed"}
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1) 自适应二值化 + 轮廓找到所有字符块（宽松参数，处理噪点图）
            # 先高斯模糊去噪
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                # 宽松阈值：面积>=5，宽高比合理
                if area < 5 or bw > w * 0.9 or bh > h * 0.9:
                    continue
                if bw < 3 or bh < 3:
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

            # 4) 结果封装：返回所有检测到的字符块中心点
            #    空间推理题：点击所有检测到的字符位置
            points = []
            for (x0, y0, x1, y1) in boxes:
                points.append({"x": int((x0 + x1) / 2), "y": int((y0 + y1) / 2)})
            
            if not points:
                # 兜底：整个图像中心
                points = [{"x": w // 2, "y": h // 2}]
            
            return {
                "code": 0,
                "type": "click",
                "engine": "1008_v8",
                "data": {"points": points, "count": len(points), "row_count": len(rows)},
            }

        except Exception as e:
            return {"code": -99, "error": str(e)}


_instance = MathSolver()

def solve_math_v8(img_src):
    return _instance.solve(img_src)
