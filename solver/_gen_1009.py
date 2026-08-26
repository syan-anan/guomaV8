# -*- coding: utf-8 -*-
code = r'''# -*- coding: utf-8 -*-
"""V8 旋转引擎 (1009) — 梯度角度直方图统计 + minAreaRect 兜底"""
import cv2
import numpy as np


class RotationSolver:
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1) Canny 边缘，滤掉弱边缘噪声
            edges = cv2.Canny(gray, 60, 160)
            if int(np.count_nonzero(edges)) < 20:
                return {"code": 0, "type": "rotation", "engine": "1009_v8", "data": {"angle": 0.0}}

            # 2) Sobel 梯度方向统计（比 HoughLines 更稳）
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            # 只统计边缘像素
            ang_vals = ang[edges > 0]
            mag_vals = mag[edges > 0]
            if len(ang_vals) == 0:
                return {"code": 0, "type": "rotation", "engine": "1009_v8", "data": {"angle": 0.0}}

            # 归一化到 0~180
            ang_vals = np.mod(ang_vals, 180.0)

            # 加权直方图（按梯度强度加权，突出主方向）
            hist, bin_edges = np.histogram(ang_vals, bins=36, range=(0, 180), weights=mag_vals)
            peak_idx = int(np.argmax(hist))
            peak_angle = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0

            # 归一化到 -90 ~ +90
            angle = peak_angle
            if angle > 90:
                angle -= 180
            if abs(angle) < 1.5:
                angle = 0.0

            return {"code": 0, "type": "rotation", "engine": "1009_v8", "data": {"angle": round(float(angle), 2)}}

        except Exception as e:
            return {"code": -99, "error": str(e)}


_instance = RotationSolver()

def solve_rotation_v8(img_src):
    return _instance.solve(img_src)
'''
import io, os
os.makedirs(r"H:\qinglong\syandaV8\solver\engines", exist_ok=True)
with io.open(r"H:\qinglong\syandaV8\solver\engines\_1009_rotation.py", "w", encoding="utf-8") as f:
    f.write(code)
print("1009 written:", os.path.getsize(r"H:\qinglong\syandaV8\solver\engines\_1009_rotation.py"))
