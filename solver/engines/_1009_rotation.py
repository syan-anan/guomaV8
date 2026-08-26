# -*- coding: utf-8 -*-
"""V8 旋转引擎 (1009) — 梯度角度直方图统计 + minAreaRect 兜底"""
import cv2
import numpy as np


class RotationSolver:
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
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {"code": 0, "type": "rotation", "engine": "1009_v8", "data": {"angle": 0.0}}
            
            largest = max(contours, key=cv2.contourArea)
            if len(largest) < 5:
                return {"code": 0, "type": "rotation", "engine": "1009_v8", "data": {"angle": 0.0}}
            
            ellipse = cv2.fitEllipse(largest)
            _, axes, raw_angle = ellipse
            
            if raw_angle > 90:
                base = raw_angle - 180
            else:
                base = raw_angle
            
            abs_angle = abs(base)
            if abs_angle > 30:
                calibrated = abs_angle * 1.55
            else:
                calibrated = abs_angle * 1.25
            
            if calibrated > 85:
                calibrated = 85
            
            angle = (1 if base >= 0 else -1) * calibrated
            
            if abs(angle) < 3:
                angle = 0.0
            
            return {"code": 0, "type": "rotation", "engine": "1009_v8", "data": {"angle": round(float(angle), 2)}}
        except Exception as e:
            return {"code": -99, "error": str(e)}
_instance = RotationSolver()

def solve_rotation_v8(img_src):
    return _instance.solve(img_src)

