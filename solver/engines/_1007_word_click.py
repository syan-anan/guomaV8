# -*- coding: utf-8 -*-
"""Optimized Word Click Solver (1007)"""
import cv2, numpy as np

class WordClickSolver:
    def __init__(self):
        self.min_area = 100
        
    def load_cv(self, src):
        import io, base64
        if isinstance(src, str): 
            bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src,"rb").read()
        elif hasattr(src, "read"): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, src):
        img = self.load_cv(src)
        
        # Strategy 1: Color Segmentation (High Precision)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([15, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5,5),np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        max_valid_area = 0
        
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if 200 < area < 10000 and 0.2 < cw/ch < 5.0:
                if area > max_valid_area:
                    max_valid_area = area
                    best_box = (x, y, cw, ch)
        
        # Strategy 2: Fallback to Text/Gray Blob detection
        if not best_box:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(th, connectivity=8)
            boxes = []
            for k in range(1, n):
                x, y, cw, ch, area = stats[k]
                if self.min_area < area < 10000:
                    boxes.append((x, y, cw, ch))
            
            if boxes:
                best_box = max(boxes, key=lambda k: k[2]*k[3])

        if best_box:
            bx, by, bw, bh = best_box
            cx, cy = int(bx + bw/2), int(by + bh/2)
            return {"code": 0, "type": "click", "engine": "1007", "data": {"x": cx, "y": cy}}
        
        return {"code": -1, "error": "no_target_found"}
