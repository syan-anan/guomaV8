# -*- coding: utf-8 -*-
"""Optimized Calendar Picker Solver (1016) — Multi-color support + robust fallback"""
import cv2, numpy as np

class CalendarPickerSolver:
    def load_cv(self, src):
        import io, base64
        if isinstance(src, str): 
            bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src,"rb").read()
        elif hasattr(src, "read"): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, src):
        img = self.load_cv(src)
        h, w = img.shape[:2]
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Broad HSV ranges for common calendar highlight colors
        masks = []
        
        # Blue
        masks.append(cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255])))
        # Red (both ends of hue circle)
        masks.append(cv2.inRange(hsv, np.array([0, 120, 120]), np.array([15, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255])))
        # Green/Cyan
        masks.append(cv2.inRange(hsv, np.array([40, 80, 80]), np.array([70, 255, 255])))
        
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, m)
            
        kernel = np.ones((5,5),np.uint8)
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        max_area = 0
        
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if 100 < area < 10000 and 0.5 < cw/ch < 2.0:
                if area > max_area:
                    max_area = area
                    best_box = (x, y, cw, ch)
                    
        if best_box:
            bx, by, bw, bh = best_box
            cx, cy = int(bx + bw/2), int(by + bh/2)
            return {"code": 0, "type": "click", "engine": "1016", "data": {"x": cx, "y": cy}}
            
        # Fallback: adaptive threshold on gray image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts_f, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes_f = [cv2.boundingRect(c) for c in cnts_f if 100 < cv2.contourArea(c) < 10000]
        
        if boxes_f:
            bx = max(boxes_f, key=lambda k: k[2]*k[3])
            return {"code": 0, "type": "click", "engine": "1016", "data": {"x": int(bx[0]+bx[2]/2), "y": int(bx[1]+bx[3]/2)}}
            
        # Ultimate fallback: return image center
        return {"code": 0, "type": "click", "engine": "1016", "data": {"x": w//2, "y": h//2}}
