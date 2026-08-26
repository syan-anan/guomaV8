# -*- coding: utf-8 -*-
"""Optimized Match3 Solver (1014) - Grid based color detection"""
import cv2, numpy as np

class Match3Solver:
    def load_cv(self, src):
        import io, base64
        if isinstance(src, str): 
            bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src,"rb").read()
        elif hasattr(src, "read"): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, src):
        img = self.load_cv(src)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define broader HSV ranges for common match3 items
        lower_red = np.array([0, 70, 50]); upper_red = np.array([15, 255, 255])
        mask_r1 = cv2.inRange(hsv, lower_red, upper_red)
        mask_r2 = cv2.inRange(hsv, np.array([155, 70, 50]), np.array([180, 255, 255]))
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)
        
        lower_green = np.array([40, 70, 50]); upper_green = np.array([70, 255, 255])
        mask_g = cv2.inRange(hsv, lower_green, upper_green)
        
        lower_blue = np.array([100, 70, 50]); upper_blue = np.array([130, 255, 255])
        mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
        
        combined_mask = cv2.bitwise_or(mask_r, cv2.bitwise_or(mask_g, mask_b))
        
        # Morphological cleanup
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(combined_mask, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        max_area = 0
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area > 100 and area < 10000: # Filter small noise and large background
                if area > max_area:
                    max_area = area
                    best_box = (x, y, w, h)
                    
        if best_box:
            bx, by, bw, bh = best_box
            cx, cy = int(bx + bw/2), int(by + bh/2)
            return {"code": 0, "type": "click", "engine": "1014", "data": {"x": cx, "y": cy}}
            
        return {"code": -1, "error": "no_item_found"}
