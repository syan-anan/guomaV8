# -*- coding: utf-8 -*-
"""Optimized Drawing Canvas Solver (1015) - Path following/Skeletonization"""
import cv2, numpy as np

class DrawingCanvasSolver:
    def load_cv(self, src):
        import io, base64
        if isinstance(src, str): 
            bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src,"rb").read()
        elif hasattr(src, "read"): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, src):
        img = self.load_cv(src)
        
        # Convert to gray and threshold to isolate the drawn line/dots
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to connect broken strokes
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(th, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_cnts = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # Filter by size to ignore noise but keep drawing parts
            if 200 < area < 50000:
                valid_cnts.append((x, y, w, h))
                
        if valid_cnts:
            # Return the first significant drawing part found
            bx, by, bw, bh = max(valid_cnts, key=lambda k: k[2]*k[3])
            cx, cy = int(bx + bw/2), int(by + bh/2)
            return {"code": 0, "type": "click", "engine": "1015", "data": {"x": cx, "y": cy}}
            
        return {"code": -1, "error": "no_drawing_found"}
