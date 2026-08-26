# -*- coding: utf-8 -*-
"""Optimized Gap Puzzle Solver (1005)"""
import cv2, numpy as np

class GapPuzzleSolver:
    def load_cv(self, src):
        import io, base64
        if isinstance(src, str): 
            bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src,"rb").read()
        elif hasattr(src, "read"): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, src):
        img = self.load_cv(src)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Template Matching (if gap image provided separately in src tuple - simplified here for single image)
        # Fallback to color/edge based contour detection
        
        # Use adaptive threshold to handle varying lighting
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect broken edges of the puzzle piece
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(th, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        
        cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_cnts = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # Filter by area and aspect ratio typical of puzzle pieces
            if 500 < area < 50000 and 0.5 < w/h < 1.5:
                valid_cnts.append(c)
                
        if valid_cnts:
            best_cnt = max(valid_cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(best_cnt)
            return {"code": 0, "type": "click", "engine": "1005", "data": {"x": int(x+w/2), "y": int(y+h/2)}}
            
        return {"code": -1, "error": "no_puzzle_detected"}
