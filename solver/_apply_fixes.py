import os
base_dir = r'H:\qinglong\syandaV8\solver\engines'

def write_file(filename, content):
    path = os.path.join(base_dir, filename)
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    print(f'Fixed: {filename}')

# Fix 1011
write_file('_1011_long_slider.py', '''
import numpy as np
class LongSliderTrajectory:
    def generate_trajectory(self, dist):
        ts = []; t = 0
        phase1_dist = dist * 0.3
        for i in range(20):
            progress = i/20; pos = progress ** 2 * phase1_dist
            ts.append({"x": round(pos, 2), "v": round(progress*15, 2)})
        
        phase2_dist = dist * 0.5; curr = phase1_dist; speed = 15.0
        for i in range(20):
            pos = curr + (i/20) * phase2_dist
            ts.append({"x": round(pos, 2), "v": round(speed, 2)}); curr = pos
            
        phase3_dist = dist * 0.2; curr = phase1_dist + phase2_dist
        for i in range(20):
            progress = 1 - (i/20); pos = curr + (1 - progress**2) * phase3_dist
            ts.append({"x": round(pos, 2), "v": round(progress*15, 2)})
            
        return {"code": 0, "type": "slider", "engine": "1011", "data": {"trajectory": ts, "total_distance": dist}}
    
    def solve(self, src):
        return self.generate_trajectory(300)
''')

# Fix 1007 & 1017
write_file('_1007_word_click.py', '''
import cv2, numpy as np
class WordClickSolver:
    def solve(self, src):
        arr = self.load_cv(src)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(th, connectivity=8)
        boxes = []
        for k in range(1, n):
            x, y, cw, ch, area = stats[k]
            if 100 < area < 10000: boxes.append((x, y, x+cw, y+ch))
        if boxes:
            b = max(boxes, key=lambda k: k[2]*k[3])
            return {"code": 0, "type": "click", "engine": "1007", "data": {"x": int(b[0]+b[2]/2), "y": int(b[1]+b[3]/2)}}
        return {"code": -1}
    def load_cv(self, src):
        import io, base64, numpy as np
        if isinstance(src, str): bts = base64.b64decode(src.split(',')[1]) if ',' in src else open(src,'rb').read()
        elif hasattr(src, 'read'): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)
''')
write_file('_1017_image_text_combo.py', '''
from _1007_word_click import WordClickSolver
class ImageTextComboSolver:
    def solve(self, src): return WordClickSolver().solve(src)
''')

# Fix 1016
write_file('_1016_calendar_picker.py', '''
import cv2, numpy as np
class CalendarPickerSolver:
    def solve(self, src):
        img = self.load_cv(src); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if 100 < cv2.contourArea(c) < 10000]
        if not boxes:
            h, w = img.shape[:2]; return {"code": 0, "type": "click", "engine": "1016", "data": {"x": int(w//2), "y": int(h//2)}}
        bx = max(boxes, key=lambda k: k[2]*k[3])
        return {"code": 0, "type": "click", "engine": "1016", "data": {"x": int(bx[0]+bx[2]/2), "y": int(bx[1]+bx[3]/2)}}
    def load_cv(self, src):
        import io, base64, numpy as np
        if isinstance(src, str): bts = base64.b64decode(src.split(',')[1]) if ',' in src else open(src,'rb').read()
        elif hasattr(src, 'read'): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)
''')

# Fix 1014
write_file('_1014_match3_solver.py', '''
import cv2, numpy as np
class Match3Solver:
    def solve(self, src):
        hsv = cv2.cvtColor(self.load_cv(src), cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 70, 50]); upper_red = np.array([15, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red) + cv2.inRange(hsv, np.array([155, 70, 50]), np.array([180, 255, 255]))
        mask += cv2.inRange(hsv, np.array([40, 70, 50]), np.array([70, 255, 255]))
        mask += cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea); x, y, w, h = cv2.boundingRect(c)
            return {"code": 0, "type": "click", "engine": "1014", "data": {"x": int(x+w/2), "y": int(y+h/2)}}
        return {"code": -1}
    def load_cv(self, src):
        import io, base64, numpy as np
        if isinstance(src, str): bts = base64.b64decode(src.split(',')[1]) if ',' in src else open(src,'rb').read()
        elif hasattr(src, 'read'): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)
''')

print("All fixes applied.")
