import os; base_dir = r'H:\qinglong\syandaV8\solver\engines'
path = os.path.join(base_dir, '_1016_calendar_picker.py')
with open(path, 'w', encoding='utf-8') as f: 
    f.write('''
import cv2, numpy as np
class CalendarPickerSolver:
    def solve(self, src):
        try:
            arr = self.load_cv(src)
            h, w = arr.shape[0], arr.shape[1]
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            _, th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in cnts if 100 < cv2.contourArea(c) < 10000]
            
            if not boxes:
                return {"code": 0, "type": "click", "engine": "1016", "data": {"x": int(w//2), "y": int(h//2)}}
                
            bx = max(boxes, key=lambda k: k[2]*k[3])
            return {"code": 0, "type": "click", "engine": "1016", "data": {"x": int(bx[0]+bx[2]/2), "y": int(bx[1]+bx[3]/2)}}
        except Exception as e: 
            return {"code": -99, "error": str(e)}
    def load_cv(self, src):
        import io, base64, numpy as np
        if isinstance(src, str): bts = base64.b64decode(src.split(',')[1]) if ',' in src else open(src,'rb').read()
        elif hasattr(src, 'read'): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)
'''); print('Fixed 1016')
