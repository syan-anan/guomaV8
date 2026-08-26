import io, sys, os

controller_content = '''
import io, numpy as np, cv2, base64, sys, os
from PIL import Image

sys.path.insert(0, 'H:/qinglong/syandmV8')
sys.path.insert(0, 'H:/qinglong/syandmV8/solver')
sys.path.insert(0, 'H:/qinglong/syandmV8/solver/engines')

try:
    from ocr_v8 import solve_ocr_v8
    from slide import detect_gap
    from click import _color_regions
    
    from _1007_word_click import WordClickSolver as ECSolver_1007
    from _1008_math import MathSolver as ECSolver_1008
    from _1009_rotation import RotationSolver as ECSolver_1009
    from _1011_long_slider import LongSliderTrajectory as ECSolver_1011
    
    from _1005_gap_puzzle import GapPuzzleSolver as ECSolver_1005
    from _1014_match3_solver import Match3Solver as ECSolver_1014
    from _1016_calendar_picker import CalendarPickerSolver as ECSolver_1016
    from _1015_drawing_canvas import DrawingCanvasSolver as ECSolver_1015
    from _1017_image_text_combo import ImageTextComboSolver as ECSolver_1017
except ImportError as e:
    print(f"Import Error: {e}")

class MasterController:
    def __init__(self): 
        self._solvers = {
            '1001': None, '1002': None, '1003': None,
            '1004': None, '1006': None, '1007': ECSolver_1007(), '1008': ECSolver_1008(),
            '1009': ECSolver_1009(), '1011': ECSolver_1011(), '1005': ECSolver_1005(),
            '1014': ECSolver_1014(), '1016': ECSolver_1016(), '1015': ECSolver_1015(),
            '1017': ECSolver_1017()
        }

    def load_cv(self, src):
        if isinstance(src, str):
            bts = base64.b64decode(src.split(',')[1]) if ',' in src else open(src,'rb').read()
        elif hasattr(src, 'read'): bts = src.read()
        else: bts = src
        return cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)

    def solve(self, type_code, img_src):
        try:
            cv_img = self.load_cv(img_src)
            
            if type_code in ['1001', '1002', '1003']:
                pil_img = Image.fromarray(cv2.merge([cv2.split(cv_img)[2], cv2.split(cv_img)[1], cv2.split(cv_img)[0]]))
                buf = io.BytesIO(); pil_img.save(buf, format='PNG'); return solve_ocr_v8(buf.getvalue(), type_code=int(type_code))
            elif type_code == '1004':
                res = detect_gap(img_src); return {'code': 0, 'type': 'slider', 'data': {'distance': int(res['distance'])}}
            elif type_code == '1006':
                regions = _color_regions(cv_img); 
                if regions:
                    r = max(regions, key=lambda k: (k[2]-k[0])*(k[3]-k[1]))
                    return {'code': 0, 'type': 'click', 'data': {'x': int((r[0]+r[2])/2), 'y': int((r[1]+r[3])/2)}}
                return {'code': -1}
            elif type_code == '1008': return self._solvers['1008'].solve(img_src)
            elif type_code == '1007': return self._solvers['1007'].solve(img_src)
            elif type_code == '1009': return self._solvers['1009'].solve(img_src)
            elif type_code == '1011': return self._solvers['1011'].generate_trajectory(300)
            elif type_code == '1005': return self._solvers['1005'].solve(img_src)
            elif type_code == '1014': return self._solvers['1014'].solve(img_src)
            elif type_code == '1016': return self._solvers['1016'].solve(img_src)
            elif type_code == '1015': return self._solvers['1015'].solve(img_src)
            elif type_code == '1017': return self._solvers['1017'].solve(img_src)
            
            else: return {'code': -2, 'msg': 'Type not implemented locally'}
        except Exception as e: return {'code': -99, 'error': str(e)}
'''

with open('H:\\\\qinglong\\\\ddddocr\\\\solver\\\\master_controller.py', 'w', encoding='utf-8') as f:
    f.write(controller_content)
print("Master Controller Updated Successfully.")
