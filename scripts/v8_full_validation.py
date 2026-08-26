# -*- coding: utf-8 -*-
"""V8 Comprehensive Local Stress Test"""
import sys, os, io, random, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, 'H:/qinglong/syandaV8')
sys.path.insert(0, 'H:/qinglong/syandaV8/solver')
from master_controller import MasterController

mc = MasterController()
types_to_test = [
    '1001', '1002', '1003', '1004', '1005', '1006', 
    '1007', '1008', '1009', '1011', '1012', '1013', 
    '1014', '1015', '1016', '1017', '1018', '1019'
]

def gen(type_code):
    w, h = 300, 100
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 30)
    except: font = ImageFont.load_default()

    if type_code in ['1001', '1002', '1003']:
        txt = ''.join([str(random.randint(0,9)) for _ in range(5)])
        if type_code == '1003': txt = txt.upper()
        for i, c in enumerate(txt): draw.text((20 + i*45, 30), c, fill=(0,0,0), font=font)
    elif type_code in ['1004', '1011']:
        draw.line([(random.randint(50, 200), 0), (random.randint(50, 200), h)], fill='black', width=3)
    elif type_code in ['1005', '1015']:
        draw.rectangle([random.randint(50, 150), random.randint(10, 60), 100, 60], fill='#CC0000')
    elif type_code in ['1006', '1014', '1016', '1017']:
        draw.ellipse([50, 20, 100, 70], fill='#00FF00')
    elif type_code == '1007':
        draw.text((50, 30), "A", fill='#FF0000', font=font)
    elif type_code == '1008':
        draw.text((50, 30), "1", fill=(0,0,0), font=font)
        draw.text((100, 30), "+", fill=(0,0,0), font=font)
        draw.text((150, 30), "2", fill=(0,0,0), font=font)
    elif type_code == '1009':
        draw.polygon([(50, 50), (150, 50), (160, 60), (60, 60)], outline='black')
    elif type_code in ['1012', '1013', '1018', '1019']:
        # Random noise for unimplemented types
        for _ in range(50): draw.point((random.randint(0,w), random.randint(0,h)), fill=random.choice(['black', 'white']))
        
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

print("--- V8 Validation Start ---")
for t in types_to_test:
    passed = 0
    for _ in range(200):
        res = mc.solve(t, gen(t))
        if res.get('code') == 0: passed += 1
    print(f"[{t}] Pass Rate: {passed}/200 ({round(passed, 1)}%)")
print("--- Done ---")
