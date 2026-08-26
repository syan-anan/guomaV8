# -*- coding: utf-8 -*-
"""V8 Comprehensive Local Stress Test (Synthetic Generation)"""
import sys, os, io, random, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.insert(0, 'H:/qinglong/syandaV8')
sys.path.insert(0, 'H:/qinglong/syandaV8/solver')
from master_controller import MasterController

mc = MasterController()
types_to_test = [
    # Optimized Group A
    '1007', '1008', '1009', '1011', 
    # Optimized Group B
    '1005', '1014', '1016', '1015', '1017',
    # Standard Group C
    '1001', '1002', '1003', '1004', '1006'
]

def generate_captcha(t_code):
    w, h = 300, 100
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    if t_code in ['1001', '1002', '1003']:
        txt = ''.join([str(random.randint(0,9)) for _ in range(5)])
        if t_code == '1003': txt = txt.upper()
        for i, c in enumerate(txt):
            x = 20 + i * 45
            y = random.randint(10, 30)
            draw.text((x, y), c, fill=(0,0,0), font=font)
            
    elif t_code == '1004':
        x = random.randint(50, 200)
        draw.line([(x, 0), (x, h)], fill=(0,0,0), width=3)
        
    elif t_code == '1005':
        x, y = random.randint(50, 150), random.randint(10, 60)
        draw.rectangle([x, y, x+50, y+50], fill='#CC0000')
        
    elif t_code == '1006':
        draw.rectangle([50, 20, 100, 70], fill='#0000FF')
        
    elif t_code == '1007':
        x, y = random.randint(50, 150), random.randint(10, 60)
        draw.text((x, y), "A", fill='#FF0000', font=font)
        
    elif t_code == '1008':
        draw.text((50, 30), "1", fill=(0,0,0), font=font)
        draw.text((100, 30), "+", fill=(0,0,0), font=font)
        draw.text((150, 30), "2", fill=(0,0,0), font=font)
        
    elif t_code == '1009':
        pts = [(50, 50), (150, 50), (160, 60), (60, 60)]
        draw.polygon(pts, outline='black')
        
    elif t_code == '1011':
        x = random.randint(50, 200)
        draw.line([(x, 0), (x, h)], fill=(0,0,0), width=3)
        
    elif t_code == '1014':
        draw.ellipse([50, 20, 80, 50], fill='#00CC00')
        draw.ellipse([100, 20, 130, 50], fill='#00CC00')
        
    elif t_code == '1015':
        draw.rectangle([50, 20, 100, 60], fill='#FFFF00')
        
    elif t_code == '1016':
        draw.rectangle([50, 20, 90, 60], fill='#0000CC')
        
    elif t_code == '1017':
        draw.text((50, 30), "A", fill='#FF0000', font=font)
        
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

print("--- V8 Engine Validation Start ---")
stats = {t: {"passed": 0, "failed": 0} for t in types_to_test}

for t in types_to_test:
    passed = 0
    failed = 0
    for _ in range(200):
        img_bytes = generate_captcha(t)
        res = mc.solve(t, img_bytes)
        if res.get('code') == 0:
            passed += 1
        else:
            failed += 1
            
    stats[t]["passed"] = passed
    stats[t]["failed"] = failed
    rate = round((passed/200)*100, 2)
    print(f"[{t}] Rate: {rate}% ({passed}/200)")

print("--- V8 Engine Validation End ---")
