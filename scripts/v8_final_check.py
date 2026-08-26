import os, sys, time, io, traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, 'H:/qinglong/syandaV8')
sys.path.insert(0, 'H:/qinglong/syandaV8/solver')
from master_controller import MasterController

mc = MasterController()
types_to_test = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', 
                 '1009', '1011', '1014', '1015', '1016', '1017']

# Use default bitmap font if arial is missing
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

def generate_mock_captcha(type_code):
    """Generates synthetic captcha images for structural validation."""
    w, h = 300, 120
    img = Image.new('RGB', (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        if type_code in ['1001', '1002', '1003']:
            txt_len = np.random.randint(4, 6)
            allowed_chars = {1001: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                             1002: "0123456789", 
                             1003: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}[int(type_code)]
            txt = ''.join([np.random.choice(list(allowed_chars)) for _ in range(txt_len)])
            for i, c in enumerate(txt):
                x = 20 + i * 45
                draw.text((x, 30), c, fill=(np.random.randint(0, 100), 0, 0), font=font)
                # Add interference line
                p1 = (x-20, np.random.randint(10, 80))
                p2 = (x+40, np.random.randint(10, 80))
                draw.line([p1, p2], fill=(0, 0, 0))
                
        elif type_code == '1004':
            x = np.random.randint(50, 200)
            # Simulate gap edge
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0), width=2)
            
        elif type_code in ['1005', '1015']:
            x, y = np.random.randint(20, 100), np.random.randint(20, 50)
            size = np.random.randint(20, 60)
            draw.rectangle([x, y, x+size, y+size], fill='#FF0000')
            # Add surrounding noise
            for _ in range(10):
                px, py = np.random.randint(0, 300), np.random.randint(0, 120)
                draw.point((px, py), fill=(0, 0, 0))

        elif type_code in ['1006', '1007', '1014', '1016', '1017']:
            # Target object (Red blob)
            x, y = np.random.randint(20, 100), np.random.randint(20, 50)
            r = 20
            draw.ellipse([x-r, y-r, x+r, y+r], fill='#FF0000')
            # Distractors (Blue/Green blobs)
            draw.ellipse([x+50, y+50, x+60, y+60], fill='#0000FF')
            draw.ellipse([x-30, y+30, x-20, y+40], fill='#00FF00')
            
        elif type_code == '1008':
            n1 = np.random.randint(0, 10)
            n2 = np.random.randint(0, 10)
            op = '+'
            draw.text((50, 30), str(n1), fill=(0,0,0), font=font)
            draw.text((100, 30), op, fill=(0,0,0), font=font)
            draw.text((150, 30), str(n2), fill=(0,0,0), font=font)
            
        elif type_code == '1009':
            # Rotated square
            center = (150, 60)
            size = 80
            angle = np.random.uniform(0, 180)
            # Using polygon to simulate rotation
            pts = [center, (center[0]+size, center[1]), (center[0]+size, center[1]+size), (center[0], center[1]+size)]
            # Simple draw doesn't rotate easily, just drawing offset rectangles for visual complexity
            for i in range(0, 100, 20):
                draw.rectangle([50+i, 10, 50+i+5, 110], fill='white' if i%10==0 else None, outline='gray')
            draw.rectangle([100, 40, 160, 80], outline='black', width=2)
            
    except Exception as e:
        pass
        
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

print("=== V8 Final Structural Validation ===")
print(f"Target: {len(types_to_test)} types | 200 iterations/type | Multi-threaded\n")

results = {t: {"passed": 0, "failed": 0, "errors": 0} for t in types_to_test}
total_runs = 200
lock = threading.Lock()

def run_one(t):
    try:
        img_bytes = generate_mock_captcha(t)
        res = mc.solve(t, img_bytes)
        with lock:
            if res.get('code') == 0:
                results[t]["passed"] += 1
            else:
                results[t]["failed"] += 1
    except Exception as e:
        with lock:
            results[t]["errors"] += 1

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for t in types_to_test:
        for _ in range(total_runs):
            futures.append(executor.submit(run_one, t))
            
    for f in as_completed(futures):
        f.result() # Ensure completion

print("-" * 30)
for t in sorted(results.keys()):
    data = results[t]
    total = data["passed"] + data["failed"] + data["errors"]
    print(f"[{t}] Pass: {data['passed']} | Fail: {data['failed']} | Error: {data['errors']} | Rate: {round(data['passed']/total*100, 2) if total>0 else 0}%")

print("\n[V8 Optimization Complete]")
