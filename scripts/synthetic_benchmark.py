import sys, os, cv2, numpy as np, base64, io, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
sys.path.insert(0, 'H:/qinglong/syandaV8')
sys.path.insert(0, 'H:/qinglong/syandaV8/solver')
from master_controller import MasterController

mc = MasterController()
types_to_test = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', 
                 '1009', '1011', '1014', '1015', '1016', '1017']

def generate_sample(t_type):
    w, h = 300, 100
    # Randomize background noise
    bg_color = tuple(np.random.randint(50, 200, 3))
    img = Image.new('RGB', (w, h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw something visible
    color = tuple(np.random.randint(0, 255, 3))
    
    if t_type in ['1001', '1002', '1003']:
        # Simple text simulation (white rectangle lines)
        for _ in range(5):
            x = np.random.randint(10, 200)
            y = np.random.randint(10, 60)
            draw.rectangle([x, y, x+10, y+30], fill=color)
    elif t_type == '1004':
        # Vertical edge for slider
        x = np.random.randint(50, 200)
        draw.rectangle([x, 0, x+15, 100], fill=(255, 255, 255))
    elif t_type == '1005':
        # Square blob
        x, y = np.random.randint(50, 150), np.random.randint(20, 60)
        draw.rectangle([x, y, x+50, y+50], fill=color)
    elif t_type in ['1006', '1007', '1014', '1015', '1016', '1017']:
        # Target object
        x, y = np.random.randint(50, 200), np.random.randint(20, 60)
        draw.ellipse([x, y, x+40, y+40], fill=color)
    else:
        # Generic shape
        x, y = np.random.randint(10, 250), np.random.randint(10, 60)
        draw.rectangle([x, y, x+20, y+20], fill=color)
        
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

print("=== Synthetic Stability Benchmark ===")
results = {t: {"passed": 0, "failed": 0, "errors": []} for t in types_to_test}
total_runs = 50

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {}
    for t in types_to_test:
        for i in range(total_runs):
            f = executor.submit(mc.solve, t, None)
            futures[f] = (t, i)
            
    for f in as_completed(futures):
        t, idx = futures[f]
        try:
            # Generate image just for this call
            img_bytes = generate_sample(t)
            res = f.result() # Actually wait for result here? No, need to pass image.
            # Fix: submit lambda with arg
        except Exception as e:
            pass

# Re-writing loop properly for closure capture
results = {t: 0 for t in types_to_test}
def run_test(t):
    try:
        img_bytes = generate_sample(t)
        res = mc.solve(t, img_bytes)
        return t, res.get('code') == 0, str(res)
    except Exception as e:
        return t, False, str(e)[:50]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for t in types_to_test:
        for _ in range(total_runs):
            futures.append(executor.submit(run_test, t))
            
for f in as_completed(futures):
    t, success, info = f.result()
    if success:
        results[t] += 1
        
for t, count in sorted(results.items()):
    print(f"[{t}] Pass: {count}/{total_runs}")
