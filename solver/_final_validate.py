import sys, os, cv2, numpy as np, base64, io
from PIL import Image
sys.path.insert(0, 'H:/qinglong/syandaV8')
sys.path.insert(0, 'H:/qinglong/syandaV8/solver')
from master_controller import MasterController

mc = MasterController()

# Directories for samples (assumed structure: H:\qinglong\samples\{type_code}\)
sample_base = "H:/qinglong/samples"
types_to_test = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', 
                 '1009', '1011', '1014', '1015', '1016', '1017']

print("=== V8 Optimized Engine Validation ===")
for t in types_to_test:
    dir_path = os.path.join(sample_base, t)
    if not os.path.exists(dir_path):
        print(f"[SKIP] {t} - No sample directory found")
        continue
    
    files = sorted(os.listdir(dir_path))[:200] # Max 200 per type
    if not files:
        print(f"[SKIP] {t} - Empty directory")
        continue
        
    passed = 0
    total = len(files)
    
    for f in files:
        try:
            img_src = os.path.join(dir_path, f)
            res = mc.solve(t, img_src)
            # Pass if code is 0 and we got a valid response
            if res.get('code') == 0:
                passed += 1
            else:
                print(f"  [FAIL] {f}: {res}")
        except Exception as e:
            print(f"  [ERROR] {f}: {e}")
            
    rate = round((passed/total)*100, 2) if total > 0 else 0
    print(f"[RESULT] {t}: {passed}/{total} -> {rate}%")

print("=== Validation Complete ===")
