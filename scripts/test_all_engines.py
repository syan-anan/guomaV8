# -*- coding: utf-8 -*-
"""批量测试所有 19 种引擎能否正常返回结果。"""
import os, sys, base64, io, json
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/dsyandaV8/__cache"
sys.path.insert(0, "H:/qinglong/syandaV8")
from captcha.image import ImageCaptcha
from solver.registry import REGISTRY, solve

# 生成一张测试图
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
buf = io.BytesIO()
gen.write("AB12", buf)
test_img = base64.b64encode(buf.getvalue()).decode()

def test_type(code, **params):
    r = solve(code, **params)
    ok = r.get("code", -1) == 0
    print(f"[{code}] {REGISTRY.get(code, {}).get('name', '?'):24s} -> {'OK' if ok else 'FAIL'} {json.dumps(r.get('data', r), ensure_ascii=False)[:80]}")
    return ok

results = []
# OCR 类型
for code in (1001, 1002, 1003):
    results.append(test_type(code, img=test_img))

# 滑块类型
for code in (1004, 1010, 1012, 1020):
    results.append(test_type(code, bg=test_img))

# 点选-文字
for code in (1005, 1015, 1022):
    results.append(test_type(code, img=test_img, words=["A", "B"]))

# 点选-图标
for code in (1006, 1017, 1023):
    results.append(test_type(code, img=test_img, icons=[test_img]))

# 语序 / 空间
results.append(test_type(1007, img=test_img, phrase="ABC"))
results.append(test_type(1008, img=test_img, question="test"))

# 九宫格
for code in (1018, 1019):
    results.append(test_type(code, img=test_img))

# 点过
for code in (1016, 1021):
    results.append(test_type(code, img=test_img, count=3))

ok_count = sum(results)
print(f"\n通过 {ok_count}/19")

