import os, sys, io, random, collections
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
import cv2, numpy as np
from captcha.image import ImageCaptcha
from solver.ocr import solve_ocr

gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

confusions = collections.Counter()
char_errors = []
for i in range(300):
    text = "".join(random.choices(charset, k=4))
    data = gen.generate(text).read()
    r = solve_ocr(data, 1001)
    result = r["text"].upper()
    if result != text:
        # 分析字符级差异（对齐）
        maxlen = max(len(text), len(result))
        a = text.ljust(maxlen, "_")
        b = result.ljust(maxlen, "_")
        for j in range(maxlen):
            if a[j] != b[j] and a[j] != "_" and b[j] != "_":
                confusions[(a[j], b[j])] += 1
                char_errors.append((a[j], b[j]))

print("=== 字符混淆 Top 15 ===")
for (wanted, got), cnt in confusions.most_common(15):
    print("  %s -> %s : %d" % (wanted, got, cnt))

print("=== 长度不符 Top 10 ===")
len_errors = collections.Counter()
for a, b in char_errors:
    len_errors[(type(a), type(b))] += 1
print("  字符级错误总数:", len(char_errors))
