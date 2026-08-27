import ddddocr
import os, sys, io, random
os.chdir("H:\\qinglong\\syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:\\qinglong\\syandaV8\\__cache"
sys.path.insert(0, ".")
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_font(size=32):
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)

img = Image.new("RGB", (200, 70), (245,)*3)
d = ImageDraw.Draw(img)
font = get_font(32)
# 直接在画布上画字符，不旋转
x = 10
for ch in "42B7":
    d.text((x, 30), ch, fill=(40,40,40), font=font)
    x += 40
img.save(r"H:\qinglong\syandaV8\__cache\check.png")
print("saved check image")
# 识别
buf = io.BytesIO(); img.save(buf, "PNG")
data = buf.getvalue()
ocr = ddddocr.DdddOcr(show_ad=False)
r1 = ocr.classification(data)
r2 = ocr.classification(data)
print("r1:", repr(r1))
print("r2:", repr(r2))
