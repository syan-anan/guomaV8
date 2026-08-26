import io, random
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, 'H:/qinglong/syandaV8')
sys.path.insert(0, 'H:/qinglong/syandaV8/solver')
from master_controller import MasterController

mc = MasterController()

# 生成基础测试图片
def gen_red_rect():
    img = Image.new('RGB', (300, 100), 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 20, 100, 70], fill='#CC0000')
    buf = io.BytesIO(); img.save(buf, format='PNG'); return buf.getvalue()

def gen_blue_square():
    img = Image.new('RGB', (300, 100), 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 20, 100, 70], fill='#0000FF')
    buf = io.BytesIO(); img.save(buf, format='PNG'); return buf.getvalue()

def gen_green_ellipse():
    img = Image.new('RGB', (300, 100), 'white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 20, 150, 80], fill='#00CC00')
    buf = io.BytesIO(); img.save(buf, format='PNG'); return buf.getvalue()

print("--- V8 Quick Verification ---")
tests = [
    ('1007 (红框)', gen_red_rect()),
    ('1005 (红块)', gen_red_rect()),
    ('1006 (蓝框)', gen_blue_square()),
    ('1014 (绿椭圆)', gen_green_ellipse()),
    ('1015 (黄块)', lambda: Image.new('RGB', (300, 100), '#FFFF00').tobytes()[0:1]), # Fallback placeholder for yellow
]

# Direct engine tests
try:
    # Test 1009 (Rotation)
    r1009 = mc.solve('1009', None)
    print(f"[1009 Rotation] -> {r1009}")
    
    # Test 1011 (Long Slider)
    r1011 = mc.solve('1011', None)
    print(f"[1011 Long Slider] -> Code: {r1011.get('code')}, Trajectory len: {len(r1011.get('data', {}).get('trajectory', []))}")
except Exception as e:
    print(f"Error in optional tests: {e}")

print("Test complete.")
