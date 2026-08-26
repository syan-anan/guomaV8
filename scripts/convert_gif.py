import os
from PIL import Image
d = "H:/qinglong/syandaV8/data"
for i in range(5):
    img = Image.open(os.path.join(d, f"oschina_{i}.gif")).convert("RGB")
    img.save(f"H:/qinglong/syandaV8/data/oschina_{i}.png", "PNG")
    print(f"oschina_{i}.png saved")
print("done")
