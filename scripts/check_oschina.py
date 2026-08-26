import os
from PIL import Image
d = "H:/qinglong/syandaV8/data"
for f in sorted(os.listdir(d)):
    if f.startswith("oschina"):
        img = Image.open(os.path.join(d, f))
        print(f"{f}: {img.size} {img.mode} {img.format}")
