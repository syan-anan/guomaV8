# -*- coding: utf-8 -*-
import os
p = r"H:\qinglong\syandaV8\solver\_gen_test.txt"
with open(p, "w", encoding="utf-8") as f:
    f.write("test ok 中文")
print("written:", os.path.exists(p))
