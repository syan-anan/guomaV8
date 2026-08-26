import os, sys, io, random, collections
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
import cv2, numpy as np, syandaV8
from captcha.image import ImageCaptcha

class OCRV2:
    def __init__(self):
        self.odels = [syandaV8.DdddOcr(show_ad=False), syandaV8.DdddOcr(show_ad=False, beta=True)]
        self.charset_an = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        # 常见混淆组：形状相近的字符
        self.confusion_groups = [
            {"I", "L", "1"},  # I/L/1
            {"O", "0"},       # O/0
            {"1", "7"},       # 1/7
        ]

    def filter(self, text):
        return "".join(c.upper() for c in text if c.upper() in self.charset_an)

    def _versions(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        vs = [img]
        _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vs.append(cv2.cvtColor(o, cv2.COLOR_GRAY2BGR))
        a = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        vs.append(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))
        return vs

    def solve(self, data, type_code=1001):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        vs = self._versions(img)
        charset = self.charset_an
        if type_code == 1002:
            charset = set("0123456789")
        elif type_code == 1003:
            charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        candidates = []
        for v in vs:
            for o in self.odels:
                for _ in range(3):
                    try:
                        raw = o.classification(cv2.imencode(".png", v)[1].tobytes())
                        if raw:
                            candidates.append(raw.upper())
                    except:
                        pass
        # 位置级字符投票
        if not candidates:
            return ""
        # 选最长的候选（4字符）
        cands4 = [c for c in candidates if len(c) == 4]
        if not cands4:
            cands4 = candidates and [max(candidates, key=len)]
        if not cands4:
            return ""
        # 位置级投票
        result_chars = []
        for pos in range(4):
            pos_chars = []
            for c in cands4:
                if len(c) > pos:
                    ch = c[pos]
                    if ch in charset:
                        pos_chars.append(ch)
            if not pos_chars:
                pos_chars.append("?")
            # 统计该位置字符
            common = collections.Counter(pos_chars).most_common()
            # 取最高票
            best_char = common[0][0]
            best_cnt = common[0][1]
            # 如果最高票和第二高票相近，用混淆组处理
            if len(common) > 1 and (common[0][1] - common[1][1]) <= 1:
                # 检查是否属于同一混淆组
                cand_chars = {c[0] for c in common}
                for grp in self.confusion_groups:
                    if cand_chars & grp:
                        # 都属于同一组，取第一个
                        picked = sorted(cand_chars & grp)[0]
                        best.append(picked)
                        break
                else:
                    best.append(best_char)
            else:
                best.append(best_char)
        result = "".join(best)
        return result

engine = OCREngine()
gen = ImageCaptcha(width=200, height=70, fonts=[r"C:\Windows\Fonts\arial.ttf"])
correct = 0
for i in range(100):
    text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))
    data = gen.generate(text).read()
    r = engine.solve(data, 1001)
    if r == text:
        correct += 1
print("Character-vote OCR: %d/100 = %.1f%%" % (correct, 100*correct/100))
