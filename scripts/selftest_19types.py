# -*- coding: utf-8 -*-
"""
syandaV8 全题型独立运行自测 — 仅用项目自身文件，不依赖外部素材。
每个题型生成 20 张合成样本，调用 solver.registry.solve() 验证。
输出: 表格 + logs/selftest_result.json
"""
import os, sys, time, json, tempfile
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# --- 确保项目根目录在 sys.path ---
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# 加载 solver.registry
from solver.registry import solve

SAMPLES_PER_TYPE = 100


# ===================== 图片生成工具 =====================

def _get_font(size):
    for p in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialb.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _save(img):
    """保存 PIL 图片到临时文件，返回路径字符串（作为 solver 输入）。"""
    p = str(ROOT / "logs" / f"syanda_test_{id(img)}.png")
    img.save(p, format="PNG")
    return p


def gen_ocr_img(chars, size=(160, 48), noisy=True):
    """生成带文字的验证码图，可选加噪声。"""
    img = Image.new("RGB", size, color=(220, 225, 235))
    dr = ImageDraw.Draw(img)
    font = _get_font(36)
    for i, ch in enumerate(chars):
        x = 6 + i * (size[0] // max(len(chars), 1))
        y = size[1] // 2 - 14 + random.randint(-4, 4)
        color = tuple(random.randint(20, 80) for _ in range(3))
        dr.text((x, y), ch, font=font, fill=color)
    if noisy:
        for _ in range(random.randint(30, 80)):
            x = random.randint(0, size[0] - 1)
            y = random.randint(0, size[1] - 1)
            dr.point((x, y), fill=(random.randint(150, 200),) * 3)
        # 干扰线
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
            dr.line([(x1, y1), (x2, y2)], fill=(200, 205, 210), width=1)
    return img


def gen_slide_pair():
    """生成 (bg, gap) 图片对。gap 是 bg 中的缺口块。"""
    W, H = 340, 200
    img = Image.new("RGB", (W, H), (random.randint(180, 230),) * 3)
    dr = ImageDraw.Draw(img)
    # 画些随机矩形增加纹理
    for _ in range(random.randint(5, 15)):
        x, y = random.randint(0, W - 60), random.randint(0, H - 40)
        w, h = random.randint(20, 60), random.randint(20, 40)
        color = tuple(random.randint(100, 200) for _ in range(3))
        dr.rectangle([x, y, x + w, y + h], outline=color, width=2)

    # 缺口位置
    gap_w, gap_h = 40, 60
    gx = random.randint(120, W - gap_w - 20)
    gy = random.randint(10, H - gap_h - 10)

    # 挖缺口（画深色块）
    dr.rectangle([gx, gy, gx + gap_w, gy + gap_h], fill=(30, 30, 30))
    # 画一个拼图形状的缺口（半圆凸出）
    dr.ellipse([gx, gy + gap_h // 2 - gap_w // 2,
                gx + gap_w, gy + gap_h // 2 + gap_w // 2], fill=(30, 30, 30))

    # 截取 gap 块
    gap_img = img.crop((gx, gy, gx + gap_w, gy + gap_h))
    bg_img = img  # 整张作为背景
    return bg_img, gap_img


def gen_click_word_img(words, size=(400, 250)):
    """生成带目标词+干扰词的图片。"""
    img = Image.new("RGB", size, (250, 250, 255))
    dr = ImageDraw.Draw(img)
    font = _get_font(28)
    all_words = list(words) + [chr(random.randint(65, 90)) for _ in range(6)]
    random.shuffle(all_words)
    positions = []
    for i, word in enumerate(all_words):
        x = 20 + (i % 4) * 90
        y = 20 + (i // 4) * 80
        dr.text((x, y), word, font=font, fill=(30, 30, 120))
        positions.append((x, y, word))
    return img, positions


def gen_click_icon_img(n_icons=3, size=(400, 250)):
    """生成带图标的图片，返回 (img, templates[]) 其中 templates 是 base64。"""
    img = Image.new("RGB", size, (245, 240, 235))
    dr = ImageDraw.Draw(img)
    shapes = [
        lambda d, x, y: d.ellipse([x, y, x + 30, y + 30], fill=(200, 50, 50)),
        lambda d, x, y: d.rectangle([x, y, x + 28, y + 28], fill=(50, 150, 50)),
        lambda d, x, y: d.polygon([(x + 15, y), (x + 30, y + 30), (x, y + 30)], fill=(50, 50, 200)),
        lambda d, x, y: d.ellipse([x, y, x + 20, y + 40], fill=(180, 100, 0)),
        lambda d, x, y: d.rectangle([x, y + 10, x + 35, y + 20], fill=(100, 0, 150)),
    ]
    templates = []
    for i in range(n_icons):
        x = 30 + i * 120
        y = 30 + random.randint(0, 150)
        shapes[i % len(shapes)](dr, x, y)
        # 截取模板
        tmpl = img.crop((x, y, x + 35, y + 35))
        buf = io.BytesIO()
        tmpl.save(buf, format="PNG")
        tmp_path = str(ROOT / "logs" / f"tmpl_{i}.png")
        with open(tmp_path, "wb") as fh:
            import base64
            fh.write(buf.getvalue())
        with open(tmp_path, "rb") as fh2:
            templates.append(base64.b64encode(fh2.read()).decode())
        os.unlink(tmp_path)
    # 加几个干扰
    for _ in range(3):
        x = random.randint(20, size[0] - 40)
        y = random.randint(20, size[1] - 40)
        dr.ellipse([x, y, x + 15, y + 15], fill=(200, 200, 200))
    return img, templates


def gen_nine_grid_img(size=(300, 300)):
    """生成九宫格图片：3x3 带编号的格子。"""
    img = Image.new("RGB", size, (255, 255, 255))
    dr = ImageDraw.Draw(img)
    font = _get_font(24)
    for i in range(9):
        row, col = divmod(i, 3)
        x = col * 100
        y = row * 100
        dr.rectangle([x, y, x + 100, y + 100], outline=(0, 0, 0), width=2)
        dr.text((x + 40, y + 40), str(i + 1), font=font, fill=(0, 0, 0))
    return img


def gen_pass_img(size=(300, 200)):
    """生成有若干可点击区域的图片。"""
    img = Image.new("RGB", size, (255, 255, 255))
    dr = ImageDraw.Draw(img)
    font = _get_font(24)
    for i in range(6):
        x = 20 + i * 45
        y = 50 + random.randint(-10, 10)
        dr.text((x, y), chr(65 + i), font=font, fill=(20, 20, 100))
    return img


import random

# ===================== 各题型测试 =====================

def test_ocr(code, chars, size=(160, 48)):
    img = gen_ocr_img(chars, size=size)
    path = _save(img)
    result = solve(code, img=path)
    os.unlink(path)
    return result


def test_slide(code):
    bg_img, gap_img = gen_slide_pair()
    bg_path = _save(bg_img)
    gap_path = _save(gap_img)
    result = solve(code, bg=bg_path, gap=gap_path)
    os.unlink(bg_path)
    os.unlink(gap_path)
    return result


def test_click_word(code, words):
    img, _ = gen_click_word_img(words)
    path = _save(img)
    result = solve(code, img=path, words=words)
    os.unlink(path)
    return result


def test_click_icon(code, icons):
    img, templates = gen_click_icon_img(n_icons=len(icons))
    path = _save(img)
    result = solve(code, img=path, icons=templates)
    os.unlink(path)
    return result


def test_word_order(code, phrase):
    img, _ = gen_click_word_img(list(phrase), size=(400, 250))
    path = _save(img)
    result = solve(code, img=path, phrase=phrase)
    os.unlink(path)
    return result


def test_spatial(code, question):
    img, _ = gen_click_word_img(["A", "B", "C", "D"], size=(300, 200))
    path = _save(img)
    result = solve(code, img=path, question=question)
    os.unlink(path)
    return result


def test_nine_grid(code, pos):
    img = gen_nine_grid_img()
    path = _save(img)
    result = solve(code, img=path, pos=pos)
    os.unlink(path)
    return result


def test_click_pass(code, count):
    img = gen_pass_img()
    path = _save(img)
    result = solve(code, img=path, count=count)
    os.unlink(path)
    return result


# ===================== 主测试循环 =====================

def _random_ocr_chars(code):
    if code == 1002:
        return [str(random.randint(0, 9)) for _ in range(random.randint(4, 6))]
    elif code == 1003:
        return [chr(random.randint(65, 90)) for _ in range(random.randint(4, 6))]
    else:  # 1001 英数混合
        pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return [random.choice(pool) for _ in range(random.randint(4, 6))]


def run_type(code, name, func, params_gen, n=SAMPLES_PER_TYPE):
    """对单个题型跑 n 次，返回 (passed, total, errors)。"""
    passed = 0
    errors = []
    for i in range(n):
        params = params_gen()
        t0 = time.time()
        try:
            result = func(**params)
            elapsed = time.time() - t0
            if result.get("code") == 0:
                passed += 1
            else:
                errors.append(f"round{i}: code={result.get('code')} err={result.get('error','')}")
        except Exception as e:
            elapsed = time.time() - t0
            errors.append(f"round{i}: EXC {type(e).__name__}: {e}")
    return passed, n, errors


def main():
    print("=" * 70)
    print(f"  syandaV8 全题型独立自测  |  每题型 {SAMPLES_PER_TYPE} 样本  |  共 19 题型")
    print("=" * 70)
    t_start = time.time()

    # 预热模型
    print("\n[预热] 加载 OCR 模型...")
    try:
        from solver.ocr import preload_ocr_models
        preload_ocr_models()
        print("  OCR 模型预热完成")
    except Exception as e:
        print(f"  OCR 预热警告: {e}")

    print("[预热] 加载 CRNN63 模型...")
    try:
        from solver.crnn63_ocr import _get_session
        _get_session()
        print("  CRNN63 模型预热完成")
    except Exception as e:
        print(f"  CRNN63 预热警告: {e}")

    print("[预热] 加载 ddddocr 点选模型...")
    try:
        from solver.click import preload_click_models
        preload_click_models()
        print("  点选模型预热完成")
    except Exception as e:
        print(f"  点选预热警告: {e}")

    # 定义各题型的测试配置
    import random
    types_to_test = [
        (1001, "英数混合",     lambda: dict(func=lambda **p: test_ocr(1001, p['chars'], size=(160,48)), chars=_random_ocr_chars(1001))),
        (1002, "纯数字",       lambda: dict(func=lambda **p: test_ocr(1002, p['chars'], size=(160,48)), chars=_random_ocr_chars(1002))),
        (1003, "纯字母",       lambda: dict(func=lambda **p: test_ocr(1003, p['chars'], size=(160,48)), chars=_random_ocr_chars(1003))),
        (1004, "极验三代滑动", lambda: dict(func=lambda **p: test_slide(1004))),
        (1005, "极验三代点选(选字)", lambda: dict(func=lambda **p: test_click_word(1005, p['words']), words=[chr(random.randint(65,90)) for _ in range(3)])),
        (1006, "极验三代点选(选物)", lambda: dict(func=lambda **p: test_click_icon(1006, p['icons']), icons=["x"]*3)),
        (1007, "极验三代语序",  lambda: dict(func=lambda **p: test_word_order(1007, p['phrase']), phrase="".join(chr(random.randint(65,90)) for _ in range(3)))),
        (1008, "极验三代空间推理", lambda: dict(func=lambda **p: test_spatial(1008, p['question']), question="选出最大的")),
        (1010, "极验二代三代通用", lambda: dict(func=lambda **p: test_slide(1010))),
        (1012, "极验四代滑动",  lambda: dict(func=lambda **p: test_slide(1012))),
        (1015, "极验四代选汉字", lambda: dict(func=lambda **p: test_click_word(1015, p['words']), words=[chr(random.randint(65,90)) for _ in range(3)])),
        (1016, "极验四代点过",  lambda: dict(func=lambda **p: test_click_pass(1016, 3))),
        (1017, "极验四代点图标", lambda: dict(func=lambda **p: test_click_icon(1017, p['icons']), icons=["x"]*3)),
        (1018, "极验四代九宫格", lambda: dict(func=lambda **p: test_nine_grid(1018, p['pos']), pos=random.sample(range(1,10), 3))),
        (1019, "极验三代九宫格", lambda: dict(func=lambda **p: test_nine_grid(1019, p['pos']), pos=random.sample(range(1,10), 3))),
        (1020, "易盾滑动拼图",  lambda: dict(func=lambda **p: test_slide(1020))),
        (1021, "易盾无感(点过)", lambda: dict(func=lambda **p: test_click_pass(1021, 3))),
        (1022, "易盾点字",      lambda: dict(func=lambda **p: test_click_word(1022, p['words']), words=[chr(random.randint(65,90)) for _ in range(3)])),
        (1023, "易盾点图标",    lambda: dict(func=lambda **p: test_click_icon(1023, p['icons']), icons=["x"]*3)),
    ]

    all_results = []
    for code, name, gen_params in types_to_test:
        print(f"\n  测试 {code} — {name} ...", end=" ", flush=True)
        passed, total, errors = 0, SAMPLES_PER_TYPE, []
        latencies = []
        for i in range(total):
            p = gen_params()
            func = p.pop("func")
            t0 = time.perf_counter()
            try:
                result = func(**p)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)
                if result.get("code") == 0:
                    passed += 1
                else:
                    errors.append(f"r{i}: code={result.get('code')} {result.get('error','')}")
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)
                errors.append(f"r{i}: EXC {type(e).__name__}: {str(e)[:80]}")
        rate = passed / total * 100 if total else 0
        status = "PASS" if passed == total else (f"PARTIAL({passed}/{total})" if passed > 0 else "FAIL")
        print(f"[{status}]")
        for e in errors[:3]:
            print(f"    {e}")
        all_results.append({
            "code": code, "name": name, "passed": passed,
            "total": total, "rate": round(rate, 1), "errors": errors,
            "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if latencies else 0,
        })

    # 汇总
    total_tests = sum(r["total"] for r in all_results)
    total_passed = sum(r["passed"] for r in all_results)
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print(f"  汇总: {total_passed}/{total_tests} 通过  |  总耗时 {elapsed:.1f}s")
    print("=" * 70)
    print(f"  {'代码':>5}  {'题型':<20} {'通过':>5} {'总数':>5} {'通过率':>7} {'平均ms':>9} {'P95ms':>9}")
    print(f"  {'-'*5}  {'-'*20} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*9}")
    for r in all_results:
        print(f"  {r['code']:>5}  {r['name']:<20} {r['passed']:>4}/{r['total']:<3} {r['rate']:>6.1f}% {r['avg_ms']:>9.2f} {r['p95_ms']:>9.2f}")
    print("=" * 70)

    # 保存 JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_per_type": SAMPLES_PER_TYPE,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_rate": round(total_passed / total_tests * 100, 2) if total_tests else 0,
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
    }
    out_path = ROOT / "logs" / "selftest_result.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)
    main()
