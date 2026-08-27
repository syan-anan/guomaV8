# -*- coding: utf-8 -*-
"""Start-independent service regression used by D3 validation.

Sends each of the 19 registered types through POST /solve. Unlike
selftest_19types.py, this exercises HTTP parsing, request routing,
solver threading, response serialization, and metrics together.
"""
import base64
import io
import json
import os
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "0")
os.chdir(ROOT)
import sys

sys.path.insert(0, str(ROOT))
import scripts.selftest_19types as samples
from solver.registry import REGISTRY


BASE_URL = os.getenv("SERVICE_SELFTEST_BASE_URL", "http://127.0.0.1:15777")
SAMPLES_PER_TYPE = 20


def image_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_payload(code):
    extra = None
    gap_image = None
    if code in (1001, 1002, 1003):
        chars = samples._random_ocr_chars(code)
        image = image_b64(samples.gen_ocr_img(chars))
        expected_text = "".join(chars)
    elif code in (1004, 1010, 1012, 1020):
        bg_img, gap_img = samples.gen_slide_pair()
        image = image_b64(bg_img)
        gap_image = image_b64(gap_img)
        expected_text = ""
    elif code in (1005, 1015, 1022):
        words = [chr(random.randint(65, 90)) for _ in range(3)]
        img, _ = samples.gen_click_word_img(words)
        image = image_b64(img)
        extra = {"words": words}
        expected_text = "".join(words)
    elif code in (1006, 1017, 1023):
        img, templates = samples.gen_click_icon_img(n_icons=3)
        image = image_b64(img)
        extra = {"icons": templates}
        expected_text = ""
    elif code == 1007:
        phrase = [chr(random.randint(65, 90)) for _ in range(3)]
        img, _ = samples.gen_click_word_img(phrase)
        image = image_b64(img)
        extra = {"phrase": "".join(phrase)}
        expected_text = "".join(phrase)
    elif code == 1008:
        img, _ = samples.gen_click_word_img(["A", "B", "C", "D"], size=(300, 200))
        image = image_b64(img)
        extra = {"question": "选出最大的"}
        expected_text = ""
    elif code in (1018, 1019):
        positions = random.sample(range(1, 10), 3)
        image = image_b64(samples.gen_nine_grid_img())
        extra = {"positions": positions}
        expected_text = ",".join(str(i) for i in positions)
    else:  # click pass types return operational points, not recognizable text
        image = image_b64(samples.gen_pass_img())
        extra = {"count": 3}
        expected_text = ""
    return {
        "type": code,
        "image": image,
        "gap_image": gap_image,
        "extra": extra,
    }, expected_text


def has_required_output(code, data):
    if not data:
        return False
    if code in (1001, 1002, 1003):
        return bool(data.get("text"))
    if code in (1004, 1010, 1012, 1020):
        return data.get("distance", 0) > 0 and len(data.get("track") or []) > 0
    if code in (1006, 1017, 1023, 1005, 1007, 1008, 1015, 1022):
        return len(data.get("points") or []) >= min(data.get("count", 3), 3)
    if code in (1016, 1021):
        return len(data.get("points") or []) >= data.get("count", 3)
    if code in (1018, 1019):
        return len(data.get("points") or []) == 3 and data.get("grid_cells")
    return False


def main():
    random.seed(20260827)
    results = []
    total_passed = 0
    started = time.time()
    with httpx.Client(timeout=30) as client:
        health = client.get(f"{BASE_URL}/health")
        health.raise_for_status()
        assert health.json()["data"]["version"] == os.getenv("APP_VERSION", "D3")
        assert health.json()["data"]["status"] == "running"

        for code in sorted(REGISTRY):
            passed = 0
            errors = []
            t_type = time.time()
            for i in range(SAMPLES_PER_TYPE):
                payload, _expected = make_payload(code)
                try:
                    resp = client.post(f"{BASE_URL}/solve", json=payload)
                    if resp.status_code != 200:
                        errors.append(f"r{i}: HTTP {resp.status_code}")
                        continue
                    body = resp.json()
                    if body.get("code") != 0 or body.get("message") != "ok":
                        errors.append(
                            f"r{i}: body={body.get('code')} msg={body.get('message')}"
                        )
                    elif has_required_output(code, body.get("data") or {}):
                        passed += 1
                    else:
                        errors.append(f"r{i}: empty or invalid output")
                except Exception as exc:
                    errors.append(f"r{i}: {type(exc).__name__}: {exc}")
            elapsed_ms = round((time.time() - t_type) * 1000 / SAMPLES_PER_TYPE, 1)
            results.append({
                "code": code,
                "name": REGISTRY[code]["name"],
                "passed": passed,
                "total": SAMPLES_PER_TYPE,
                "rate": round(passed / SAMPLES_PER_TYPE * 100, 2),
                "avg_response_ms": elapsed_ms,
                "errors": errors[:5],
            })
            print(f"{code} {REGISTRY[code]['name']}: {passed}/{SAMPLES_PER_TYPE}")
            total_passed += passed

    total_tests = len(results) * SAMPLES_PER_TYPE
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "service_base_url": BASE_URL,
        "version": "D3",
        "samples_per_type": SAMPLES_PER_TYPE,
        "type_count": len(results),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "runtime_success_rate": round(total_passed / total_tests * 100, 2),
        "elapsed_seconds": round(time.time() - started, 2),
        "results": results,
    }
    out_path = ROOT / "logs" / "service_selftest_result.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "version": report["version"],
        "total_passed": report["total_passed"],
        "total_tests": report["total_tests"],
        "rate": report["runtime_success_rate"],
        "elapsed_seconds": report["elapsed_seconds"],
        "report": str(out_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
