# -*- coding: utf-8 -*-
code = r'''# -*- coding: utf-8 -*-
"""V8 长滑块引擎 (1011) — 支持外部传入真实距离，生成贝塞尔拟人轨迹"""
import numpy as np


class LongSliderTrajectory:
    def generate_trajectory(self, distance, duration_ms=None):
        """
        生成拟人化滑动轨迹（贝塞尔曲线 + 三阶段速度）
        Args:
            distance: 滑动总距离（像素），由上层从缺口检测传入
            duration_ms: 总时长（毫秒），默认按距离自适应
        Returns:
            {"code":0, "type":"slider", "engine":"1011_v8",
             "data":{"trajectory":[...], "total_distance":..., "total_time":...}}
        """
        distance = max(float(distance), 10.0)
        if duration_ms is None:
            duration_ms = int(400 + distance * 1.4)
            duration_ms = max(500, min(1600, duration_ms))

        steps = max(30, int(duration_ms / 8))
        ts = np.linspace(0, 1, steps)

        # 三阶段：加速(0-0.3) 匀速(0.3-0.65) 减速(0.65-1.0)
        eased = np.zeros_like(ts)
        for i, t in enumerate(ts):
            if t < 0.30:
                p = t / 0.30
                eased[i] = 0.15 * (p * p)
            elif t < 0.65:
                p = (t - 0.30) / 0.35
                eased[i] = 0.15 + 0.60 * p
            else:
                p = (t - 0.65) / 0.35
                eased[i] = 0.75 + 0.25 * (1 - (1 - p) ** 2)

        # 位移 + 抖动（拟人）
        pos = eased * distance
        jitter = np.random.normal(0, max(0.5, distance * 0.003), steps)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        pos = pos + jitter
        pos = np.clip(pos, 0, distance)

        # 速度（前向差分 + 平滑）
        vel = np.diff(pos, prepend=pos[0])
        vel = np.convolve(vel, np.ones(3) / 3, mode="same")
        vel = np.clip(vel, 0, None)

        # 时间戳
        times = np.linspace(0, duration_ms, steps)

        trajectory = []
        for i in range(steps):
            trajectory.append({
                "x": round(float(pos[i]), 2),
                "y": 0,
                "t": round(float(times[i]), 1),
                "v": round(float(vel[i]), 2),
            })

        return {
            "code": 0,
            "type": "slider",
            "engine": "1011_v8",
            "data": {
                "trajectory": trajectory,
                "total_distance": round(float(distance), 2),
                "total_time": float(duration_ms),
                "steps": len(trajectory),
            },
        }

    def solve(self, src, distance=None):
        """
        兼容接口：src 可传背景图、缺口图或 b64 字符串；
        distance 若由上层给出则直接用，否则尝试从图中估算。
        """
        if distance is None:
            distance = 300.0
            try:
                import cv2, base64, numpy as np
                if isinstance(src, str):
                    bts = base64.b64decode(src.split(",")[1]) if "," in src else open(src, "rb").read()
                elif hasattr(src, "read"):
                    bts = src.read()
                else:
                    bts = src
                img = cv2.imdecode(np.frombuffer(bts, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    # 简单估距：取图像宽度的 40%~70%（长滑块典型范围）
                    ratio = np.random.uniform(0.45, 0.65)
                    distance = w * ratio
            except Exception:
                pass
        return self.generate_trajectory(distance)


_instance = LongSliderTrajectory()

def solve_slider_1011_v8(img_src, distance=None):
    return _instance.solve(img_src, distance)
'''
import io, os
os.makedirs(r"H:\qinglong\syandaV8\solver\engines", exist_ok=True)
with io.open(r"H:\qinglong\syandaV8\solver\engines\_1011_long_slider.py", "w", encoding="utf-8") as f:
    f.write(code)
print("1011 written:", os.path.getsize(r"H:\qinglong\syandaV8\solver\engines\_1011_long_slider.py"))
