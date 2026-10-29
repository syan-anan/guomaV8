# -*- coding: utf-8 -*-
"""生物轨迹生成器 v2 — 在 trajectory.py 基础上增加微抖动层和分段噪声注入"""
import math, random

def generate_track(distance, y_offset_range=(-5, 5), steps=None):
    if distance == 0:
        return [{"x": 0, "y": 0, "t": 0}]
    duration_ms = random.randint(650, 1300)
    n_steps = max(40, min(90, steps or int(distance // 2) + random.randint(8, 15)))
    
    # Log-normal velocity profile (保留原逻辑)
    mu = random.uniform(0.45, 0.55)
    sigma = random.uniform(0.18, 0.28)
    profile = []
    for i in range(n_steps):
        t_norm = i / max(1, n_steps - 1)
        if t_norm <= 0:
            v = 0.0
        else:
            exponent = -0.5 * ((math.log(t_norm) - math.log(mu)) / sigma) ** 2
            v = (1.0 / (t_norm * sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)
        profile.append(max(v, 0.0))
    
    total_v = sum(profile) or 1e-6
    cum_dist = [0.0]
    for v in profile:
        cum_dist.append(cum_dist[-1] + (v / total_v) * distance)
    
    # 加减速区域参数
    hold_frames = random.randint(2, 6)
    overshoot_amt = random.choice([0, 0, random.uniform(1, 4)])
    effective_dist = distance + overshoot_amt
    
    # Y 轴漂移
    drift_amp = random.uniform(1.0, 4.0)
    drift_freq = random.uniform(0.03, 0.08)
    drift_phase = random.uniform(0, math.pi * 2)
    
    track = []
    for i in range(hold_frames):
        track.append({
            "x": 0.0, 
            "y": round(random.uniform(-0.5, 0.5), 2), 
            "t": i * 16
        })
    
    base_time = duration_ms / n_steps
    prev_x = 0.0
    
    for i in range(n_steps):
        raw_x = cum_dist[i + 1] if i + 1 < len(cum_dist) else cum_dist[-1]
        progress = i / max(1, n_steps - 1)
        
        # X 轴微抖动（前端慢时抖得厉害，匀速段减小）
        jitter_x = random.gauss(0, 0.5 * (1 - progress * 0.5))
        x_raw = raw_x + jitter_x
        
        # 前进约束
        if x_raw < prev_x:
            x_raw = prev_x + random.uniform(0, 0.1)
        prev_x = x_raw
        
        # Y 轴正弦漂移 + 高斯噪声
        jitter_y = drift_amp * math.sin((i + 1) * drift_freq + drift_phase)
        jitter_y += random.gauss(0, 0.4)
        
        timestamp = int(hold_frames * 16 + (i + 1) * base_time + random.gauss(0, base_time * 0.1))
        track.append({
            "x": round(min(x_raw, effective_dist), 2),
            "y": round(jitter_y, 2),
            "t": max(timestamp, 0),
        })
    
    # 最终位置修正
    if track:
        track[-1]["x"] = distance
        track[-1]["y"] = round(track[-1]["y"] * 0.5, 2)
    
    # 微停顿机制（匀速段随机插入 10-30ms 暂停）
    if 0.25 < progress < 0.70 and random.random() < 0.06:
        last_x = track[-1]["x"]
        pause_ms = random.uniform(10, 30)
        track.append({
            "x": last_x,
            "y": round(last_x * 0 + random.gauss(0, 0.1), 2),
            "t": track[-1]["t"] + int(pause_ms),
        })
    
    # 收尾微抖动
    if random.random() < 0.33:
        for _ in range(random.randint(1, 3)):
            track.append({
                "x": round(distance + random.gauss(0, 0.2), 2),
                "y": round(track[-1]["y"] + random.gauss(0, 0.15), 2),
                "t": track[-1]["t"] + random.randint(12, 28),
            })
    
    # 应用全局 Y 偏移补偿
    if y_offset_range and y_offset_range != (-5, 5):
        bias = random.uniform(y_offset_range[0], y_offset_range[1])
        for pt in track:
            pt["y"] = round(pt["y"] + bias * 0.2, 2)
    
    return track
