# -*- coding: utf-8 -*-
"""Human-hand kinematic trajectory generator."""
import math
import random


def _lognormal_velocity_profile(steps):
    """Generate a velocity profile that mimics human hand movement.

    Human reaching movements follow Fitts's law / log-normal velocity profiles:
    slow start, rapid acceleration, single peak velocity, then smooth deceleration.
    """
    mu = random.uniform(0.45, 0.55)
    sigma = random.uniform(0.18, 0.28)
    profile = []
    for i in range(steps):
        t_norm = i / max(1, steps - 1)
        if t_norm <= 0:
            v = 0.0
        else:
            # lognormal PDF
            exponent = -0.5 * ((math.log(t_norm) - math.log(mu)) / sigma) ** 2
            v = (1.0 / (t_norm * sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)
        profile.append((t_norm, max(v, 0.0)))
    return profile


def _generate_human_track(distance, duration_ms=800, steps=None):
    steps = max(25, min(90, steps or distance // 2 + random.randint(5, 15)))

    # Small initial hold (reaction time)
    hold_frames = random.randint(2, 6)

    # Slight overshoot and settle
    overshoot = random.choice([0, 0, random.uniform(2, 6)])
    settle = random.choice([0, random.uniform(0.5, 2)]) if overshoot else 0
    effective_distance = distance + overshoot - settle

    profile = _lognormal_velocity_profile(steps)
    total_v = sum(v for _, v in profile)
    if total_v <= 0:
        total_v = 1e-6
    cumulative = [0.0]
    for _, v in profile:
        cumulative.append(cumulative[-1] + (v / total_v) * effective_distance)
    cumulative = cumulative[1:]

    # Y-axis wandering
    y_drift_amp = random.uniform(1.0, 4.0)
    y_drift_freq = random.uniform(0.03, 0.08)
    y_phase = random.uniform(0, math.pi * 2)

    track = []
    for i in range(hold_frames):
        track.append({"x": 0.0, "y": round(random.uniform(-0.5, 0.5), 2), "t": i * 16})

    base_time = duration_ms / steps
    prev_x = 0.0
    for i, x in enumerate(cumulative):
        jitter_y = y_drift_amp * math.sin((i + 1) * y_drift_freq + y_phase)
        jitter_y += random.gauss(0, 0.4)

        x_noisy = x + random.gauss(0, 0.3)
        if x_noisy < prev_x:
            x_noisy = prev_x + random.uniform(0, 0.2)
        prev_x = x_noisy

        timestamp = int(hold_frames * 16 + (i + 1) * base_time + random.gauss(0, base_time * 0.1))
        track.append({
            "x": round(min(x_noisy, effective_distance), 2),
            "y": round(jitter_y, 2),
            "t": max(timestamp, 0),
        })

    if track:
        track[-1]["x"] = distance
        track[-1]["y"] = round(track[-1]["y"] * 0.5, 2)

    # Micro-pause before final release
    if track and random.random() < 0.33:
        last = track[-1]
        for _ in range(random.randint(1, 3)):
            track.append({
                "x": last["x"],
                "y": round(last["y"] + random.gauss(0, 0.2), 2),
                "t": track[-1]["t"] + random.randint(16, 40),
            })

    return track


def generate_track(distance, y_offset_range=(-5, 5), steps=None):
    """Generate a human-like slide track.

    Backward-compatible interface; y_offset_range is kept for API compatibility.
    """
    if distance == 0:
        return [{"x": 0, "y": 0, "t": 0}]
    duration_ms = random.randint(650, 1300)
    track = _generate_human_track(distance, duration_ms=duration_ms, steps=steps)
    if y_offset_range and y_offset_range != (-5, 5):
        y_bias = random.uniform(y_offset_range[0], y_offset_range[1])
        for pt in track:
            pt["y"] = round(pt["y"] + y_bias * 0.2, 2)
    return track
