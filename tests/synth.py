"""Synthetic activity files with drift known by construction."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

# real runs wobble in pace; without it the two halves share no speeds at all
DEFAULT_WOBBLE = 0.06

TCX_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
 <Activities><Activity Sport="Running"><Id>{start}</Id><Lap StartTime="{start}"><Track>
"""
TCX_POINT = """<Trackpoint><Time>{time}</Time><Position>
<LatitudeDegrees>{lat:.7f}</LatitudeDegrees><LongitudeDegrees>8.5</LongitudeDegrees></Position>
<AltitudeMeters>{alt:.1f}</AltitudeMeters><DistanceMeters>{dist:.1f}</DistanceMeters>
<HeartRateBpm><Value>{hr}</Value></HeartRateBpm></Trackpoint>
"""
TCX_TAIL = "</Track></Lap></Activity></Activities></TrainingCenterDatabase>\n"


def _wobble(seconds, size=DEFAULT_WOBBLE, seed=7, smooth=45):
    """Seeded, smoothed multiplicative noise, one value per second."""
    if size <= 0:
        return np.ones(seconds + 1)
    raw = np.random.default_rng(seed).normal(0.0, size, seconds + 1)
    kernel = np.ones(smooth) / smooth
    padded = np.concatenate([np.full(smooth, raw[0]), raw, np.full(smooth, raw[-1])])
    smoothed = np.convolve(padded, kernel, mode="same")[smooth:smooth + seconds + 1]
    # Convolution shrinks the variance; rescale so the requested size survives.
    smoothed *= size / max(smoothed.std(), 1e-9)
    return 1.0 + smoothed


def write_tcx(path, hr_fn, speed_fn, seconds, start=None, alt_fn=None,
              wobble=DEFAULT_WOBBLE, seed=7):
    """Write a synthetic TCX.

    Args:
        path: Destination file.
        hr_fn: ``t -> bpm``.
        speed_fn: ``t -> m/s``.
        seconds: Total duration.
        start: Activity start time, defaults to a fixed instant.
        alt_fn: ``t -> metres``, defaults to flat.
        wobble: Relative size of the seeded pace noise; 0 disables it.
        seed: Noise seed, so every fixture is reproducible.
    """
    start = start or dt.datetime(2026, 1, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
    alt_fn = alt_fn or (lambda t: 400.0)
    noise = _wobble(seconds, wobble, seed)
    parts = [TCX_HEAD.format(start=start.strftime("%Y-%m-%dT%H:%M:%SZ"))]
    dist = 0.0
    for t in range(seconds + 1):
        if t:
            dist += max(0.3, speed_fn(t) * noise[t])
        parts.append(TCX_POINT.format(
            time=(start + dt.timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            lat=47.4 + dist * 1e-7,
            alt=alt_fn(t),
            dist=dist,
            hr=int(round(hr_fn(t))),
        ))
    parts.append(TCX_TAIL)
    path = str(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return path


def constant_pace_rising_hr(path, block_min=60, warmup_min=15, hr0=140.0,
                            rise_pct=6.0, speed=2.8, **kw):
    """Pace pinned, heart rate rising linearly across the block by ``rise_pct``.

    True drift is ``rise_pct`` over the block: the second half's mean heart rate
    exceeds the first half's by that fraction of the first half's mean.
    """
    total = (warmup_min + block_min) * 60
    t0 = warmup_min * 60

    def hr(t):
        if t < t0:
            return hr0
        frac = (t - t0) / (block_min * 60)
        return hr0 * (1 + rise_pct / 100 * frac)

    return write_tcx(path, hr, lambda t: speed, total, **kw)


def constant_hr_slowing_pace(path, block_min=60, warmup_min=15, hr=140.0,
                             speed0=2.8, decay_pct=6.0, **kw):
    """Heart rate pinned, pace decaying linearly across the block."""
    total = (warmup_min + block_min) * 60
    t0 = warmup_min * 60

    def spd(t):
        if t < t0:
            return speed0
        frac = (t - t0) / (block_min * 60)
        return speed0 * (1 - decay_pct / 100 * frac)

    return write_tcx(path, lambda t: hr, spd, total, **kw)


def both_rising(path, block_min=60, warmup_min=15, hr0=140.0,
                hr_rise_pct=7.0, speed0=2.5, speed_rise_pct=5.0, **kw):
    """Heart rate *and* pace both rising -- the case that fools Pa:Hr.

    True cardiac drift at matched speed is ``hr_rise_pct``, but the efficiency
    ratio cancels most of it against the speed increase.
    """
    total = (warmup_min + block_min) * 60
    t0 = warmup_min * 60

    def frac(t):
        return max(0.0, (t - t0) / (block_min * 60))

    return write_tcx(
        path,
        lambda t: hr0 * (1 + hr_rise_pct / 100 * frac(t)),
        lambda t: speed0 * (1 + speed_rise_pct / 100 * frac(t)),
        total,
        **kw,
    )


def erratic_wander(path, total_min=85, hr0=145.0, hr_swing=20.0,
                   speed0=2.7, speed_swing=0.7, period_s=3000.0, **kw):
    """Effort wandering on a period comparable to the block length.

    Different warm-up and block choices then capture different phases of the
    wander, so the drift you measure depends on where you cut. This is the shape
    the robustness sweep exists to catch: no single block is wrong, but no
    single block is trustworthy either.
    """
    return write_tcx(
        path,
        lambda t: hr0 + hr_swing * math.sin(2 * math.pi * t / period_s),
        lambda t: speed0 - speed_swing * math.sin(2 * math.pi * t / period_s + 1.1),
        int(total_min * 60),
        **kw,
    )


def rolling_hills(path, block_min=60, warmup_min=15, hr=145.0, speed=2.6,
                  amplitude_m=12.0, period_s=600.0, **kw):
    """Steady effort over sinusoidal terrain, for grade-adjustment tests."""
    total = (warmup_min + block_min) * 60
    return write_tcx(path, lambda t: hr, lambda t: speed, total,
                     alt_fn=lambda t: 400.0 + amplitude_m * math.sin(2 * math.pi * t / period_s),
                     **kw)
