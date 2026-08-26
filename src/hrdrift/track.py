"""TCX/GPX parsing. Produces heart-rate, speed and grade-adjusted speed
streams on a 1 Hz grid.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Track", "minetti_factor", "moving_average", "normalized_graded_pace"]

TCX_NS = {
    "t": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
    "x": "http://www.garmin.com/xmlschemas/ActivityExtension/v2",
}
GPX_NS = {
    "g": "http://www.topografix.com/GPX/1/1",
    "e": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
}

STOP_SPEED = 0.5          # m/s, below this counts as stopped
GRADE_WINDOW_M = 40.0     # distance window for the grade estimate
ALT_SMOOTH_S = 30
MAX_SPEED = 12.0          # m/s, rejects GPS spikes
NGP_WINDOW_S = 30
YARDS_PER_METRE = 1.0936133


class TrackError(ValueError):
    """Raised when a file cannot be turned into usable streams."""


# --------------------------------------------------------------------- parsing

def _parse_time(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))


def _read_tcx(path: str) -> list[dict]:
    root = ET.parse(path).getroot()
    points = []
    for tp in root.findall(".//t:Trackpoint", TCX_NS):
        when = tp.find("t:Time", TCX_NS)
        if when is None:
            continue
        hr = tp.find("t:HeartRateBpm/t:Value", TCX_NS)
        dist = tp.find("t:DistanceMeters", TCX_NS)
        alt = tp.find("t:AltitudeMeters", TCX_NS)
        lat = tp.find("t:Position/t:LatitudeDegrees", TCX_NS)
        lon = tp.find("t:Position/t:LongitudeDegrees", TCX_NS)
        points.append(
            {
                "time": _parse_time(when.text),
                "hr": float(hr.text) if hr is not None else math.nan,
                "dist": float(dist.text) if dist is not None else math.nan,
                "alt": float(alt.text) if alt is not None else math.nan,
                "lat": float(lat.text) if lat is not None else math.nan,
                "lon": float(lon.text) if lon is not None else math.nan,
            }
        )
    return points


def _read_gpx(path: str) -> list[dict]:
    root = ET.parse(path).getroot()
    points = []
    for tp in root.findall(".//g:trkpt", GPX_NS):
        when = tp.find("g:time", GPX_NS)
        if when is None:
            continue
        hr = tp.find(".//e:hr", GPX_NS)
        if hr is None:
            hr = next((el for el in tp.iter() if el.tag.rpartition("}")[2] == "hr"), None)
        ele = tp.find("g:ele", GPX_NS)
        points.append(
            {
                "time": _parse_time(when.text),
                "hr": float(hr.text) if hr is not None else math.nan,
                "dist": math.nan,
                "alt": float(ele.text) if ele is not None else math.nan,
                "lat": float(tp.get("lat")),
                "lon": float(tp.get("lon")),
            }
        )
    return points


READERS = {".tcx": _read_tcx, ".gpx": _read_gpx}


# ------------------------------------------------------------------- utilities

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, so length is preserved."""
    if window <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float).copy()
    window = min(window, len(x))
    kernel = np.ones(window) / window
    left = window // 2
    padded = np.concatenate([np.full(left, x[0]), x, np.full(window - 1 - left, x[-1])])
    return np.convolve(padded, kernel, mode="valid")


def normalized_graded_pace(gap_speed: np.ndarray, window: int = NGP_WINDOW_S) -> float:
    """Normalised graded pace in m/s.
    """
    if len(gap_speed) == 0:
        return math.nan
    rolled = moving_average(np.asarray(gap_speed, dtype=float), window)
    return float(np.mean(rolled**4) ** 0.25)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    good = ~np.isnan(x)
    if not good.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def _haversine_cumulative(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance, for files with no distance stream."""
    radius = 6371000.0
    la, lo = np.radians(_fill_nan(lat)), np.radians(_fill_nan(lon))
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    step = 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate([[0.0], np.cumsum(step)])


def minetti_factor(grade: np.ndarray | float) -> np.ndarray:
    """Cost of running at a gradient, relative to level. Clipped to +/-45%."""
    i = np.clip(grade, -0.45, 0.45)
    cost = 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6
    return cost / 3.6


def _local_grade(dist: np.ndarray, alt: np.ndarray, window_m: float) -> np.ndarray:
    """Rise over run across a fixed distance window, not per sample."""
    lo = np.searchsorted(dist, dist - window_m / 2, side="left")
    hi = np.searchsorted(dist, dist + window_m / 2, side="right") - 1
    hi = np.clip(hi, 0, len(dist) - 1)
    run = dist[hi] - dist[lo]
    rise = alt[hi] - alt[lo]
    grade = np.where(run > 1.0, rise / np.maximum(run, 1e-9), 0.0)
    return np.clip(grade, -0.45, 0.45)


# ----------------------------------------------------------------------- track

@dataclass
class Track:
    """Activity streams resampled to 1 Hz.

    Attributes:
        t: Elapsed seconds from the first sample, one entry per second.
        hr: Heart rate in bpm.
        dist: Cumulative distance in metres, monotonically non-decreasing.
        alt: Altitude in metres.
        speed: Ground speed in m/s.
        grade: Local gradient as a fraction (0.05 is a 5% climb).
        gap_speed: Grade-adjusted speed in m/s -- the speed that would cost the
            same metabolically on level ground.
        moving: Boolean mask of samples above :data:`STOP_SPEED`.
    """

    path: str
    start: dt.datetime
    t: np.ndarray
    hr: np.ndarray
    dist: np.ndarray
    alt: np.ndarray
    speed: np.ndarray
    grade: np.ndarray
    gap_speed: np.ndarray
    moving: np.ndarray
    max_gap_s: float = 0.0
    has_altitude: bool = True
    _meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Track":
        ext = os.path.splitext(path)[1].lower()
        reader = READERS.get(ext)
        if reader is None:
            raise TrackError(f"unsupported file type {ext!r}; expected one of {sorted(READERS)}")

        points = reader(path)
        if len(points) < 60:
            raise TrackError(f"only {len(points)} trackpoints in {os.path.basename(path)}")
        points.sort(key=lambda p: p["time"])

        start = points[0]["time"]
        raw_t = np.array([(p["time"] - start).total_seconds() for p in points])
        raw_hr = np.array([p["hr"] for p in points])
        raw_alt = np.array([p["alt"] for p in points])
        raw_lat = np.array([p["lat"] for p in points])
        raw_lon = np.array([p["lon"] for p in points])
        raw_dist = np.array([p["dist"] for p in points])

        if np.isnan(raw_hr).all():
            raise TrackError(f"{os.path.basename(path)} contains no heart-rate data")

        if np.isnan(raw_dist).all():
            if np.isnan(raw_lat).all():
                raise TrackError(f"{os.path.basename(path)} has neither distance nor position")
            raw_dist = _haversine_cumulative(raw_lat, raw_lon)
        raw_dist = np.maximum.accumulate(_fill_nan(raw_dist))

        has_alt = not np.isnan(raw_alt).all()

        # uniform grid; gap size kept so the audit can flag it
        t = np.arange(0.0, float(int(raw_t[-1])) + 1.0)
        max_gap = float(np.max(np.diff(raw_t))) if len(raw_t) > 1 else 0.0
        hr = np.interp(t, raw_t, _fill_nan(raw_hr))
        dist = np.interp(t, raw_t, raw_dist)
        alt = np.interp(t, raw_t, _fill_nan(raw_alt)) if has_alt else np.zeros_like(t)

        speed = np.clip(np.gradient(moving_average(dist, 10), t), 0.0, MAX_SPEED)
        grade = _local_grade(dist, moving_average(alt, ALT_SMOOTH_S), GRADE_WINDOW_M)

        return cls(
            path=path,
            start=start,
            t=t,
            hr=hr,
            dist=dist,
            alt=alt,
            speed=speed,
            grade=grade,
            gap_speed=speed * minetti_factor(grade),
            moving=speed >= STOP_SPEED,
            max_gap_s=max_gap,
            has_altitude=has_alt,
        )

    @property
    def duration_min(self) -> float:
        return float(self.t[-1]) / 60.0

    @property
    def distance_km(self) -> float:
        return float(self.dist[-1]) / 1000.0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)
