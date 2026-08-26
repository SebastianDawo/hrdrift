"""Drift metrics and validity checks.

pa_hr           efficiency factor compared between the two halves
speed_matched   heart rate compared within matched speed bins
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .track import STOP_SPEED, YARDS_PER_METRE, Track, moving_average, normalized_graded_pace

__all__ = [
    "Analysis", "Check", "DriftMetric", "HalfStats", "Stability",
    "analyse", "find_steadiest_block",
]

DEFAULT_TARGET = 5.0     # %/hour; above this the effort was above AeT
SPEED_BIN_MS = 0.05      # width of the speed-matching bins
BIN_EPSILON = 1e-9       # 3.0/0.05 == 59.999999999999993, so nudge before floor
MIN_BIN_SAMPLES = 25     # per half, for a bin to count
DISAGREEMENT_PTS = 2.0        # gap at which the two metrics disagree
MIN_MATCHED_COVERAGE = 40.0   # % of block needed inside matched bins
PINNED_SLOPE_BPH = 5.0        # HR counts as held below this trend
MIN_BLOCK_MIN = 15.0          # below this, don't even try

# validity gates
VALID_BLOCK_MIN = 45.0
VALID_MAX_STOPPED_PCT = 15.0
VALID_MAX_SPEED_CV = 15.0     # 60 s pace variation
VALID_MAX_HR_SPREAD = 35.0

Level = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    """One protocol compliance finding."""
    level: Level
    message: str


@dataclass(frozen=True)
class HalfStats:
    """Aggregate statistics for one half of the test block."""
    n: int
    hr: float
    speed: float
    gap_speed: float
    ngp_speed: float = math.nan
    distance_m: float = 0.0

    @property
    def ef(self) -> float:
        """Efficiency factor from NGP, in m/s per bpm."""
        speed = self.ngp_speed if np.isfinite(self.ngp_speed) else self.gap_speed
        return speed / self.hr if self.hr > 0 else math.nan

    @property
    def ef_mean_gap(self) -> float:
        """Efficiency factor from the plain mean of grade-adjusted speed."""
        return self.gap_speed / self.hr if self.hr > 0 else math.nan

    @property
    def ef_trainingpeaks(self) -> float:
        """Efficiency factor in TrainingPeaks' units: yards per minute per bpm."""
        return self.ef * 60.0 * YARDS_PER_METRE

    @property
    def ngp_pace(self) -> float:
        """Normalised graded pace in minutes per kilometre."""
        return 1000.0 / self.ngp_speed / 60.0 if self.ngp_speed > 0 else math.nan

    @property
    def pace(self) -> float:
        """Minutes per kilometre."""
        return 1000.0 / self.speed / 60.0 if self.speed > 0 else math.nan

    @property
    def gap_pace(self) -> float:
        return 1000.0 / self.gap_speed / 60.0 if self.gap_speed > 0 else math.nan


@dataclass(frozen=True)
class Stability:
    """How steadily the effort was held."""
    mean: float
    sd: float
    cv: float
    slope_bph: float
    within_3bpm: float
    within_5bpm: float

    @property
    def pinned(self) -> bool:
        """True when heart rate was held steady: within 5 bpm and under 5 bpm/h."""
        return self.within_5bpm >= 60.0 and abs(self.slope_bph) <= PINNED_SLOPE_BPH


@dataclass(frozen=True)
class DriftMetric:
    """A drift measurement and its per-hour equivalent."""
    pct: float
    block_min: float
    label: str
    detail: str = ""
    coverage: float = 100.0

    @property
    def rate(self) -> float:
        """Drift in percent per hour, comparable across block lengths."""
        return self.pct * 60.0 / self.block_min if self.block_min > 0 else math.nan

    @property
    def usable(self) -> bool:
        """False when too little of the block fell inside matched speed bins."""
        return self.coverage >= MIN_MATCHED_COVERAGE


@dataclass
class Analysis:
    """Everything computed for one heart-rate drift test."""

    track: Track
    t_start: float
    t_end: float
    first: HalfStats
    second: HalfStats
    whole: HalfStats
    stability: Stability
    pa_hr: DriftMetric
    speed_matched: Optional[DriftMetric]
    checks: list[Check]
    target: float
    stopped_pct: float
    mean_grade_pct: float
    rolling: list[tuple[float, float, float]]
    speed_cv: float = math.nan
    hr_spread: float = math.nan
    sweep: Optional["Sweep"] = None

    @property
    def block_min(self) -> float:
        return (self.t_end - self.t_start) / 60.0

    @property
    def matched_usable(self) -> bool:
        return self.speed_matched is not None and self.speed_matched.usable

    @property
    def primary(self) -> DriftMetric:  # noqa: D401
        """Whichever metric suits how the effort was held.

        Heart rate pinned -> Pa:Hr, since the drift is in the pace decay.
        Heart rate free -> speed-matched, if the halves overlap in speed.
        """
        if self.stability.pinned or not self.matched_usable:
            return self.pa_hr
        return self.speed_matched

    @property
    def metrics_disagree(self) -> bool:
        """True when the chosen metric materially contradicts the other one."""
        if self.speed_matched is None or self.primary is self.pa_hr:
            return False
        return abs(self.speed_matched.rate - self.pa_hr.rate) > DISAGREEMENT_PTS

    @property
    def reliable(self) -> bool:
        return not any(c.level == "fail" for c in self.checks)

    @property
    def usability(self) -> Usability:
        """Run the validity gates. See :class:`Usability`."""
        reasons: list[str] = []

        if self.block_min < VALID_BLOCK_MIN:
            reasons.append(
                f"The test block is {self.block_min:.0f} min. A drift test needs at least "
                f"{VALID_BLOCK_MIN:.0f} min of steady effort after the warm-up, and the 5% "
                f"criterion is defined for 60. Record a longer run.")

        if self.stopped_pct > VALID_MAX_STOPPED_PCT:
            reasons.append(
                f"{self.stopped_pct:.0f}% of the block was stopped or walking. Drift needs a "
                f"continuous effort.")

        if np.isfinite(self.speed_cv) and self.speed_cv > VALID_MAX_SPEED_CV:
            reasons.append(
                f"Pace varied by {self.speed_cv:.0f}% across the block. The test needs a "
                f"steady effort.")

        if np.isfinite(self.hr_spread) and self.hr_spread > VALID_MAX_HR_SPREAD:
            reasons.append(
                f"Heart rate ranged over {self.hr_spread:.0f} bpm within the block. That is not "
                f"one sustained intensity.")

        if self.sweep_speaks and self.sweep.agreement == "unstable":
            lo, hi = self.sweep.rates.min(), self.sweep.rates.max()
            reasons.append(
                f"The answer depends on where the block is drawn: drift ranges from {lo:+.0f} to "
                f"{hi:+.0f}%/h across reasonable warm-up and block choices, a wider spread than "
                f"the {self.target:.0f}%/h threshold itself. The effort was too variable to read "
                f"a threshold from.")

        return Usability(usable=not reasons, reasons=reasons)

    @property
    def answer(self) -> str:
        """The one-line result: the question the test is asked to settle."""
        u = self.usability
        if not u.usable:
            return "NOT A VALID DRIFT TEST"
        rate = self.sweep.median if self.sweep_speaks else self.primary.rate
        side = "ABOVE" if rate > self.target else "BELOW"
        return f"DRIFT {rate:+.1f}%/h  -  {side} the {self.target:.0f}%/h threshold"

    @property
    def result_line(self) -> str:
        """Plain-English result, naming the heart rate the answer applies to."""
        if not self.usability.usable:
            return "This run is not a drift test, so no threshold can be read from it."
        rate = self.sweep.median if self.sweep_speaks else self.primary.rate
        hr = self.whole.hr
        if rate > self.target:
            return (f"{hr:.0f} bpm was above your aerobic threshold. "
                    f"Retest around {hr - 6:.0f} bpm.")
        return (f"{hr:.0f} bpm was at or below your aerobic threshold. "
                f"Retest around {hr + 5:.0f} bpm to find the ceiling.")

    @property
    def sweep_speaks(self) -> bool:
        """True when the sweep saw enough block choices to be worth deferring to."""
        return self.sweep is not None and self.sweep.n >= 6

    @property
    def verdict(self) -> tuple[str, str]:
        """Headline verdict. Uses the sweep when there is one."""
        if self.sweep_speaks:
            return self.sweep.verdict
        return self.single_verdict

    @property
    def single_verdict(self) -> tuple[str, str]:
        """Verdict from this block alone, ignoring the sweep."""
        rate, target = self.primary.rate, self.target
        if rate > target:
            return ("ABOVE AeT", f"Drift of {rate:.1f}%/h exceeds the {target:.0f}%/h threshold. "
                                 f"The heart rate held was above the aerobic threshold. Retest lower.")
        if rate < target * 0.4:
            return ("WELL BELOW AeT", f"Drift of {rate:.1f}%/h is far under the {target:.0f}%/h "
                                      f"threshold. Retest meaningfully higher.")
        return ("AT OR JUST BELOW AeT", f"Drift of {rate:.1f}%/h is under the {target:.0f}%/h "
                                        f"threshold. This heart rate is at or just below AeT.")

    @property
    def aet_estimate(self) -> str:
        hr = self.whole.hr
        if self.sweep_speaks and self.sweep.agreement == "unstable":
            return (f"No estimate. Drift depends more on where the block is drawn than on the "
                    f"effort itself, so this file cannot place AeT. Rerun a controlled test.")
        rate = self.sweep.median if self.sweep_speaks else self.primary.rate
        if rate <= self.target:
            return (f"AeT is at or above {hr:.0f} bpm. Retest 3-5 bpm higher until drift "
                    f"crosses {self.target:.0f}%/h.")
        return (f"AeT is below {hr:.0f} bpm. Retest holding roughly {self.first.hr - 5:.0f} bpm.")


# ----------------------------------------------------------------- computation

def _half_stats(track: Track, mask: np.ndarray) -> HalfStats:
    idx = np.flatnonzero(mask)
    return HalfStats(
        n=int(mask.sum()),
        hr=float(np.mean(track.hr[mask])),
        speed=float(np.mean(track.speed[mask])),
        gap_speed=float(np.mean(track.gap_speed[mask])),
        ngp_speed=normalized_graded_pace(track.gap_speed[mask]),
        distance_m=float(track.dist[idx[-1]] - track.dist[idx[0]]) if len(idx) else 0.0,
    )


def _split(track: Track, t_start: float, t_end: float, drop_stops: bool):
    """Return (block mask, first-half mask, second-half mask). Split by moving time."""
    in_block = (track.t >= t_start) & (track.t < t_end)
    mask = in_block & track.moving if drop_stops else in_block
    idx = np.flatnonzero(mask)
    if len(idx) < 120:
        raise ValueError("fewer than two minutes of usable data in the block")
    half = len(idx) // 2
    first = np.zeros_like(mask)
    second = np.zeros_like(mask)
    first[idx[:half]] = True
    second[idx[-half:]] = True
    return mask, first, second, in_block


def _stability(track: Track, mask: np.ndarray) -> Stability:
    hr = track.hr[mask]
    mean = float(hr.mean())
    slope = float(np.polyfit(track.t[mask], hr, 1)[0] * 3600.0)
    return Stability(
        mean=mean,
        sd=float(hr.std()),
        cv=float(hr.std() / mean * 100) if mean else math.nan,
        slope_bph=slope,
        within_3bpm=float(np.mean(np.abs(hr - mean) <= 3) * 100),
        within_5bpm=float(np.mean(np.abs(hr - mean) <= 5) * 100),
    )


def _pa_hr(first: HalfStats, second: HalfStats, block_min: float) -> DriftMetric:
    return DriftMetric(
        pct=(first.ef - second.ef) / first.ef * 100.0,
        block_min=block_min,
        label="Pa:Hr decoupling",
        detail="efficiency factor, first half vs second",
    )


def _speed_matched(track: Track, first_mask: np.ndarray, second_mask: np.ndarray,
                   base_hr: float, block_min: float) -> Optional[DriftMetric]:
    """Compare heart rate between halves inside matched speed bins."""
    i1, i2 = np.flatnonzero(first_mask), np.flatnonzero(second_mask)
    v1, v2 = track.gap_speed[i1], track.gap_speed[i2]
    h1, h2 = track.hr[i1], track.hr[i2]
    if len(v1) == 0 or len(v2) == 0:
        return None

    # anchored at zero so bin edges don't move when the range changes
    b1 = np.floor(v1 / SPEED_BIN_MS + BIN_EPSILON).astype(int)
    b2 = np.floor(v2 / SPEED_BIN_MS + BIN_EPSILON).astype(int)

    weighted_sum = weight = 0.0
    used_lo, used_hi, used_n = math.inf, -math.inf, 0
    for b in np.unique(b1):
        s1, s2 = b1 == b, b2 == b
        n1, n2 = int(s1.sum()), int(s2.sum())
        if n1 < MIN_BIN_SAMPLES or n2 < MIN_BIN_SAMPLES:
            continue
        w = float(min(n1, n2))
        weighted_sum += w * (h2[s2].mean() - h1[s1].mean())
        weight += w
        used_lo = min(used_lo, float(b) * SPEED_BIN_MS)
        used_hi = max(used_hi, float(b + 1) * SPEED_BIN_MS)
        used_n += n1 + n2

    if weight == 0 or base_hr <= 0:
        return None

    delta = weighted_sum / weight
    coverage = used_n / (len(i1) + len(i2)) * 100.0
    return DriftMetric(
        pct=delta / base_hr * 100.0,
        block_min=block_min,
        label="speed-matched drift",
        detail=(f"{delta:+.1f} bpm at equal grade-adjusted speed "
                f"({used_lo:.2f}-{used_hi:.2f} m/s, {coverage:.0f}% of block)"),
        coverage=coverage,
    )


def _core(track: Track, t_start: float, t_end: float, drop_stops: bool):
    """Halves, stability and both drift metrics for one range."""
    block_min = (t_end - t_start) / 60.0
    mask, first_mask, second_mask, in_block = _split(track, t_start, t_end, drop_stops)
    first = _half_stats(track, first_mask)
    second = _half_stats(track, second_mask)
    whole = _half_stats(track, mask)
    stability = _stability(track, mask)
    pa_hr = _pa_hr(first, second, block_min)
    matched = _speed_matched(track, first_mask, second_mask, first.hr, block_min)
    return dict(mask=mask, in_block=in_block, first=first, second=second, whole=whole,
                stability=stability, pa_hr=pa_hr, matched=matched, block_min=block_min)


def _select_primary(stability: Stability, pa_hr: DriftMetric,
                    matched: Optional[DriftMetric]) -> DriftMetric:
    """Apply the metric-selection rule. See :attr:`Analysis.primary`."""
    if stability.pinned or matched is None or not matched.usable:
        return pa_hr
    return matched


@dataclass(frozen=True)
class Usability:
    """Whether the file is a drift test at all."""
    usable: bool
    reasons: list[str]

    @property
    def headline(self) -> str:
        return "VALID DRIFT TEST" if self.usable else "NOT A VALID DRIFT TEST"


@dataclass(frozen=True)
class SweepPoint:
    """One block choice and the drift it produced."""
    start_min: float
    block_min: float
    rate: float
    hr: float
    pinned: bool
    used_matched: bool


@dataclass
class Sweep:
    """Drift across many block boundaries. The spread says how solid the answer is."""
    points: list[SweepPoint]
    target: float

    @property
    def rates(self) -> np.ndarray:
        return np.array([p.rate for p in self.points])

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def median(self) -> float:
        return float(np.median(self.rates)) if self.n else math.nan

    @property
    def p25(self) -> float:
        return float(np.percentile(self.rates, 25)) if self.n else math.nan

    @property
    def p75(self) -> float:
        return float(np.percentile(self.rates, 75)) if self.n else math.nan

    @property
    def spread(self) -> float:
        return self.p75 - self.p25 if self.n else math.nan

    @property
    def fraction_above(self) -> float:
        """Share of block choices that put the effort above threshold."""
        return float(np.mean(self.rates > self.target) * 100) if self.n else math.nan

    UNSTABLE_SPREAD = 5.0    # wider than the threshold itself: no verdict
    TIGHT_SPREAD = 2.0
    AGREEMENT_SHARE = 75.0   # % of block choices needed to call a direction

    @property
    def agreement(self) -> str:
        """How consistently the block choices tell the same story."""
        if self.n == 0:
            return "none"
        if self.spread > self.UNSTABLE_SPREAD:
            return "unstable"
        return "tight" if self.spread < self.TIGHT_SPREAD else "moderate"

    @property
    def verdict(self) -> tuple[str, str]:
        """Verdict over all block choices, not from one arbitrary boundary."""
        if self.n == 0:
            return ("NO VERDICT", "No block choice produced a usable analysis.")
        share, med = self.fraction_above, self.median
        iqr = f"IQR {self.p25:.1f}-{self.p75:.1f}"

        if self.spread > self.UNSTABLE_SPREAD:
            return ("UNSTABLE", f"Moving the warm-up or block length swings drift across "
                                f"{self.spread:.1f} points ({iqr}), wider than the "
                                f"{self.target:.0f}%/h threshold itself. The effort was too "
                                f"variable to read a threshold from. No verdict.")
        if share >= self.AGREEMENT_SHARE:
            return ("ABOVE AeT", f"{share:.0f}% of {self.n} block choices exceed "
                                 f"{self.target:.0f}%/h (median {med:.1f}, {iqr}). The heart rate "
                                 f"held was above AeT. Retest lower.")
        if share <= 100 - self.AGREEMENT_SHARE:
            return ("BELOW AeT", f"{share:.0f}% of {self.n} block choices exceed "
                                 f"{self.target:.0f}%/h (median {med:.1f}, {iqr}). The heart rate "
                                 f"held was at or below AeT. Retest higher.")
        return ("BORDERLINE", f"{share:.0f}% of {self.n} block choices land above "
                              f"{self.target:.0f}%/h and the rest below (median {med:.1f}, {iqr}). "
                              f"The answer sits on the threshold; this test cannot separate them.")


def robustness_sweep(track: Track, target: float = DEFAULT_TARGET,
                     starts: tuple[float, ...] = (8, 10, 12, 15, 18, 20),
                     blocks: tuple[float, ...] = (40, 45, 50, 55, 60),
                     drop_stops: bool = True) -> Sweep:
    """Evaluate drift across every warm-up and block-length combination."""
    points: list[SweepPoint] = []
    for start in starts:
        for block in blocks:
            t_start, t_end = start * 60.0, (start + block) * 60.0
            if t_end > track.t[-1] + 1 or (t_end - t_start) / 60.0 < MIN_BLOCK_MIN:
                continue
            try:
                c = _core(track, t_start, t_end, drop_stops)
            except ValueError:
                continue
            metric = _select_primary(c["stability"], c["pa_hr"], c["matched"])
            points.append(SweepPoint(
                start_min=float(start), block_min=float(block), rate=metric.rate,
                hr=c["whole"].hr, pinned=c["stability"].pinned,
                used_matched=metric is c["matched"],
            ))
    return Sweep(points=points, target=target)


def rolling_drift(track: Track, block_min: float = 20.0, step_s: int = 120,
                  drop_stops: bool = True) -> list[tuple[float, float, float]]:
    """Sliding-window drift. Returns (start_seconds, mean_hr, rate_pct_per_hour)."""
    out = []
    width = block_min * 60
    for start in range(0, int(track.t[-1] - width) + 1, step_s):
        try:
            mask, first, second, _ = _split(track, start, start + width, drop_stops)
        except ValueError:
            continue
        whole = _half_stats(track, mask)
        metric = _speed_matched(track, first, second, _half_stats(track, first).hr, block_min)
        if metric is None:
            continue
        out.append((float(start), whole.hr, metric.rate))
    return out


def find_steadiest_block(track: Track, duration_min: float, drop_stops: bool = True):
    """Start time of the block with the most stable heart rate, or None."""
    width = duration_min * 60
    if track.t[-1] < width:
        return None
    best = None
    for start in range(0, int(track.t[-1] - width) + 1, 30):
        try:
            mask, *_ = _split(track, start, start + width, drop_stops)
        except ValueError:
            continue
        cv = _stability(track, mask).cv
        if best is None or cv < best[0]:
            best = (cv, float(start))
    return best[1] if best else None


# --------------------------------------------------------------------- checks

def _audit(track: Track, analysis_inputs: dict, target: float) -> list[Check]:
    a = analysis_inputs
    checks: list[Check] = []
    warmup_min, duration_min = a["warmup_min"], a["duration_min"]
    needed = warmup_min + duration_min
    st: Stability = a["stability"]

    if track.duration_min + 0.5 < needed:
        checks.append(Check("fail", f"Recording is {track.duration_min:.0f} min; the protocol needs "
                                    f"{warmup_min:.0f} min warm-up plus {duration_min:.0f} min test "
                                    f"= {needed:.0f} min."))
    else:
        checks.append(Check("ok", f"Recording length {track.duration_min:.0f} min covers the "
                                  f"{needed:.0f} min protocol."))

    block = a["block_min"]
    if block < 45:
        checks.append(Check("fail", f"Analysis block is only {block:.0f} min. Below about 45 min the "
                                    f"drift signal is small next to the noise; 60 min is prescribed."))
    elif block < 60:
        checks.append(Check("warn", f"Analysis block is {block:.0f} min, short of the prescribed 60."))
    else:
        checks.append(Check("ok", f"Analysis block is {block:.0f} min."))

    hr_free = not st.pinned
    pace_change = abs(a["pace_change_pct"])
    if hr_free and pace_change > 2:
        checks.append(Check("fail", f"Neither variable was held constant: heart rate moved "
                                    f"{st.slope_bph:+.0f} bpm/h and grade-adjusted pace changed "
                                    f"{a['pace_change_pct']:+.1f}% between halves. The test requires "
                                    f"pinning one of the two."))
    elif hr_free:
        checks.append(Check("warn", f"Heart rate was not pinned ({st.slope_bph:+.0f} bpm/h), but pace "
                                    f"held to {a['pace_change_pct']:+.1f}%, so this reads as a valid "
                                    f"constant-pace variant of the test."))
    elif st.within_3bpm < 60:
        checks.append(Check("warn", f"Heart rate held within 3 bpm only {st.within_3bpm:.0f}% of the "
                                    f"time ({st.slope_bph:+.0f} bpm/h)."))
    else:
        checks.append(Check("ok", f"Heart rate steady: {st.within_3bpm:.0f}% of samples within 3 bpm."))

    if a["disagree"]:
        checks.append(Check("fail", f"The drift metrics disagree: Pa:Hr says {a['pa_hr_rate']:+.1f}%/h "
                                    f"but speed-matched heart rate says {a['matched_rate']:+.1f}%/h. "
                                    f"Heart rate was not pinned, so the speed-matched figure is the "
                                    f"one being reported."))
    if a["matched_coverage"] is not None and a["matched_coverage"] < MIN_MATCHED_COVERAGE and not st.pinned:
        checks.append(Check("warn", f"Only {a['matched_coverage']:.0f}% of the block fell inside speed "
                                    f"bins present in both halves, so the speed-matched cross-check is "
                                    f"weak here and Pa:Hr is reported instead."))

    if a["stopped_pct"] > 5:
        checks.append(Check("warn", f"{a['stopped_pct']:.0f}% of the block was below {STOP_SPEED} m/s. "
                                    f"Excluded from pace, but interruptions blunt the drift signal."))
    else:
        checks.append(Check("ok", f"Essentially continuous ({a['stopped_pct']:.1f}% stopped)."))

    if not track.has_altitude:
        checks.append(Check("warn", "File carries no altitude, so pace is not grade-adjusted."))
    elif abs(a["mean_grade_pct"]) > 2 or a["grade_spread_pct"] > 12:
        checks.append(Check("warn", f"Terrain varies (mean grade {a['mean_grade_pct']:+.1f}%). Pace is "
                                    f"grade-adjusted, but a treadmill or flat loop is cleaner."))
    else:
        checks.append(Check("ok", f"Terrain effectively flat (mean grade {a['mean_grade_pct']:+.1f}%)."))

    if track.max_gap_s > 5:
        checks.append(Check("warn", f"Recording has a {track.max_gap_s:.0f} s gap, interpolated across."))

    return checks


# ----------------------------------------------------------------- entry point

def analyse(track: Track, warmup_min: float = 15.0, duration_min: float = 60.0,
            target: float = DEFAULT_TARGET, drop_stops: bool = True,
            auto: bool = False, rolling_min: float = 20.0,
            t_from: float | None = None, t_to: float | None = None,
            sweep: bool = True) -> Analysis:
    """Analyse one heart-rate drift test.

    Args:
        track: A loaded :class:`~hrdrift.track.Track`.
        warmup_min: Minutes to exclude at the start.
        duration_min: Length of the test block.
        target: Drift threshold in percent per hour.
        drop_stops: Exclude samples below walking speed from pace.
        auto: Ignore ``warmup_min`` and use the steadiest block that fits.
        rolling_min: Window width for the rolling drift table.
        t_from: Explicit selection start in minutes, as with dragging a range
            in TrainingPeaks. Overrides ``warmup_min``.
        t_to: Explicit selection end in minutes. Overrides ``duration_min``.
        sweep: Also evaluate every plausible block boundary, so the verdict
            reflects the spread rather than one arbitrary choice.

    Raises:
        ValueError: If no usable block can be formed.
    """
    if t_from is not None or t_to is not None:
        t_start = (t_from or 0.0) * 60.0
        end = (t_to * 60.0) if t_to is not None else float(track.t[-1])
        duration_min = (end - t_start) / 60.0
        auto = False
    elif auto:
        for candidate in (duration_min, 45.0, 40.0, 30.0):
            start = find_steadiest_block(track, candidate, drop_stops)
            if start is not None:
                t_start, duration_min = start, candidate
                break
        else:
            raise ValueError("recording too short for any analysis block")
    else:
        t_start = warmup_min * 60.0

    t_end = min(t_start + duration_min * 60.0, float(track.t[-1]))
    block_min = (t_end - t_start) / 60.0
    if block_min < MIN_BLOCK_MIN:
        raise ValueError(
            f"analysis block is only {block_min:.0f} min; at least {MIN_BLOCK_MIN:.0f} min is "
            f"needed for drift to mean anything. Lower --warmup or use a longer recording."
        )
    mask, first_mask, second_mask, in_block = _split(track, t_start, t_end, drop_stops)

    first = _half_stats(track, first_mask)
    second = _half_stats(track, second_mask)
    whole = _half_stats(track, mask)
    stability = _stability(track, mask)
    pa_hr = _pa_hr(first, second, block_min)
    matched = _speed_matched(track, first_mask, second_mask, first.hr, block_min)

    grade = track.grade[mask] * 100
    stopped_pct = float(1 - mask.sum() / max(1, in_block.sum())) * 100
    smoothed = moving_average(track.gap_speed[mask], 60)
    speed_cv = float(smoothed.std() / max(smoothed.mean(), 1e-9) * 100)
    block_hr = track.hr[mask]
    hr_spread = float(np.percentile(block_hr, 95) - np.percentile(block_hr, 5))
    pace_change = (second.gap_pace - first.gap_pace) / first.gap_pace * 100

    disagree = (
        matched is not None
        and matched.usable
        and not stability.pinned
        and abs(matched.rate - pa_hr.rate) > DISAGREEMENT_PTS
    )
    checks = _audit(track, {
        "warmup_min": t_start / 60.0,
        "duration_min": duration_min,
        "block_min": block_min,
        "stability": stability,
        "pace_change_pct": pace_change,
        "stopped_pct": stopped_pct,
        "mean_grade_pct": float(grade.mean()),
        "grade_spread_pct": float(np.percentile(grade, 95) - np.percentile(grade, 5)),
        "disagree": disagree,
        "pa_hr_rate": pa_hr.rate,
        "matched_rate": matched.rate if matched else math.nan,
        "matched_coverage": matched.coverage if matched else None,
    }, target)

    return Analysis(
        track=track,
        t_start=t_start,
        t_end=t_end,
        first=first,
        second=second,
        whole=whole,
        stability=stability,
        pa_hr=pa_hr,
        speed_matched=matched,
        checks=checks,
        target=target,
        stopped_pct=stopped_pct,
        mean_grade_pct=float(grade.mean()),
        speed_cv=speed_cv,
        hr_spread=hr_spread,
        rolling=rolling_drift(track, rolling_min, 120, drop_stops),
        sweep=robustness_sweep(track, target, drop_stops=drop_stops) if sweep else None,
    )
