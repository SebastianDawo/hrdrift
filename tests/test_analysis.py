import numpy as np
import pytest

import synth
from hrdrift import Track, analyse
from hrdrift.analysis import find_steadiest_block, rolling_drift


def load(path):
    return Track.load(path)


class TestSpeedMatchedDrift:
    """Recovering drift that was built into the file."""

    @pytest.mark.parametrize("rise", [3.0, 6.0, 9.0])
    def test_recovers_known_hr_rise_at_constant_pace(self, tmp_path, rise):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=rise)
        a = analyse(load(path))
        # Half means differ by half the total linear rise.
        assert a.speed_matched.pct == pytest.approx(rise / 2, abs=0.6)

    def test_catches_drift_that_pa_hr_misses(self, tmp_path):
        path = synth.both_rising(tmp_path / "a.tcx", hr_rise_pct=12.0, speed_rise_pct=8.0)
        a = analyse(load(path))
        assert a.pa_hr.pct < 3.0                        # ratio is fooled
        assert a.speed_matched.pct == pytest.approx(6.0, abs=1.0)   # matched is not
        assert a.metrics_disagree
        assert a.primary is a.speed_matched

    def test_no_drift_when_nothing_changes(self, tmp_path):
        path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 2.8, 4500, wobble=0)
        a = analyse(load(path))
        assert abs(a.speed_matched.pct) < 0.2
        assert abs(a.pa_hr.pct) < 0.2

    def test_pace_wobble_alone_does_not_manufacture_drift(self, tmp_path):
        path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 2.8, 4500)
        a = analyse(load(path))
        assert abs(a.primary.rate) < 1.5


class TestPaHrDecoupling:
    def test_detects_pace_decay_at_pinned_hr(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=8.0)
        a = analyse(load(path))
        assert a.pa_hr.pct == pytest.approx(4.0, abs=1.0)
        assert not a.metrics_disagree     # both metrics agree when HR is pinned

    def test_agrees_with_speed_matched_when_one_variable_pinned(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
        a = analyse(load(path))
        assert abs(a.pa_hr.rate - a.speed_matched.rate) < 2.0


class TestMetricSelection:
    """Which metric gets reported."""

    def test_pinned_heart_rate_selects_pa_hr(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=8.0)
        a = analyse(load(path))
        assert a.stability.pinned
        assert a.primary is a.pa_hr
        assert a.primary.pct > 3.0
        assert abs(a.speed_matched.pct) < 0.5     # the trap this rule avoids

    def test_free_heart_rate_selects_speed_matched(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=8.0)
        a = analyse(load(path))
        assert not a.stability.pinned
        assert a.primary is a.speed_matched

    def test_falls_back_to_pa_hr_when_speeds_do_not_overlap(self, tmp_path):
        path = synth.both_rising(tmp_path / "a.tcx", hr_rise_pct=10.0,
                                 speed_rise_pct=25.0, wobble=0)
        a = analyse(load(path))
        assert not a.matched_usable
        assert a.primary is a.pa_hr
        assert any("speed bins" in c.message for c in a.checks)


class TestDriftRate:
    def test_rate_normalises_block_length(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=60, rise_pct=6.0)
        trk = load(path)
        long_block = analyse(trk, duration_min=60)
        short_block = analyse(trk, duration_min=30)
        assert long_block.primary.rate == pytest.approx(short_block.primary.rate, abs=1.5)
        # Raw percentages, by contrast, are not comparable.
        assert short_block.primary.pct < long_block.primary.pct

    def test_rate_is_pct_scaled_to_an_hour(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=30)
        a = analyse(load(path), duration_min=30)
        assert a.primary.rate == pytest.approx(a.primary.pct * 2, rel=1e-6)


class TestVerdict:
    def test_above_threshold_when_drift_is_large(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=16.0)
        a = analyse(load(path))
        assert a.primary.rate > a.target
        assert a.verdict[0] == "ABOVE AeT"
        assert "below" in a.aet_estimate

    def test_below_threshold_when_drift_is_small(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=1.0)
        a = analyse(load(path))
        assert a.primary.rate < a.target
        assert "at or above" in a.aet_estimate

    def test_target_is_configurable(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
        strict = analyse(load(path), target=1.0)
        lax = analyse(load(path), target=20.0)
        assert strict.verdict[0] == "ABOVE AeT"
        assert lax.verdict[0] != "ABOVE AeT"


class TestProtocolAudit:
    def test_compliant_test_passes(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=3.0)
        a = analyse(load(path))
        assert a.reliable, [c.message for c in a.checks if c.level == "fail"]

    def test_short_recording_fails(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=25, warmup_min=5)
        a = analyse(load(path))
        assert not a.reliable
        assert any("min" in c.message for c in a.checks if c.level == "fail")

    def test_uncontrolled_test_is_flagged(self, tmp_path):
        path = synth.both_rising(tmp_path / "a.tcx", hr_rise_pct=12.0, speed_rise_pct=8.0)
        a = analyse(load(path))
        messages = " ".join(c.message for c in a.checks if c.level == "fail")
        assert "Neither variable" in messages
        assert "disagree" in messages

    def test_flat_terrain_reported_as_flat(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx")
        a = analyse(load(path))
        assert any("flat" in c.message for c in a.checks if c.level == "ok")


class TestBlockSelection:
    def test_halves_are_equal_length(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx")
        a = analyse(load(path))
        assert abs(a.first.n - a.second.n) <= 1

    def test_warmup_is_excluded(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", warmup_min=15)
        a = analyse(load(path), warmup_min=15)
        assert a.t_start == pytest.approx(900)

    def test_block_truncated_to_recording_length(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=40, warmup_min=10)
        trk = load(path)
        a = analyse(trk, warmup_min=10, duration_min=60)
        assert a.t_end <= trk.t[-1]

    def test_auto_picks_a_block(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        a = analyse(load(path), auto=True)
        assert a.t_end > a.t_start

    def test_raises_when_block_unusable(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=10, warmup_min=1)
        with pytest.raises(ValueError):
            analyse(load(path), warmup_min=10, duration_min=60)

    def test_find_steadiest_block_returns_none_when_too_short(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=10, warmup_min=1)
        assert find_steadiest_block(load(path), 60) is None


def test_rolling_drift_covers_the_run(tmp_path):
    path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
    windows = rolling_drift(load(path), block_min=20, step_s=300)
    assert len(windows) > 5
    starts = [w[0] for w in windows]
    assert starts == sorted(starts)


def test_stability_detects_pinned_heart_rate(tmp_path):
    path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx")
    a = analyse(load(path))
    assert a.stability.within_3bpm > 95
    assert abs(a.stability.slope_bph) < 1.0


def test_stability_detects_rising_heart_rate(tmp_path):
    path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", hr0=140, rise_pct=6.0)
    a = analyse(load(path))
    assert a.stability.slope_bph == pytest.approx(140 * 0.06, abs=2.0)
