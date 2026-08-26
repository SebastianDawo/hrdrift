"""The robustness sweep: drift across every block boundary."""

import numpy as np
import pytest

import synth
from hrdrift import Track, analyse
from hrdrift.analysis import Sweep, robustness_sweep


def load(p):
    return Track.load(p)


class TestSweepMechanics:
    def test_covers_many_block_choices(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=60, warmup_min=15)
        sw = robustness_sweep(load(path))
        assert sw.n >= 10
        assert len({(p.start_min, p.block_min) for p in sw.points}) == sw.n

    def test_skips_blocks_that_run_past_the_recording(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=30, warmup_min=10)
        trk = load(path)
        sw = robustness_sweep(trk)
        for p in sw.points:
            assert (p.start_min + p.block_min) * 60 <= trk.t[-1] + 1

    def test_empty_sweep_when_recording_is_short(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=12, warmup_min=2)
        sw = robustness_sweep(load(path))
        assert sw.n == 0
        assert sw.verdict[0] == "NO VERDICT"
        assert np.isnan(sw.median)

    def test_quartiles_bracket_the_median(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        sw = robustness_sweep(load(path))
        assert sw.p25 <= sw.median <= sw.p75
        assert sw.spread >= 0


class TestSweepVerdicts:
    def test_clean_test_well_above_reads_above(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=18.0)
        sw = robustness_sweep(load(path))
        assert sw.verdict[0] == "ABOVE AeT"
        assert sw.fraction_above >= 75

    def test_clean_test_well_below_reads_below(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=1.0)
        sw = robustness_sweep(load(path))
        assert sw.verdict[0] == "BELOW AeT"
        assert sw.fraction_above <= 25

    def test_steady_effort_gives_a_tight_spread(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
        sw = robustness_sweep(load(path))
        assert sw.agreement == "tight"
        assert sw.spread < Sweep.TIGHT_SPREAD

    def test_erratic_effort_is_reported_unstable(self, tmp_path):
        path = synth.erratic_wander(tmp_path / "a.tcx")
        sw = robustness_sweep(load(path))
        assert sw.agreement == "unstable"
        assert sw.verdict[0] == "UNSTABLE"
        assert "No verdict" in sw.verdict[1]

    def test_unstable_refuses_to_estimate_aet(self, tmp_path):
        path = synth.erratic_wander(tmp_path / "a.tcx")
        a = analyse(load(path))
        assert "No estimate" in a.aet_estimate


class TestAnalysisUsesTheSweep:
    def test_verdict_defers_to_the_sweep(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
        a = analyse(load(path))
        assert a.sweep_speaks
        assert a.verdict == a.sweep.verdict

    def test_single_block_verdict_still_available(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
        a = analyse(load(path))
        assert a.single_verdict[0] in ("ABOVE AeT", "AT OR JUST BELOW AeT", "WELL BELOW AeT")

    def test_sweep_can_be_switched_off(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        a = analyse(load(path), sweep=False)
        assert a.sweep is None
        assert not a.sweep_speaks
        assert a.verdict == a.single_verdict

    def test_falls_back_to_single_block_when_sweep_is_thin(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=35, warmup_min=10)
        a = analyse(load(path), warmup_min=10, duration_min=35)
        if not a.sweep_speaks:
            assert a.verdict == a.single_verdict

    def test_report_shows_the_robustness_section(self, tmp_path):
        from hrdrift import text_report
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        out = text_report(analyse(load(path)))
        assert "ROBUSTNESS" in out
        assert "interquartile range" in out
        assert "agreement" in out


class TestCliExitStatus:
    """0 below threshold, 1 above, 3 not a valid test, 2 error."""

    def test_unstable_exits_three(self, tmp_path):
        from hrdrift.cli import main
        path = synth.erratic_wander(tmp_path / "a.tcx")
        assert main([path, "--quiet"]) == 3

    def test_below_threshold_exits_zero(self, tmp_path):
        from hrdrift.cli import main
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=1.0)
        assert main([path, "--quiet"]) == 0

    def test_above_threshold_exits_one(self, tmp_path):
        from hrdrift.cli import main
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=18.0)
        assert main([path, "--quiet"]) == 1

    def test_missing_file_exits_two(self, tmp_path):
        from hrdrift.cli import main
        assert main([str(tmp_path / "nope.tcx"), "--quiet"]) == 2
