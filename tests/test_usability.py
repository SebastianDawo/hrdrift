"""The validity gates: is this file a drift test at all?"""

import math

import pytest

import synth
from hrdrift import Track, analyse
from hrdrift.analysis import VALID_BLOCK_MIN
from hrdrift.cli import main
from hrdrift.report import brief_report, text_report


def load(p):
    return Track.load(p)


class TestAcceptsRealTests:
    def test_clean_test_is_valid(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=4.0)
        a = analyse(load(path))
        assert a.usability.usable, a.usability.reasons

    def test_loose_test_is_still_valid(self, tmp_path):
        path = synth.both_rising(tmp_path / "a.tcx", hr_rise_pct=8.0, speed_rise_pct=4.0)
        a = analyse(load(path))
        assert a.usability.usable, a.usability.reasons
        assert "DRIFT" in a.answer
        assert a.primary is a.speed_matched   # and the right metric is chosen


class TestRejects:
    def test_too_short(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=25, warmup_min=10)
        a = analyse(load(path), warmup_min=10, duration_min=25)
        assert not a.usability.usable
        assert any("at least" in r for r in a.usability.reasons)
        assert a.block_min < VALID_BLOCK_MIN

    def test_pace_varies_too_much(self, tmp_path):
        def speed(t):
            return 4.2 if (t // 180) % 2 else 2.2

        path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 150, speed, 75 * 60, wobble=0)
        a = analyse(load(path))
        assert not a.usability.usable
        assert any("varied" in r for r in a.usability.reasons)

    def test_erratic_effort(self, tmp_path):
        path = synth.erratic_wander(tmp_path / "a.tcx")
        a = analyse(load(path))
        assert not a.usability.usable
        assert any("where the block is drawn" in r for r in a.usability.reasons)

    def test_wide_heart_rate_range(self, tmp_path):
        def hr(t):
            return 120 + 60 * ((t // 300) % 2)

        path = synth.write_tcx(tmp_path / "a.tcx", hr, lambda t: 2.7, 75 * 60)
        a = analyse(load(path))
        assert not a.usability.usable

    def test_stop_start_run(self, tmp_path):
        def speed(t):
            return 0.0 if (t % 420) < 110 else 2.9

        path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 150, speed, 75 * 60, wobble=0)
        a = analyse(load(path))
        assert not a.usability.usable
        assert any("stopped" in r or "walking" in r for r in a.usability.reasons)


class TestReporting:
    def test_answer_is_one_line(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=4.0)
        a = analyse(load(path))
        assert "\n" not in a.answer
        assert "threshold" in a.answer

    def test_invalid_answer_says_so(self, tmp_path):
        path = synth.erratic_wander(tmp_path / "a.tcx")
        a = analyse(load(path))
        assert a.answer == "NOT A VALID DRIFT TEST"
        assert "not a drift test" in a.result_line

    def test_invalid_report_withholds_the_number(self, tmp_path):
        path = synth.erratic_wander(tmp_path / "a.tcx")
        out = text_report(analyse(load(path)))
        assert "NOT A VALID DRIFT TEST" in out
        assert "VERDICT:" not in out
        assert "not reported as a result" in out

    def test_brief_report_is_short(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx")
        out = brief_report(analyse(load(path)))
        assert len(out.splitlines()) < 12
        assert "DRIFT" in out

    def test_brief_report_lists_reasons_when_invalid(self, tmp_path):
        path = synth.erratic_wander(tmp_path / "a.tcx")
        out = brief_report(analyse(load(path)))
        assert "cannot answer the question" in out
        assert "15 minutes of warm-up" in out

    def test_cli_brief_flag(self, tmp_path, capsys):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=4.0)
        main([path, "--brief"])
        out = capsys.readouterr().out
        assert "DRIFT" in out
        assert "PROTOCOL COMPLIANCE" not in out
