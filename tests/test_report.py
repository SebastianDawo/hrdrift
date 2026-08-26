import re

import pytest

import synth
from hrdrift import Track, analyse, format_pace, html_report, text_report
from hrdrift.cli import main


@pytest.fixture
def analysis(tmp_path):
    path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
    return analyse(Track.load(path))


class TestFormatPace:
    @pytest.mark.parametrize("value,expected", [
        (6.0, "6:00"), (6.5, "6:30"), (5.999, "6:00"), (0.0, "--:--"), (-1.0, "--:--"),
    ])
    def test_formats(self, value, expected):
        assert format_pace(value) == expected

    def test_handles_nan(self):
        assert format_pace(float("nan")) == "--:--"


class TestTextReport:
    def test_contains_the_key_sections(self, analysis):
        out = text_report(analysis)
        for heading in ("PROTOCOL COMPLIANCE", "ANALYSIS BLOCK", "DRIFT",
                        "VERDICT", "AEROBIC THRESHOLD ESTIMATE"):
            assert heading in out

    def test_marks_the_primary_metric(self, analysis):
        assert "<- primary" in text_report(analysis)

    def test_reports_both_metrics(self, analysis):
        out = text_report(analysis)
        assert "Pa:Hr decoupling" in out
        assert "speed-matched drift" in out


class TestHtmlReport:
    def test_is_self_contained(self, analysis):
        html = html_report(analysis)
        assert "<title>" in html
        assert "http://" not in html.replace("http://www.w3.org", "")
        assert "<svg" in html

    def test_declares_both_themes(self, analysis):
        html = html_report(analysis)
        assert "prefers-color-scheme: dark" in html
        assert '[data-theme=dark]' in html

    def test_escapes_the_filename(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a<script>.tcx")
        html = html_report(analyse(Track.load(path)))
        assert "<script>" not in html


class TestCli:
    def test_exit_code_zero_below_threshold(self, tmp_path, capsys):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=1.0)
        assert main([path]) == 0
        assert "VERDICT" in capsys.readouterr().out

    def test_exit_code_one_above_threshold(self, tmp_path, capsys):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=16.0)
        assert main([path]) == 1

    def test_writes_html(self, tmp_path, capsys):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        out = tmp_path / "r.html"
        main([path, "--html", str(out), "--quiet"])
        assert out.exists() and out.stat().st_size > 1000

    def test_missing_file_exits_two(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope.tcx")]) == 2
        assert "hrdrift:" in capsys.readouterr().err

    def test_unusable_window_exits_two(self, tmp_path, capsys):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", block_min=8, warmup_min=1)
        assert main([path, "--warmup", "30", "--duration", "60"]) == 2
