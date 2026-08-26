"""NGP, efficiency factor, range selection, browser parity."""

import json
import re
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

import synth
from hrdrift import Track, analyse, interactive_report
from hrdrift.track import YARDS_PER_METRE, normalized_graded_pace


class TestNormalizedGradedPace:
    def test_equals_the_mean_when_pace_is_perfectly_steady(self):
        v = np.full(3600, 2.8)
        assert normalized_graded_pace(v) == pytest.approx(2.8, rel=1e-6)

    def test_exceeds_the_mean_when_pace_varies(self):
        v = np.concatenate([np.full(1800, 2.0), np.full(1800, 3.6)])
        assert normalized_graded_pace(v) > v.mean()

    def test_barely_moves_on_a_steady_run(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=4.0)
        a = analyse(Track.load(path))
        assert a.whole.ngp_speed == pytest.approx(a.whole.gap_speed, rel=0.03)

    def test_empty_input_is_nan(self):
        assert np.isnan(normalized_graded_pace(np.array([])))


class TestEfficiencyFactor:
    def test_trainingpeaks_units_are_yards_per_minute_per_bpm(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx")
        a = analyse(Track.load(path))
        assert a.whole.ef_trainingpeaks == pytest.approx(
            a.whole.ef * 60 * YARDS_PER_METRE, rel=1e-9)

    def test_lands_in_the_range_trainingpeaks_shows_for_running(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", hr=145.0, speed0=2.8)
        a = analyse(Track.load(path))
        assert 0.7 < a.whole.ef_trainingpeaks < 2.0

    def test_pa_hr_is_built_on_ngp(self, tmp_path):
        path = synth.constant_hr_slowing_pace(tmp_path / "a.tcx", decay_pct=8.0)
        a = analyse(Track.load(path))
        expected = (a.first.ngp_speed / a.first.hr - a.second.ngp_speed / a.second.hr)
        expected = expected / (a.first.ngp_speed / a.first.hr) * 100
        assert a.pa_hr.pct == pytest.approx(expected, rel=1e-9)


class TestRangeSelection:
    def test_from_and_to_match_warmup_and_duration(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        trk = Track.load(path)
        by_warmup = analyse(trk, warmup_min=15, duration_min=60)
        by_range = analyse(trk, t_from=15, t_to=75)
        assert by_range.t_start == by_warmup.t_start
        assert by_range.t_end == by_warmup.t_end
        assert by_range.primary.rate == pytest.approx(by_warmup.primary.rate)

    def test_to_defaults_to_end_of_recording(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        trk = Track.load(path)
        a = analyse(trk, t_from=20)
        assert a.t_end == pytest.approx(trk.t[-1])

    def test_short_selection_is_refused(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        with pytest.raises(ValueError, match="at least"):
            analyse(Track.load(path), t_from=20, t_to=25)

    def test_speed_bins_do_not_shift_with_the_selection(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx", rise_pct=6.0)
        trk = Track.load(path)
        wide = analyse(trk, t_from=15, t_to=75).speed_matched.pct
        again = analyse(trk, t_from=15.0, t_to=75.0).speed_matched.pct
        assert wide == pytest.approx(again, rel=1e-12)


class TestInteractivePage:
    def test_is_self_contained(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        html = interactive_report(Track.load(path))
        assert "<title>" in html and "<script>" in html
        assert "src=" not in html            # no external scripts
        assert "@import" not in html

    def test_embeds_the_streams(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        trk = Track.load(path)
        html = interactive_report(trk)
        payload = json.loads(re.search(r"const P = (\{.*?\});", html, re.S).group(1))
        assert len(payload["s"]["hr"]) == len(trk.t)
        assert set(payload["s"]) == {"t", "hr", "v", "gv", "d", "grade"}

    def test_speed_survives_the_round_trip(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        trk = Track.load(path)
        html = interactive_report(trk)
        payload = json.loads(re.search(r"const P = (\{.*?\});", html, re.S).group(1))
        assert np.allclose(payload["s"]["gv"], trk.gap_speed, atol=1e-6)
        assert np.allclose(payload["s"]["v"], trk.speed, atol=1e-6)

    def test_downsampling_shrinks_the_payload(self, tmp_path):
        path = synth.constant_pace_rising_hr(tmp_path / "a.tcx")
        trk = Track.load(path)
        assert len(interactive_report(trk, step=4)) < len(interactive_report(trk, step=1))


NODE = shutil.which("node")

BRIDGE = textwrap.dedent("""
    const fs = require('fs');
    const html = fs.readFileSync(process.argv[2], 'utf8');
    const js = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));
    const stub = () => ({ innerHTML:'', textContent:'', className:'', dataset:{},
      classList:{toggle(){},add(){},remove(){}}, addEventListener(){},
      setPointerCapture(){}, getBoundingClientRect:()=>({left:0,width:1000}) });
    global.document = { getElementById: stub, querySelectorAll: () => [] };
    global.__svg = stub();
    new Function(js.replace('const svg = document.getElementById("chart");',
                            'const svg = global.__svg;') +
                 '\\nglobal.__analyse = analyse;')();
    const [from, to] = process.argv.slice(3).map(Number);
    const a = global.__analyse(from, to);
    console.log(JSON.stringify(a && {
      hr: a.whole.hr, ngp: a.whole.ngp, efTp: a.whole.efTp,
      paHrRate: a.paHrRate, smRate: a.smRate,
      coverage: a.sm ? a.sm.coverage : null,
      pinned: a.stab.pinned, primaryRate: a.primaryRate }));
""")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
class TestBrowserMatchesLibrary:
    """The page recomputes every metric in JavaScript. It must not drift from
    the Python it mirrors, or the same file would give two different answers."""

    @pytest.fixture
    def bridge(self, tmp_path):
        p = tmp_path / "bridge.js"
        p.write_text(BRIDGE)
        return str(p)

    def run_js(self, bridge, page, i0, i1):
        out = subprocess.run([NODE, bridge, str(page), str(i0), str(i1)],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    @pytest.mark.parametrize("fixture,kwargs", [
        ("constant_pace_rising_hr", {"rise_pct": 6.0}),
        ("constant_hr_slowing_pace", {"decay_pct": 8.0}),
        ("both_rising", {"hr_rise_pct": 12.0, "speed_rise_pct": 8.0}),
    ])
    @pytest.mark.parametrize("start_min,end_min", [(15, 75), (10, 60), (0, 70)])
    def test_same_numbers(self, tmp_path, bridge, fixture, kwargs, start_min, end_min):
        src = getattr(synth, fixture)(tmp_path / "a.tcx", **kwargs)
        trk = Track.load(src)
        page = tmp_path / "page.html"
        page.write_text(interactive_report(trk))

        end_min = min(end_min, trk.duration_min)
        py = analyse(trk, t_from=start_min, t_to=end_min)
        js = self.run_js(bridge, page, int(start_min * 60), int(end_min * 60))

        assert js["hr"] == pytest.approx(py.whole.hr, abs=0.05)
        assert js["ngp"] == pytest.approx(py.whole.ngp_speed, abs=1e-3)
        assert js["efTp"] == pytest.approx(py.whole.ef_trainingpeaks, abs=1e-3)
        assert js["paHrRate"] == pytest.approx(py.pa_hr.rate, abs=0.05)
        assert js["pinned"] is py.stability.pinned
        assert js["primaryRate"] == pytest.approx(py.primary.rate, abs=0.1)
        if py.speed_matched is not None:
            assert js["smRate"] == pytest.approx(py.speed_matched.rate, abs=0.1)
            assert js["coverage"] == pytest.approx(py.speed_matched.coverage, abs=2.0)
