"""The drop-a-file page: run its JavaScript under Node, check it agrees."""

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import textwrap

import pytest

import synth
from hrdrift import Track, analyse, webapp, wrap_document

NODE = shutil.which("node")


def node_env() -> dict:
    """Env for node, with NODE_PATH pointing at the repo's node_modules.

    bridge.js lives in a tmp dir, and node resolves modules relative to the
    script, so without this it never finds the shim.
    """
    env = dict(os.environ)
    root = pathlib.Path(__file__).resolve().parents[1] / "node_modules"
    if root.is_dir():
        existing = env.get("NODE_PATH")
        env["NODE_PATH"] = f"{root}{os.pathsep}{existing}" if existing else str(root)
    return env


def has_xml_shim() -> bool:
    if NODE is None:
        return False
    # probe with the same env and from the same kind of cwd the bridge uses
    probe = subprocess.run([NODE, "-e", "require('@xmldom/xmldom')"],
                           capture_output=True, text=True,
                           cwd=tempfile.gettempdir(), env=node_env())
    return probe.returncode == 0


BRIDGE = textwrap.dedent("""
    const fs = require('fs');
    const html = fs.readFileSync(process.argv[2], 'utf8');
    const js = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));
    const stub = () => ({ innerHTML: '', classList: { add(){}, remove(){} },
      addEventListener(){}, click(){}, files: [] });
    const doc = { getElementById: stub, querySelector: () => null };
    const mod = { exports: {} };
    new Function('module', 'document', 'DOMParser', 'FileReader', js)(
      mod, doc, require('@xmldom/xmldom').DOMParser, function(){});
    const api = mod.exports;
    const S = api.buildStreams(api.parseActivity(fs.readFileSync(process.argv[3], 'utf8')));
    const r = api.analyse(S);
    console.log(JSON.stringify({
      usable: !!r.usable, fatal: r.fatal || null, rate: r.rate ?? null,
      blockMin: r.blockMin ?? null, hr: r.c ? r.c.whole.hr : null,
      paHr: r.c ? r.c.paHr.rate : null, sm: r.c && r.c.sm ? r.c.sm.rate : null,
      agreement: r.sw ? r.sw.agreement : null, reasons: r.reasons || [],
      totalMin: S.total / 60, distKm: S.dist[S.total] / 1000,
    }));
""")


class TestPageStructure:
    def test_is_self_contained(self):
        html = webapp()
        assert "<title>" in html and "<script>" in html
        assert "src=" not in html.replace('type="file"', "")
        assert "http://" not in html and "https://" not in html

    def test_declares_both_themes(self):
        html = webapp()
        assert "prefers-color-scheme: dark" in html
        assert "[data-theme=dark]" in html

    def test_config_is_injected_from_python(self):
        html = webapp(target=4.0, warmup_min=12, duration_min=50)
        cfg = json.loads(re.search(r"const CFG = (\{.*?\});", html, re.S).group(1))
        assert cfg["target"] == 4.0
        assert cfg["warmup"] == 12
        assert cfg["duration"] == 50

    def test_thresholds_come_from_the_library(self):
        from hrdrift.analysis import (VALID_BLOCK_MIN, VALID_MAX_HR_SPREAD,
                                      VALID_MAX_SPEED_CV, VALID_MAX_STOPPED_PCT)
        cfg = json.loads(re.search(r"const CFG = (\{.*?\});", webapp(), re.S).group(1))
        assert cfg["minBlock"] == VALID_BLOCK_MIN
        assert cfg["maxStopped"] == VALID_MAX_STOPPED_PCT
        assert cfg["maxSpeedCv"] == VALID_MAX_SPEED_CV
        assert cfg["maxHrSpread"] == VALID_MAX_HR_SPREAD

    def test_states_that_nothing_is_uploaded(self):
        assert "never uploaded" in webapp()

    def test_wraps_into_a_document(self):
        doc = wrap_document(webapp())
        assert doc.startswith("<!doctype html>")
        assert doc.rstrip().endswith("</html>")


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    d = tmp_path_factory.mktemp("bridge")
    (d / "bridge.js").write_text(BRIDGE)
    (d / "page.html").write_text(wrap_document(webapp()))
    return str(d / "bridge.js"), str(d / "page.html")


def write_gpx(path, seconds, hr_fn, speed_mps=2.7, lat0=47.4, lon0=8.5, alt0=412.0):
    """A GPX carrying no distance stream, so it must be rebuilt from positions."""
    import datetime as dt
    deg_per_m = 1.0 / 111320.0
    start = dt.datetime(2026, 1, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
    pts = []
    for i in range(seconds + 1):
        lat = lat0 + i * speed_mps * deg_per_m
        stamp = (start + dt.timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append(
            f'<trkpt lat="{lat:.7f}" lon="{lon0:.7f}"><ele>{alt0:.1f}</ele>'
            f"<time>{stamp}</time><extensions><gpxtpx:TrackPointExtension>"
            f"<gpxtpx:hr>{hr_fn(i):.0f}</gpxtpx:hr>"
            f"</gpxtpx:TrackPointExtension></extensions></trkpt>")
    path.write_text(
        '<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
        f"<trk><trkseg>{''.join(pts)}</trkseg></trk></gpx>")
    return path


@pytest.mark.skipif(not has_xml_shim(),
                    reason="needs node with @xmldom/xmldom (npm i @xmldom/xmldom)")
class TestBrowserMatchesLibrary:
    def run_js(self, bridge, activity):
        js, page = bridge
        proc = subprocess.run([NODE, js, page, str(activity)],
                              capture_output=True, text=True, check=True,
                              env=node_env())
        return json.loads(proc.stdout)

    @pytest.mark.parametrize("fixture,kwargs", [
        ("constant_pace_rising_hr", {"rise_pct": 6.0}),
        ("constant_hr_slowing_pace", {"decay_pct": 8.0}),
        ("both_rising", {"hr_rise_pct": 12.0, "speed_rise_pct": 8.0}),
        ("rolling_hills", {}),
    ])
    def test_same_answer_on_valid_tests(self, tmp_path, bridge, fixture, kwargs):
        src = getattr(synth, fixture)(tmp_path / "a.tcx", **kwargs)
        js = self.run_js(bridge, src)
        py = analyse(Track.load(src))

        assert js["usable"] is py.usability.usable
        assert js["hr"] == pytest.approx(py.whole.hr, abs=0.05)
        assert js["paHr"] == pytest.approx(py.pa_hr.rate, abs=0.05)
        assert js["agreement"] == py.sweep.agreement
        expected = py.sweep.median if py.sweep_speaks else py.primary.rate
        assert js["rate"] == pytest.approx(expected, abs=0.05)
        if py.speed_matched is not None:
            assert js["sm"] == pytest.approx(py.speed_matched.rate, abs=0.05)

    def test_rejects_the_same_files(self, tmp_path, bridge):
        src = synth.erratic_wander(tmp_path / "a.tcx")
        js = self.run_js(bridge, src)
        py = analyse(Track.load(src))
        assert js["usable"] is False and py.usability.usable is False
        assert len(js["reasons"]) == len(py.usability.reasons)

    def test_reads_gpx_without_a_distance_stream(self, tmp_path, bridge):
        gpx = write_gpx(tmp_path / "a.gpx", 75 * 60,
                        lambda i: 145 * (1 + 0.06 * max(0, i - 900) / 3600))
        js = self.run_js(bridge, gpx)
        py = analyse(Track.load(str(gpx)))
        assert js["usable"] is py.usability.usable
        assert js["hr"] == pytest.approx(py.whole.hr, abs=0.05)
        assert js["distKm"] == pytest.approx(py.track.distance_km, rel=0.01)
        expected = py.sweep.median if py.sweep_speaks else py.primary.rate
        assert js["rate"] == pytest.approx(expected, abs=0.1)

    def test_rejects_a_file_with_no_heart_rate(self, tmp_path, bridge):
        js, page = bridge
        bad = tmp_path / "nohr.gpx"
        pts = "".join(
            f'<trkpt lat="47.4" lon="8.5"><ele>400</ele>'
            f"<time>2026-01-01T09:{i // 60:02d}:{i % 60:02d}Z</time></trkpt>"
            for i in range(120))
        bad.write_text('<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1">'
                       f"<trk><trkseg>{pts}</trkseg></trk></gpx>")
        proc = subprocess.run([NODE, js, page, str(bad)], capture_output=True,
                              text=True, env=node_env())
        assert proc.returncode != 0
        assert "no heart-rate data" in proc.stderr
