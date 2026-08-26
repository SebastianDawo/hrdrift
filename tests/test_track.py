import datetime as dt

import numpy as np
import pytest

import synth
from hrdrift.track import Track, TrackError, minetti_factor, moving_average


def test_loads_tcx_and_resamples_to_1hz(tmp_path):
    path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 3.0, 600)
    trk = Track.load(path)
    assert len(trk.t) == 601
    assert np.allclose(np.diff(trk.t), 1.0)
    assert trk.duration_min == pytest.approx(10.0, abs=0.02)


def test_speed_recovered_from_distance(tmp_path):
    path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 3.0, 900, wobble=0)
    trk = Track.load(path)
    mid = (trk.t > 120) & (trk.t < 780)
    assert trk.speed[mid].mean() == pytest.approx(3.0, abs=0.05)


def test_distance_is_monotonic(tmp_path):
    path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 2.5, 600)
    trk = Track.load(path)
    assert np.all(np.diff(trk.dist) >= -1e-9)


def test_rejects_file_without_heart_rate(tmp_path):
    path = tmp_path / "nohr.gpx"
    pts = "".join(
        f'<trkpt lat="47.4" lon="8.5"><ele>400</ele>'
        f'<time>2026-01-01T09:{i//60:02d}:{i%60:02d}Z</time></trkpt>'
        for i in range(120)
    )
    path.write_text(
        '<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1">'
        f"<trk><trkseg>{pts}</trkseg></trk></gpx>"
    )
    with pytest.raises(TrackError, match="heart-rate"):
        Track.load(str(path))


def test_rejects_unknown_extension(tmp_path):
    p = tmp_path / "x.fit"
    p.write_bytes(b"\x00" * 100)
    with pytest.raises(TrackError, match="unsupported"):
        Track.load(str(p))


def test_rejects_too_few_points(tmp_path):
    path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 3.0, 30)
    with pytest.raises(TrackError, match="trackpoints"):
        Track.load(path)


class TestMinettiFactor:
    def test_level_ground_is_unity(self):
        assert minetti_factor(0.0) == pytest.approx(1.0)

    def test_uphill_costs_more(self):
        assert minetti_factor(0.10) > 1.4

    def test_shallow_downhill_costs_less(self):
        assert minetti_factor(-0.10) < 1.0

    def test_clipped_beyond_valid_range(self):
        assert minetti_factor(5.0) == pytest.approx(minetti_factor(0.45))


def test_grade_adjustment_flattens_rolling_terrain(tmp_path):
    path = synth.rolling_hills(tmp_path / "hills.tcx", block_min=30, warmup_min=5)
    trk = Track.load(path)
    mid = (trk.t > 300) & (trk.t < trk.t[-1] - 300)
    assert trk.grade[mid].std() > 0.005          # the terrain really does roll
    assert abs(trk.grade[mid].mean()) < 0.01     # and averages out flat


def test_moving_average_preserves_length_and_mean():
    x = np.arange(100, dtype=float)
    out = moving_average(x, 11)
    assert len(out) == len(x)
    assert out.mean() == pytest.approx(x.mean(), abs=0.5)


def test_moving_average_window_of_one_is_identity():
    x = np.array([1.0, 5.0, 3.0])
    assert np.array_equal(moving_average(x, 1), x)


def test_start_time_parsed(tmp_path):
    start = dt.datetime(2026, 3, 3, 15, 8, 0, tzinfo=dt.timezone.utc)
    path = synth.write_tcx(tmp_path / "a.tcx", lambda t: 140, lambda t: 3.0, 300, start=start)
    assert Track.load(path).start == start
