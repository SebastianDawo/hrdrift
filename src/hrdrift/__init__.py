"""Heart-rate drift analysis for the aerobic threshold field test.

    from hrdrift import Track, analyse, text_report

    a = analyse(Track.load("run.tcx"))
    print(text_report(a))
    print(a.primary.rate)   # %/hour
"""

from .analysis import (
    DEFAULT_TARGET,
    Analysis,
    Sweep,
    SweepPoint,
    Check,
    DriftMetric,
    HalfStats,
    Stability,
    analyse,
    find_steadiest_block,
    robustness_sweep,
    rolling_drift,
)
from .interactive import interactive_report
from .webapp import webapp
from .report import format_pace, html_report, text_report, wrap_document
from .track import Track, TrackError, minetti_factor, normalized_graded_pace

__version__ = "1.0.0"

__all__ = [
    "Analysis", "Check", "DriftMetric", "HalfStats", "Stability", "Track", "TrackError",
    "Sweep", "SweepPoint", "DEFAULT_TARGET", "analyse", "find_steadiest_block",
    "robustness_sweep", "rolling_drift",
    "format_pace", "html_report", "interactive_report", "text_report", "wrap_document",
    "webapp", "minetti_factor",
    "normalized_graded_pace", "__version__",
]
