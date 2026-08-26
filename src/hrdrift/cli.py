"""Command-line interface: ``hrdrift TRACK.tcx``."""

from __future__ import annotations

import argparse
import sys

from .analysis import DEFAULT_TARGET, analyse
from .interactive import interactive_report
from .webapp import webapp
from .report import brief_report, html_report, text_report, wrap_document
from .track import Track, TrackError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hrdrift",
        description="Analyse a heart-rate drift test (aerobic threshold field test).",
        epilog="Protocol: 15 min warm-up, then 60 min holding a steady heart rate "
               "or a steady pace. Drift under 5%/h means the effort was at or below AeT.",
    )
    p.add_argument("file", nargs="?", help="activity export (.tcx or .gpx)")
    p.add_argument("--warmup", type=float, default=15.0,
                   help="minutes to exclude before the test block (default: 15)")
    p.add_argument("--duration", type=float, default=60.0,
                   help="length of the test block in minutes (default: 60)")
    p.add_argument("--target", type=float, default=DEFAULT_TARGET,
                   help="drift threshold in %%/hour (default: 5)")
    p.add_argument("--rolling", type=float, default=20.0,
                   help="rolling window width in minutes (default: 20)")
    p.add_argument("--auto", action="store_true",
                   help="ignore --warmup and use the steadiest block that fits; "
                        "exploratory only, it can exclude the drift you are looking for")
    p.add_argument("--keep-stops", action="store_true",
                   help="include stopped samples when computing pace")
    p.add_argument("--from", dest="t_from", type=float, metavar="MIN",
                   help="selection start in minutes; overrides --warmup")
    p.add_argument("--to", dest="t_to", type=float, metavar="MIN",
                   help="selection end in minutes; overrides --duration")
    p.add_argument("--html", metavar="PATH", help="also write a self-contained HTML report")
    p.add_argument("--interactive", metavar="PATH",
                   help="write a drag-to-select analysis page, in the spirit of "
                        "TrainingPeaks' Analyze tool")
    p.add_argument("--webapp", metavar="PATH",
                   help="write the standalone drop-a-file analyser page and exit; "
                        "no activity file is needed")
    p.add_argument("--step", type=int, default=1, metavar="N",
                   help="keep every Nth sample in the interactive page; trades a "
                        "little accuracy for a smaller file (default: 1)")
    p.add_argument("--no-sweep", dest="sweep", action="store_false",
                   help="skip the robustness sweep and judge from one block only")
    p.add_argument("--brief", action="store_true",
                   help="print only the answer: valid or not, and which side of the threshold")
    p.add_argument("--quiet", action="store_true", help="suppress the text report")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.webapp:
        with open(args.webapp, "w", encoding="utf-8") as fh:
            fh.write(wrap_document(webapp(target=args.target, warmup_min=args.warmup,
                                          duration_min=args.duration)))
        print(f"Analyser page written to {args.webapp}")
        return 0

    if not args.file:
        parser.error("an activity file is required (or use --webapp PATH)")

    try:
        track = Track.load(args.file)
    except (TrackError, OSError) as exc:
        print(f"hrdrift: {exc}", file=sys.stderr)
        return 2

    try:
        analysis = analyse(
            track,
            warmup_min=args.warmup,
            duration_min=args.duration,
            target=args.target,
            drop_stops=not args.keep_stops,
            auto=args.auto,
            rolling_min=args.rolling,
            t_from=args.t_from,
            t_to=args.t_to,
            sweep=args.sweep,
        )
    except ValueError as exc:
        print(f"hrdrift: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(brief_report(analysis) if args.brief else text_report(analysis))

    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(wrap_document(html_report(analysis)))
        print(f"HTML report written to {args.html}")

    if args.interactive:
        with open(args.interactive, "w", encoding="utf-8") as fh:
            fh.write(wrap_document(interactive_report(
                track, target=args.target, warmup_min=args.warmup,
                duration_min=args.duration, step=max(1, args.step))))
        print(f"Interactive page written to {args.interactive}")

    # Exit 1 when the effort was above threshold, so the tool composes in scripts.
    # 0 below threshold, 1 above, 3 not a valid test, 2 error (returned above).
    if not analysis.usability.usable:
        return 3
    rate = analysis.sweep.median if analysis.sweep_speaks else analysis.primary.rate
    return 1 if rate > analysis.target else 0


if __name__ == "__main__":
    sys.exit(main())
