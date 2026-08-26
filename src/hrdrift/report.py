"""Rendering of an :class:`~hrdrift.analysis.Analysis` as text or HTML."""

from __future__ import annotations

import math
from html import escape

import numpy as np

from .analysis import Analysis
from .track import moving_average

__all__ = ["format_pace", "text_report", "brief_report", "html_report", "wrap_document"]

WIDTH = 78
TAGS = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}

DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
{head}
</head>
<body>
{body}
</body>
</html>
"""


def wrap_document(fragment: str) -> str:
    """Wrap a rendered fragment in a complete HTML document."""
    marker = "</style>"
    cut = fragment.find(marker)
    if cut == -1:
        return DOCUMENT.format(head="", body=fragment)
    cut += len(marker)
    return DOCUMENT.format(head=fragment[:cut].strip(), body=fragment[cut:].strip())


def format_pace(minutes: float) -> str:
    """Format decimal minutes as ``m:ss``."""
    if not np.isfinite(minutes) or minutes <= 0:
        return "--:--"
    whole = int(minutes)
    seconds = int(round((minutes - whole) * 60))
    if seconds == 60:
        whole, seconds = whole + 1, 0
    return f"{whole}:{seconds:02d}"


def _wrap(text: str, width: int = WIDTH) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


# ------------------------------------------------------------------ text

def brief_report(a: Analysis) -> str:
    """The answer and nothing else: valid or not, and which side of 5%."""
    out: list[str] = []
    add = out.append
    u = a.usability

    add("")
    add("=" * WIDTH)
    add(f"  {a.answer}")
    out.extend(f"  {line}" for line in _wrap(a.result_line, WIDTH - 2))
    add("=" * WIDTH)
    if not u.usable:
        add("")
        add("  This run cannot answer the question:")
        for reason in u.reasons:
            head, *rest = _wrap(reason, WIDTH - 6)
            add(f"    - {head}")
            out.extend(f"      {line}" for line in rest)
        add("")
        out.extend(f"  {line}" for line in _wrap(
            "A drift test is 15 minutes of warm-up followed by 60 minutes of steady effort, "
            "flat or on a treadmill, holding either heart rate or pace constant.", WIDTH - 2))
    add("")
    return "\n".join(out)


def text_report(a: Analysis) -> str:
    out: list[str] = []
    add = out.append
    rule = lambda ch="-": add(ch * WIDTH)
    u = a.usability

    add("")
    rule("=")
    add(f"  {a.answer}")
    out.extend(f"  {line}" for line in _wrap(a.result_line, WIDTH - 2))
    rule("=")
    add(f"{a.track.name}   {a.track.start:%Y-%m-%d %H:%M} UTC   "
        f"{a.track.duration_min:.1f} min   {a.track.distance_km:.2f} km")

    if not u.usable:
        add("")
        add("WHY THIS RUN CANNOT ANSWER THE QUESTION")
        rule()
        for reason in u.reasons:
            head, *rest = _wrap(reason, WIDTH - 4)
            add(f"  - {head}")
            out.extend(f"    {line}" for line in rest)
        add("")
        out.extend(f"  {line}" for line in _wrap(
            "A drift test is 15 minutes of warm-up followed by 60 minutes of steady effort, "
            "flat ground or a treadmill, holding either heart rate or pace constant. The "
            "diagnostics below describe what was recorded instead.", WIDTH - 2))
        add("")
        add("DIAGNOSTICS")

    add("")
    add("PROTOCOL COMPLIANCE")
    rule()
    for check in a.checks:
        head, *rest = _wrap(check.message, WIDTH - 8)
        add(f"{TAGS[check.level]} {head}")
        out.extend(f"       {line}" for line in rest)

    add("")
    add(f"ANALYSIS BLOCK  {a.t_start/60:.0f}-{a.t_end/60:.0f} min "
        f"({a.block_min:.0f} min, stops excluded)")
    rule()
    add(f"{'':18}{'1st half':>13}{'2nd half':>13}{'change':>14}")
    f, s = a.first, a.second
    add(f"{'Heart rate':18}{f.hr:>12.1f}{s.hr:>13.1f}{s.hr - f.hr:>+13.1f} bpm")
    add(f"{'Pace':18}{format_pace(f.pace) + '/km':>13}{format_pace(s.pace) + '/km':>13}"
        f"{(s.pace - f.pace) / f.pace * 100:>+13.1f} %")
    add(f"{'Grade-adj pace':18}{format_pace(f.gap_pace) + '/km':>13}"
        f"{format_pace(s.gap_pace) + '/km':>13}"
        f"{(s.gap_pace - f.gap_pace) / f.gap_pace * 100:>+13.1f} %")
    add(f"{'Efficiency (EF)':18}{f.ef:>13.5f}{s.ef:>13.5f}"
        f"{(s.ef - f.ef) / f.ef * 100:>+13.1f} %")

    add("")
    add("DRIFT" if u.usable else "DRIFT (not reported as a result -- see above)")
    rule()
    for metric in (a.pa_hr, a.speed_matched):
        if metric is None:
            continue
        marker = " <- primary" if metric is a.primary else ""
        add(f"  {metric.label:<24}{metric.pct:>+8.2f} % over {metric.block_min:.0f} min"
            f"   = {metric.rate:>+6.2f} %/h{marker}")
        if metric.detail:
            out.extend(f"      {line}" for line in _wrap(metric.detail, WIDTH - 6))
    add(f"  {'threshold':<24}{a.target:>8.2f} %/h")

    if a.sweep is not None and a.sweep.n:
        sw = a.sweep
        add("")
        add(f"ROBUSTNESS  ({sw.n} block choices: warm-up 8-20 min x block 40-60 min)")
        rule()
        add(f"  {'median drift':<24}{sw.median:>+8.2f} %/h")
        add(f"  {'interquartile range':<24}{sw.p25:>+8.2f} to {sw.p75:+.2f} %/h"
            f"   (spread {sw.spread:.2f})")
        add(f"  {'above threshold':<24}{sw.fraction_above:>8.0f} % of choices")
        add(f"  {'agreement':<24}{sw.agreement:>8}")
        lo = min(sw.points, key=lambda p: p.rate)
        hi = max(sw.points, key=lambda p: p.rate)
        add(f"  lowest  {lo.rate:+6.2f} %/h at warm-up {lo.start_min:.0f}' block {lo.block_min:.0f}'")
        add(f"  highest {hi.rate:+6.2f} %/h at warm-up {hi.start_min:.0f}' block {hi.block_min:.0f}'")
        if sw.agreement == "unstable":
            out.extend(f"  ! {line}" for line in _wrap(
                "Where you draw the block moves the answer more than the threshold itself. "
                "That is a property of the effort, not of the analysis.", WIDTH - 4))

    if not u.usable:
        add("")
        return "\n".join(out)

    head, body = a.verdict
    add("")
    add(f"VERDICT: {head}")
    out.extend(f"  {line}" for line in _wrap(body))
    if a.metrics_disagree:
        out.extend(f"  {line}" for line in _wrap(
            "Based on the speed-matched figure. Pace was not held constant, so the Pa:Hr "
            "ratio credits the faster half and reads artificially low."))
    if a.sweep_speaks and a.sweep.agreement != "unstable":
        out.extend(f"  {line}" for line in _wrap(
            f"This block alone reads {a.primary.rate:+.2f}%/h ({a.single_verdict[0]}); the verdict "
            f"above is the median across all block choices, which is less sensitive to where the "
            f"warm-up is cut."))
    if not a.reliable:
        add("")
        out.extend(f"  ! {line}" for line in _wrap(
            "CAVEAT: the protocol checks above did not pass. On a loosely executed test the "
            "robustness spread matters more than any single number; read that before setting "
            "zones.", WIDTH - 4))

    add("")
    add("AEROBIC THRESHOLD ESTIMATE")
    rule()
    out.extend(f"  {line}" for line in _wrap(a.aet_estimate))

    if a.rolling:
        add("")
        add("ROLLING DRIFT  (where drift sets in)")
        rule()
        add(f"{'start':>7}{'avg HR':>10}{'%/h':>9}")
        step = max(1, len(a.rolling) // 14)
        for start, hr, rate in a.rolling[::step]:
            bar = "#" * int(min(30, max(0.0, rate)))
            add(f"{start/60:>6.0f}'{hr:>10.0f}{rate:>9.1f}   {bar}")

    add("")
    return "\n".join(out)


# ------------------------------------------------------------------ html

def _polyline(xs, ys, w, h, colour, width=1.8):
    ylo, yhi = float(np.nanmin(ys)), float(np.nanmax(ys))
    span = max(yhi - ylo, 1e-9)
    xlo, xhi = float(xs[0]), float(xs[-1])
    xspan = max(xhi - xlo, 1e-9)
    step = max(1, len(xs) // 1200)
    pts = " ".join(
        f"{(xs[i] - xlo) / xspan * w:.1f},{h - (ys[i] - ylo) / span * h:.1f}"
        for i in range(0, len(xs), step)
    )
    return (f'<polyline fill="none" stroke="{colour}" stroke-width="{width}" '
            f'stroke-linejoin="round" points="{pts}"/>'), ylo, yhi


def html_report(a: Analysis) -> str:
    """Render a self-contained HTML page. Returns the markup."""
    w_px, h_px = 900, 220
    minutes = a.track.t / 60
    hr_smooth = moving_average(a.track.hr, 15)
    gap = np.where(a.track.moving, a.track.gap_speed, np.nan)
    filled = np.nan_to_num(gap, nan=float(np.nanmean(gap)) if np.isfinite(np.nanmean(gap)) else 0.0)
    gap_smooth = moving_average(filled, 30)

    hr_line, hr_lo, hr_hi = _polyline(minutes, hr_smooth, w_px, h_px, "#e05252")
    sp_line, sp_lo, sp_hi = _polyline(minutes, gap_smooth, w_px, h_px, "#3b82f6")

    span = max(minutes[-1] - minutes[0], 1e-9)
    x0 = (a.t_start / 60 - minutes[0]) / span * w_px
    x1 = (a.t_end / 60 - minutes[0]) / span * w_px
    xm = (x0 + x1) / 2
    shade = (f'<rect x="{x0:.1f}" y="0" width="{x1-x0:.1f}" height="{h_px}" fill="#94a3b8" '
             f'opacity="0.13"/><line x1="{xm:.1f}" y1="0" x2="{xm:.1f}" y2="{h_px}" '
             f'stroke="#64748b" stroke-dasharray="4 4"/>')

    head, body = a.verdict
    over = a.primary.rate > a.target
    f, s = a.first, a.second

    rows = [
        ("Heart rate", f"{f.hr:.1f}", f"{s.hr:.1f}", f"{s.hr - f.hr:+.1f} bpm"),
        ("Grade-adj pace", f"{format_pace(f.gap_pace)}/km", f"{format_pace(s.gap_pace)}/km",
         f"{(s.gap_pace - f.gap_pace) / f.gap_pace * 100:+.1f}%"),
        ("Efficiency factor", f"{f.ef:.5f}", f"{s.ef:.5f}",
         f"{(s.ef - f.ef) / f.ef * 100:+.1f}%"),
        ("Pa:Hr decoupling", "", "", f"{a.pa_hr.rate:+.1f}%/h"),
    ]
    if a.speed_matched is not None:
        rows.append(("Speed-matched drift", "", "", f"{a.speed_matched.rate:+.1f}%/h"))
    table = "".join(
        f"<tr><td>{escape(n)}</td><td>{v1}</td><td>{v2}</td><td>{d}</td></tr>"
        for n, v1, v2, d in rows
    )

    note = ""
    if a.metrics_disagree:
        note = (f'<p class="note">Pa:Hr reads {a.pa_hr.rate:+.2f}%/h, but pace was not held constant, '
                f'so that ratio credits the faster half. The speed-matched figure compares heart rate '
                f'only within equal grade-adjusted speed bins and is the one to trust.</p>')
    caveat = ""
    if not a.reliable:
        caveat = ('<p class="note warn">The protocol checks below did not pass. Treat this as '
                  'indicative, not as a threshold measurement.</p>')

    checks = "".join(f'<li class="{c.level}"><b>{c.level.upper()}</b> {escape(c.message)}</li>'
                     for c in a.checks)
    roll = "".join(f"<tr><td>{st/60:.0f}'</td><td>{hr:.0f}</td><td>{rate:+.1f}%/h</td></tr>"
                   for st, hr, rate in a.rolling[::max(1, len(a.rolling) // 20)]) if a.rolling else ""

    return f"""<title>HR Drift Analysis</title>
<style>
 :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#6b7280; --line:#e5e7eb; --card:#f8fafc; }}
 @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
   --bg:#0f1115; --fg:#e8e8e8; --muted:#9aa3af; --line:#262b33; --card:#161a20; }} }}
 :root[data-theme=dark] {{ --bg:#0f1115; --fg:#e8e8e8; --muted:#9aa3af; --line:#262b33; --card:#161a20; }}
 body {{ background:var(--bg); color:var(--fg);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
   max-width:960px; margin:0 auto; padding:32px 20px; }}
 h1 {{ font-size:22px; margin:0 0 4px; }}
 .meta {{ color:var(--muted); margin-bottom:24px; }}
 h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
   border-bottom:1px solid var(--line); padding-bottom:6px; margin-top:34px; }}
 .verdict {{ background:var(--card); border-left:4px solid #16a34a; padding:16px 20px;
   border-radius:6px; }}
 .verdict.over {{ border-left-color:#dc2626; }}
 .big {{ font-size:30px; font-weight:600; }}
 .sub {{ color:var(--muted); margin-bottom:8px; }}
 .note {{ color:var(--muted); font-size:13px; margin:10px 0 0; }}
 .note.warn {{ color:#d97706; }}
 table {{ border-collapse:collapse; width:100%; margin-top:14px; font-variant-numeric:tabular-nums; }}
 th,td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }}
 th:first-child, td:first-child {{ text-align:left; }}
 ul {{ list-style:none; padding:0; }}
 li {{ padding:7px 12px; border-radius:4px; margin-bottom:5px; background:var(--card); }}
 li b {{ font-size:11px; letter-spacing:.05em; margin-right:8px; }}
 li.ok b {{ color:#16a34a; }} li.warn b {{ color:#d97706; }} li.fail b {{ color:#dc2626; }}
 .chartwrap {{ overflow-x:auto; }} svg {{ display:block; }}
 .legend {{ color:var(--muted); font-size:13px; margin-top:6px; }}
 .sw {{ display:inline-block; width:22px; height:3px; vertical-align:middle; margin-right:5px; }}
</style>
<h1>Heart rate drift analysis</h1>
<div class="meta">{escape(a.track.name)} &middot; {a.track.start:%Y-%m-%d %H:%M} UTC &middot;
  {a.track.duration_min:.0f} min &middot; {a.track.distance_km:.2f} km</div>
<div class="verdict{' over' if over else ''}">
  <div class="big">{a.primary.rate:+.2f}%/h drift</div>
  <div class="sub">{escape(head)} &middot; threshold {a.target:.0f}%/h &middot;
    {escape(a.primary.label)}</div>
  <p>{escape(body)}</p>{note}{caveat}
</div>
<table><tr><th></th><th>1st half</th><th>2nd half</th><th>change</th></tr>{table}</table>
<h2>Heart rate and grade-adjusted speed</h2>
<div class="chartwrap"><svg width="{w_px}" height="{h_px}" viewBox="0 0 {w_px} {h_px}">
{shade}{hr_line}{sp_line}</svg></div>
<div class="legend"><span class="sw" style="background:#e05252"></span>heart rate
  ({hr_lo:.0f}-{hr_hi:.0f} bpm) &nbsp;&nbsp;
  <span class="sw" style="background:#3b82f6"></span>grade-adjusted speed
  ({format_pace(1000 / max(sp_hi, 0.1) / 60)}-{format_pace(1000 / max(sp_lo, 0.1) / 60)}/km)
  &nbsp;&nbsp; shaded = analysis block, dashed = half split</div>
<h2>Protocol compliance</h2>
<ul>{checks}</ul>
<h2>Rolling drift</h2>
<table><tr><th>block start</th><th>avg HR</th><th>drift</th></tr>{roll}</table>
"""
