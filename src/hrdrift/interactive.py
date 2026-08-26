"""Drag-to-select analysis page. Streams are embedded; metrics recompute in
the browser as the selection changes.
"""

from __future__ import annotations

import json
from html import escape

import numpy as np

from .analysis import BIN_EPSILON, DEFAULT_TARGET, MIN_BIN_SAMPLES, SPEED_BIN_MS
from .track import NGP_WINDOW_S, STOP_SPEED, YARDS_PER_METRE, Track

__all__ = ["interactive_report"]


def _streams(track: Track, step: int) -> dict:
    idx = np.arange(0, len(track.t), step)
    return {
        "t": [int(v) for v in track.t[idx]],
        "hr": [round(float(v), 1) for v in track.hr[idx]],
        # full precision: these feed the stop cutoff and the bin floor, and
        # rounding moves samples across those edges
        "v": [float(v) for v in track.speed[idx]],
        "gv": [float(v) for v in track.gap_speed[idx]],
        "d": [round(float(v), 1) for v in track.dist[idx]],
        "grade": [round(float(v) * 100, 2) for v in track.grade[idx]],
    }


def interactive_report(track: Track, target: float = DEFAULT_TARGET,
                       warmup_min: float = 15.0, duration_min: float = 60.0,
                       step: int = 1) -> str:
    """Render a self-contained interactive analysis page."""
    data = _streams(track, step)
    config = {
        "target": target,
        "warmup": warmup_min,
        "duration": duration_min,
        "stopSpeed": STOP_SPEED,
        "ngpWindow": NGP_WINDOW_S,
        "yardsPerMetre": YARDS_PER_METRE,
        "binMs": SPEED_BIN_MS,
        "binEps": BIN_EPSILON,
        "minBin": MIN_BIN_SAMPLES,
        "sampleStep": step,
        "name": track.name,
        "start": track.start.strftime("%Y-%m-%d %H:%M UTC"),
        "durationMin": round(track.duration_min, 1),
        "distanceKm": round(track.distance_km, 2),
    }
    payload = json.dumps({"cfg": config, "s": data}, separators=(",", ":"))
    return _TEMPLATE.replace("__PAYLOAD__", payload).replace("__NAME__", escape(track.name))


_TEMPLATE = r"""<title>HR Drift Analyze</title>
<style>
 :root { --bg:#fff; --fg:#1a1a1a; --muted:#6b7280; --line:#e5e7eb; --card:#f8fafc;
         --hr:#e05252; --sp:#3b82f6; --ok:#16a34a; --bad:#dc2626; --warn:#d97706; }
 @media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
   --bg:#0f1115; --fg:#e8e8e8; --muted:#9aa3af; --line:#262b33; --card:#161a20; } }
 :root[data-theme=dark] { --bg:#0f1115; --fg:#e8e8e8; --muted:#9aa3af; --line:#262b33; --card:#161a20; }
 * { box-sizing:border-box; }
 body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:1040px; padding:28px 20px 60px;
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
 h1 { font-size:21px; margin:0 0 3px; }
 .meta { color:var(--muted); font-size:14px; margin-bottom:20px; }
 .hint { color:var(--muted); font-size:13px; margin:10px 0 14px; }
 .chartwrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }
 svg { display:block; touch-action:none; cursor:crosshair; }
 .presets { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }
 button { font:inherit; font-size:13px; padding:6px 13px; border-radius:6px; cursor:pointer;
   border:1px solid var(--line); background:var(--card); color:var(--fg); }
 button:hover { border-color:var(--muted); }
 button.on { background:var(--fg); color:var(--bg); border-color:var(--fg); }
 .verdict { border-radius:8px; padding:16px 20px; background:var(--card); border-left:4px solid var(--ok);
   margin:18px 0; }
 .verdict.over { border-left-color:var(--bad); }
 .verdict.warn { border-left-color:var(--warn); }
 .big { font-size:29px; font-weight:600; letter-spacing:-.01em; }
 .sub { color:var(--muted); font-size:14px; }
 .note { color:var(--muted); font-size:13px; margin:9px 0 0; }
 .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:1px;
   background:var(--line); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
 .cell { background:var(--bg); padding:11px 14px; }
 .cell .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
 .cell .v { font-size:19px; font-variant-numeric:tabular-nums; margin-top:2px; }
 .cell .x { color:var(--muted); font-size:12px; }
 h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
   border-bottom:1px solid var(--line); padding-bottom:6px; margin:32px 0 4px; }
 table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
 th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); font-size:14px; }
 th:first-child, td:first-child { text-align:left; }
 td.dim { color:var(--muted); }
</style>
<h1>Analyze &mdash; __NAME__</h1>
<div class="meta" id="meta"></div>
<div class="chartwrap"><svg id="chart" width="1000" height="260"></svg></div>
<div class="hint">Drag across the chart to select a range. Shift-drag to nudge the edges of an
  existing selection; double-click to reset.</div>
<div class="presets" id="presets"></div>
<div class="verdict" id="verdict"></div>
<div class="grid" id="stats"></div>
<h2>Halves</h2>
<table id="halves"></table>
<script>
const P = __PAYLOAD__;
const S = P.s, CFG = P.cfg;
const N = S.t.length;

/* ---------- metric helpers, mirroring the Python implementation ---------- */

const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);

function rollingMean(a, win) {
  // centred, edge-padded; matches moving_average()
  const n = a.length, out = new Float64Array(n);
  if (win <= 1 || !n) return Array.from(a);
  const w = Math.min(win, n), left = w >> 1;
  let sum = 0;
  const at = i => a[Math.min(n - 1, Math.max(0, i))];
  for (let i = -left; i < w - left; i++) sum += at(i);
  for (let i = 0; i < n; i++) {
    out[i] = sum / w;
    sum += at(i + w - left) - at(i - left);
  }
  return out;
}

function ngp(gv) {
  // 30 s rolling speed, fourth-power weighted
  if (!gv.length) return NaN;
  const win = Math.max(1, Math.round(CFG.ngpWindow / CFG.sampleStep));
  const r = rollingMean(gv, win);
  let s = 0;
  for (const v of r) s += v ** 4;
  return Math.pow(s / r.length, 0.25);
}

function slice(i0, i1) {
  // half-open, same as the Python
  const idx = [];
  for (let i = i0; i < i1; i++) if (S.v[i] >= CFG.stopSpeed) idx.push(i);
  return idx;
}

function statsFor(idx) {
  if (idx.length < 2) return null;
  const hr = idx.map(i => S.hr[i]), gv = idx.map(i => S.gv[i]), v = idx.map(i => S.v[i]);
  const mHr = mean(hr), n = ngp(gv);
  return {
    n: idx.length, hr: mHr, speed: mean(v), gapSpeed: mean(gv), ngp: n,
    ef: n / mHr, efTp: (n / mHr) * 60 * CFG.yardsPerMetre,
    dist: S.d[idx[idx.length - 1]] - S.d[idx[0]],
    secs: (S.t[idx[idx.length - 1]] - S.t[idx[0]]),
    grade: mean(idx.map(i => S.grade[i])),
  };
}

function hrStability(idx) {
  const hr = idx.map(i => S.hr[i]), t = idx.map(i => S.t[i]);
  const m = mean(hr), mt = mean(t);
  let num = 0, den = 0;
  for (let i = 0; i < hr.length; i++) { num += (t[i] - mt) * (hr[i] - m); den += (t[i] - mt) ** 2; }
  const slope = (den ? num / den : 0) * 3600;
  const within = k => hr.filter(x => Math.abs(x - m) <= k).length / hr.length * 100;
  return { slope, within3: within(3), within5: within(5),
           pinned: within(5) >= 60 && Math.abs(slope) <= 5 };
}

function speedMatched(a, b, baseHr) {
  // compare HR only inside speed bins present in both halves
  const bin = x => Math.floor(x / CFG.binMs + CFG.binEps);
  const group = idx => {
    const m = new Map();
    for (const i of idx) {
      const k = bin(S.gv[i]);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(S.hr[i]);
    }
    return m;
  };
  const g1 = group(a), g2 = group(b);
  const minN = Math.max(3, Math.round(CFG.minBin / CFG.sampleStep));
  let num = 0, den = 0, used = 0, lo = Infinity, hi = -Infinity;
  for (const [k, h1] of g1) {
    const h2 = g2.get(k);
    if (!h2 || h1.length < minN || h2.length < minN) continue;
    const w = Math.min(h1.length, h2.length);
    num += w * (mean(h2) - mean(h1));
    den += w; used += h1.length + h2.length;
    lo = Math.min(lo, k * CFG.binMs); hi = Math.max(hi, (k + 1) * CFG.binMs);
  }
  if (!den || !baseHr) return null;
  const delta = num / den;
  return { deltaBpm: delta, pct: delta / baseHr * 100,
           coverage: used / (a.length + b.length) * 100, lo, hi };
}

function analyse(i0, i1) {
  const idx = slice(i0, i1);
  if (idx.length < 120 / CFG.sampleStep) return null;
  const half = idx.length >> 1;
  const a = idx.slice(0, half), b = idx.slice(-half);
  const first = statsFor(a), second = statsFor(b), whole = statsFor(idx);
  const blockMin = (S.t[i1] - S.t[i0]) / 60;
  const paHr = (first.ef - second.ef) / first.ef * 100;
  const sm = speedMatched(a, b, first.hr);
  const stab = hrStability(idx);
  const rate = x => x * 60 / blockMin;

  const matchedUsable = sm && sm.coverage >= 40;
  const usingMatched = !stab.pinned && matchedUsable;
  const primaryPct = usingMatched ? sm.pct : paHr;
  return { idx, first, second, whole, blockMin, paHr, sm, stab, usingMatched,
           matchedUsable, primaryPct, primaryRate: rate(primaryPct),
           paHrRate: rate(paHr), smRate: sm ? rate(sm.pct) : NaN,
           disagree: usingMatched && sm && Math.abs(rate(sm.pct) - rate(paHr)) > 2 };
}

// --- formatting

const pace = mps => {
  if (!(mps > 0) || !isFinite(mps)) return "--:--";
  const m = 1000 / mps / 60, w = Math.floor(m);
  let s = Math.round((m - w) * 60);
  return s === 60 ? `${w + 1}:00` : `${w}:${String(s).padStart(2, "0")}`;
};
const hms = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const sign = x => (x >= 0 ? "+" : "") + x.toFixed(2);

// --- chart

const svg = document.getElementById("chart");
const W = 1000, H = 260, PAD_B = 24, PAD_T = 8;
const tMin = S.t[0], tMax = S.t[N - 1];
const xOf = t => (t - tMin) / (tMax - tMin) * W;
const tOf = x => tMin + Math.max(0, Math.min(1, x / W)) * (tMax - tMin);
const iOf = t => Math.max(0, Math.min(N - 1, Math.round((t - tMin) / (tMax - tMin) * (N - 1))));

function series(vals, colour, smooth) {
  const a = smooth ? rollingMean(vals, Math.max(1, Math.round(smooth / CFG.sampleStep))) : vals;
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = Math.max(hi - lo, 1e-9);
  const h = H - PAD_B - PAD_T;
  const stride = Math.max(1, Math.floor(N / 1400));
  let d = "";
  for (let i = 0; i < N; i += stride)
    d += `${i ? "L" : "M"}${xOf(S.t[i]).toFixed(1)},${(PAD_T + h - (a[i] - lo) / span * h).toFixed(1)}`;
  return { d, lo, hi, colour };
}

const hrSeries = series(S.hr, "var(--hr)", 15);
const spSeries = series(S.gv, "var(--sp)", 30);

let sel = null;   // {i0, i1}

function draw() {
  const parts = [];
  if (sel) {
    const x0 = xOf(S.t[sel.i0]), x1 = xOf(S.t[sel.i1]), xm = (x0 + x1) / 2;
    parts.push(`<rect x="${x0}" y="0" width="${x1 - x0}" height="${H}" fill="#94a3b8" opacity="0.16"/>`);
    parts.push(`<line x1="${xm}" y1="0" x2="${xm}" y2="${H}" stroke="#64748b" stroke-dasharray="4 4"/>`);
    for (const x of [x0, x1])
      parts.push(`<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="#64748b" stroke-width="1.5"/>`);
  }
  for (const s of [spSeries, hrSeries])
    parts.push(`<path d="${s.d}" fill="none" stroke="${s.colour}" stroke-width="1.7" stroke-linejoin="round"/>`);
  for (let m = 0; m <= tMax / 60; m += 10)
    parts.push(`<text x="${xOf(m * 60)}" y="${H - 7}" font-size="11" fill="currentColor"
      opacity=".5" text-anchor="middle">${m}'</text>`);
  svg.innerHTML = parts.join("");
}

// --- interaction

let dragFrom = null;
const xAt = ev => {
  const r = svg.getBoundingClientRect();
  return (ev.clientX - r.left) * (W / r.width);
};

svg.addEventListener("pointerdown", ev => {
  svg.setPointerCapture(ev.pointerId);
  dragFrom = tOf(xAt(ev));
});
svg.addEventListener("pointermove", ev => {
  if (dragFrom === null) return;
  const t = tOf(xAt(ev));
  setSelection(Math.min(dragFrom, t), Math.max(dragFrom, t), null);
});
svg.addEventListener("pointerup", () => { dragFrom = null; });
svg.addEventListener("dblclick", () => applyPreset(presets[0]));

function setSelection(t0, t1, presetName) {
  if (t1 - t0 < 60) return;
  sel = { i0: iOf(t0), i1: iOf(t1) };
  document.querySelectorAll("#presets button").forEach(b =>
    b.classList.toggle("on", b.dataset.name === presetName));
  draw();
  render();
}

const presets = [
  { name: "Protocol", from: CFG.warmup, to: CFG.warmup + CFG.duration },
  { name: "Last 60 min", from: Math.max(0, CFG.durationMin - 60), to: CFG.durationMin },
  { name: "Skip 10 min", from: 10, to: CFG.durationMin },
  { name: "Whole activity", from: 0, to: CFG.durationMin },
];

function applyPreset(p) {
  setSelection(Math.min(p.from, CFG.durationMin) * 60,
               Math.min(p.to, CFG.durationMin) * 60, p.name);
}

document.getElementById("presets").innerHTML = presets
  .map(p => `<button data-name="${p.name}">${p.name}</button>`).join("");
document.querySelectorAll("#presets button").forEach((b, i) =>
  b.addEventListener("click", () => applyPreset(presets[i])));

// --- render

document.getElementById("meta").textContent =
  `${CFG.start} · ${CFG.durationMin} min · ${CFG.distanceKm} km · ` +
  `heart rate ${hrSeries.lo.toFixed(0)}–${hrSeries.hi.toFixed(0)} bpm`;

function cell(k, v, x) {
  return `<div class="cell"><div class="k">${k}</div><div class="v">${v}</div>` +
         (x ? `<div class="x">${x}</div>` : "") + `</div>`;
}

function render() {
  const a = analyse(sel.i0, sel.i1);
  const V = document.getElementById("verdict");
  if (!a) {
    V.className = "verdict warn";
    V.innerHTML = `<div class="big">Selection too short</div>
      <div class="sub">Drag a longer range — drift needs at least a few minutes to mean anything.</div>`;
    document.getElementById("stats").innerHTML = "";
    document.getElementById("halves").innerHTML = "";
    return;
  }

  const over = a.primaryRate > CFG.target;
  const short = a.blockMin < 45;
  const label = a.usingMatched ? "speed-matched drift" : "Pa:Hr decoupling";
  let head, body;
  if (over) { head = "ABOVE AeT"; body = `Drift of ${a.primaryRate.toFixed(1)}%/h exceeds the
    ${CFG.target}%/h threshold. The heart rate held was above the aerobic threshold. Retest lower.`; }
  else if (a.primaryRate < CFG.target * 0.4) { head = "WELL BELOW AeT";
    body = `Drift of ${a.primaryRate.toFixed(1)}%/h is far under the ${CFG.target}%/h threshold.
    Retest meaningfully higher.`; }
  else { head = "AT OR JUST BELOW AeT"; body = `Drift of ${a.primaryRate.toFixed(1)}%/h is under the
    ${CFG.target}%/h threshold. This heart rate is at or just below AeT.`; }

  let notes = "";
  if (a.disagree) notes += `<p class="note">Pa:Hr reads ${sign(a.paHrRate)}%/h, but heart rate was not
    pinned (${a.stab.slope >= 0 ? "+" : ""}${a.stab.slope.toFixed(0)} bpm/h), so that ratio credits the
    faster half. The speed-matched figure compares heart rate only within equal grade-adjusted speed
    bins and is the one reported.</p>`;
  if (a.stab.pinned) notes += `<p class="note">Heart rate was pinned, so drift shows up as pace decay
    and Pa:Hr is the right instrument here.</p>`;
  if (!a.matchedUsable && !a.stab.pinned) notes += `<p class="note">The two halves barely overlap in
    speed, so the speed-matched cross-check is unavailable and Pa:Hr is reported.</p>`;
  if (short) notes += `<p class="note">Selection is ${a.blockMin.toFixed(0)} min. The 5% criterion is
    defined for 60 minutes; drift is scaled to %/hour, but a short block is noisy.</p>`;

  V.className = "verdict" + (over ? " over" : short ? " warn" : "");
  V.innerHTML = `<div class="big">${sign(a.primaryRate)}%/h drift</div>
    <div class="sub">${head} · threshold ${CFG.target}%/h · ${label}</div>
    <p style="margin:8px 0 0">${body}</p>${notes}`;

  const w = a.whole;
  document.getElementById("stats").innerHTML = [
    cell("Selection", `${hms(S.t[sel.i0])}–${hms(S.t[sel.i1])}`, `${a.blockMin.toFixed(1)} min`),
    cell("Distance", `${(w.dist / 1000).toFixed(2)} km`, `${(w.n * CFG.sampleStep / 60).toFixed(0)} min moving`),
    cell("Avg heart rate", `${w.hr.toFixed(1)}`, "bpm"),
    cell("Avg pace", `${pace(w.speed)}/km`, "raw"),
    cell("NGP", `${pace(w.ngp)}/km`, "normalised graded pace"),
    cell("Efficiency factor", `${w.efTp.toFixed(3)}`, "yd/min per bpm"),
    cell("Pa:Hr", `${sign(a.paHrRate)}%/h`, `${sign(a.paHr)}% over block`),
    cell("Speed-matched", a.sm ? `${sign(a.smRate)}%/h` : "n/a",
         a.sm ? `${a.sm.deltaBpm >= 0 ? "+" : ""}${a.sm.deltaBpm.toFixed(1)} bpm · ${a.sm.coverage.toFixed(0)}% cover` : "no overlap"),
    cell("HR trend", `${a.stab.slope >= 0 ? "+" : ""}${a.stab.slope.toFixed(0)}`, "bpm/hour"),
    cell("HR steadiness", `${a.stab.within5.toFixed(0)}%`, "within 5 bpm · " + (a.stab.pinned ? "pinned" : "free")),
    cell("Avg grade", `${w.grade >= 0 ? "+" : ""}${w.grade.toFixed(1)}%`, ""),
  ].join("");

  const f = a.first, s = a.second;
  const row = (k, x, y, d, dim) =>
    `<tr><td>${k}</td><td>${x}</td><td>${y}</td><td class="${dim ? "dim" : ""}">${d}</td></tr>`;
  document.getElementById("halves").innerHTML =
    `<tr><th></th><th>1st half</th><th>2nd half</th><th>change</th></tr>` +
    row("Heart rate", f.hr.toFixed(1), s.hr.toFixed(1),
        `${s.hr - f.hr >= 0 ? "+" : ""}${(s.hr - f.hr).toFixed(1)} bpm`) +
    row("Pace", pace(f.speed) + "/km", pace(s.speed) + "/km",
        `${sign((f.speed / s.speed - 1) * 100)}%`) +
    row("NGP", pace(f.ngp) + "/km", pace(s.ngp) + "/km",
        `${sign((f.ngp / s.ngp - 1) * 100)}%`) +
    row("Efficiency factor", f.efTp.toFixed(3), s.efTp.toFixed(3),
        `${sign((s.ef - f.ef) / f.ef * 100)}%`);
}

applyPreset(presets[0]);
</script>
"""
