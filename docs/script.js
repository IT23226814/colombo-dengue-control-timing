"use strict";
// Seasonal SEIR model for dengue in Colombo District, solved with classical RK4.
// A line-by-line port of src/seir.py (one-day step, intervention multiplier held
// at its value at each step's midpoint), with reported cases = rho * infections + bg.

// ---------- Fitted values: the highest-posterior MCMC sample of the 2026 calibration ----------
const FIT = {
  N: 2375415,                 // Colombo District, Census 2024
  sigma: 7 / 5.9,             // 1/week: incubation period 5.9 days
  gamma: 7 / 5.0,             // 1/week: infectious period 5 days
  Re: 1.9760186245594058,     // effective reproduction number at the start of the year
  amp: 0.5759169862233384,    // seasonal amplitude a
  phase: 33.0680893086082,    // week of peak transmission
  rho: 0.1177711382647448,    // reporting rate
  s0: 0.05949991930735215,    // initial susceptible fraction
  i0: 1837.479616271096,      // initial infectious people
  bg: 243.2693273559676,      // background reported cases per week
};
// Reference scenario, computed in the notebook (Section 5.5) with the same parameters:
// a 40% cut sustained from week 21 to the end of the year prevents this many reported cases.
const NOTEBOOK_REFERENCE = { start: 21, reduction: 0.40, prevented: 4665.155615190242, tolerance: 0.01 };
const WEEKS = 52;
const STEPS_PER_WEEK = 7;
const OBSERVED_2026 = [525, 634, 616, 539, 420, 399, 425, 416, 357, 365, 301, 392, 351, 313, 323,
  288, 370, 370, 378, 356, 505, 550, 733, 1201, 1292, 1178, 1668, 1634, 1536, 1448, 1008, 742,
  625, 441, 295, 222, 203, 267, 232, 247, 246, 233, 300, 261, 306];

// ---------- Model ----------
function rhs(t, y, p, u) {
  const [S, E, I] = y;
  const beta = p.beta0 * (1 + p.amp * Math.cos(2 * Math.PI * (t - p.phase) / 52)) * u;
  const inf = beta * S * I / p.N;
  return [-inf, inf - p.sigma * E, p.sigma * E - p.gamma * I, p.gamma * I, p.sigma * E];
}

function rk4Step(t, y, h, p, u) {
  const add = (a, b, s) => a.map((v, i) => v + s * b[i]);
  const k1 = rhs(t, y, p, u);
  const k2 = rhs(t + h / 2, add(y, k1, h / 2), p, u);
  const k3 = rhs(t + h / 2, add(y, k2, h / 2), p, u);
  const k4 = rhs(t + h, add(y, k3, h), p, u);
  return y.map((v, i) => v + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// Weekly reported cases (with background) for parameters p; start/end in weeks, reduction 0..1.
function simulate(p) {
  const h = 1 / STEPS_PER_WEEK;
  const e0 = p.i0 * p.gamma / p.sigma;               // E0 = I0 gamma / sigma, as in the calibration
  const S0 = p.s0 * p.N - e0 - p.i0;
  let y = [S0, e0, p.i0, p.N - S0 - e0 - p.i0, 0];
  const C = [0];
  for (let k = 0; k < WEEKS * STEPS_PER_WEEK; k++) {
    const t = k * h;
    const tm = t + h / 2;
    const u = (tm >= p.start && tm < p.end) ? 1 - p.reduction : 1;
    y = rk4Step(t, y, h, p, u);
    if ((k + 1) % STEPS_PER_WEEK === 0) C.push(y[4]);
  }
  const reported = [];
  for (let w = 0; w < WEEKS; w++) reported.push(p.rho * (C[w + 1] - C[w]) + p.bg);
  return reported;
}

// Same scenario computed here, independent of the sliders.
function referenceCheck() {
  const p = { ...FIT, beta0: FIT.Re * FIT.gamma / FIT.s0, start: Infinity, end: Infinity, reduction: 0 };
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const none = sum(simulate(p));
  const cut = sum(simulate({ ...p, start: NOTEBOOK_REFERENCE.start - 1, reduction: NOTEBOOK_REFERENCE.reduction }));
  const demo = none - cut;
  const relDiff = Math.abs(demo - NOTEBOOK_REFERENCE.prevented) / NOTEBOOK_REFERENCE.prevented;
  return { demo, notebook: NOTEBOOK_REFERENCE.prevented, relDiff, ok: relDiff <= NOTEBOOK_REFERENCE.tolerance };
}

// ---------- UI ----------
const $ = (id) => document.getElementById(id);
const sliders = ["start", "reduction", "duration", "re", "amp", "phase", "speed"];
const DEFAULTS = { start: 21, reduction: 40, duration: 52, re: round(FIT.Re, 2), amp: round(FIT.amp, 2),
                   phase: Math.round(FIT.phase * 2) / 2, speed: 6 };
const fmt = (x) => Math.round(x).toLocaleString("en-US");
function round(x, d) { return Math.round(x * 10 ** d) / 10 ** d; }

let base = [], withInt = [], shown = WEEKS, playing = false, lastFrame = 0, acc = 0;
// Until a transmission slider is moved, its exact fitted value is used (the slider shows it rounded).
let touched = { re: false, amp: false, phase: false };

function params() {
  const start = +$("start").value, dur = +$("duration").value;
  const re = touched.re ? +$("re").value : FIT.Re;
  const amp = touched.amp ? +$("amp").value : FIT.amp;
  const phase = touched.phase ? +$("phase").value : FIT.phase;
  const common = { ...FIT, Re: re, beta0: re * FIT.gamma / FIT.s0, amp, phase };
  return {
    base: { ...common, start: Infinity, end: Infinity, reduction: 0 },
    int: { ...common, start: start - 1, end: Math.min(start - 1 + dur, WEEKS) >= WEEKS ? Infinity : start - 1 + dur,
           reduction: +$("reduction").value / 100 },
  };
}

function labels() {
  const start = +$("start").value, dur = +$("duration").value;
  $("startVal").textContent = `week ${start}`;
  $("reductionVal").textContent = `${$("reduction").value}%`;
  $("durationVal").textContent = start - 1 + dur >= WEEKS ? "rest of the year" : `${dur} weeks (to week ${start + dur - 1})`;
  const fitted = (id) => (touched[id] ? "" : " (fitted)");
  $("reVal").textContent = (touched.re ? +$("re").value : FIT.Re).toFixed(2) + fitted("re");
  $("ampVal").textContent = (touched.amp ? +$("amp").value : FIT.amp).toFixed(2) + fitted("amp");
  $("phaseVal").textContent = `week ${(touched.phase ? +$("phase").value : FIT.phase).toFixed(1)}${fitted("phase")}`;
  $("speedVal").textContent = `${$("speed").value} weeks/s`;
}

function recompute() {
  labels();
  const p = params();
  base = simulate(p.base);
  withInt = simulate(p.int);
  draw();
  stats();
}

function stats() {
  const n = Math.max(1, Math.floor(shown));
  const sum = (a) => a.slice(0, n).reduce((s, v) => s + v, 0);
  const tb = sum(base), ti = sum(withInt);
  const outbreak = tb - FIT.bg * n;
  $("weekNow").textContent = `${n} of ${WEEKS}`;
  $("totBase").textContent = fmt(tb);
  $("totInt").textContent = fmt(ti);
  $("prevented").textContent = fmt(tb - ti);
  $("share").textContent = outbreak > 0 ? `${(100 * (tb - ti) / outbreak).toFixed(1)}%` : "–";
  const argmax = (a) => a.indexOf(Math.max(...a)) + 1;
  $("peak").textContent = `${argmax(base)} → ${argmax(withInt)}`;
}

function draw() {
  const cv = $("plot"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, m = { l: 64, r: 16, t: 16, b: 44 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const yMax = Math.max(...base, ...withInt, ...OBSERVED_2026) * 1.08;
  const step = niceStep(yMax / 6);
  const X = (w) => m.l + (w - 0.5) / WEEKS * pw;     // week w plotted at its centre
  const Y = (v) => m.t + ph - v / yMax * ph;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "13px system-ui, sans-serif";

  // intervention window
  const p = params().int;
  if (p.reduction > 0) {
    const x0 = m.l + p.start / WEEKS * pw, x1 = m.l + Math.min(p.end, WEEKS) / WEEKS * pw;
    ctx.fillStyle = "rgba(31,111,139,0.10)";
    ctx.fillRect(x0, m.t, x1 - x0, ph);
  }
  // grid and axes
  ctx.strokeStyle = "#e5e9ef"; ctx.fillStyle = "#5d6876"; ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let v = 0; v <= yMax; v += step) {
    ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(W - m.r, Y(v)); ctx.stroke();
    ctx.fillText(fmt(v), m.l - 6, Y(v));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let w = 1; w <= WEEKS; w += (w === 1 ? 4 : 5)) ctx.fillText(w, X(w), m.t + ph + 6);
  ctx.fillText("Week of 2026", m.l + pw / 2, H - 18);
  ctx.save(); ctx.translate(14, m.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("Reported cases per week", 0, -6); ctx.restore();
  ctx.strokeStyle = "#9aa4b1";
  ctx.beginPath(); ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, m.t + ph); ctx.lineTo(W - m.r, m.t + ph); ctx.stroke();

  // observed data
  ctx.fillStyle = "#c0392b";
  OBSERVED_2026.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i + 1), Y(v), 3.2, 0, 2 * Math.PI); ctx.fill(); });

  // model curves up to the week shown
  const n = Math.max(1, Math.floor(shown));
  line(ctx, base.slice(0, n), X, Y, "#6b7280", 2.2, [6, 4]);
  line(ctx, withInt.slice(0, n), X, Y, "#1f6f8b", 2.6, []);
  if (n < WEEKS) {
    ctx.strokeStyle = "#1b2430"; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(n), m.t); ctx.lineTo(X(n), m.t + ph); ctx.stroke(); ctx.setLineDash([]);
  }
}

function line(ctx, ys, X, Y, colour, width, dash) {
  ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath();
  ys.forEach((v, i) => (i ? ctx.lineTo(X(i + 1), Y(v)) : ctx.moveTo(X(i + 1), Y(v))));
  ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
}

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x)), f = x / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

// ---------- Animation ----------
function frame(ts) {
  if (!playing) return;
  if (lastFrame) acc += (ts - lastFrame) / 1000 * +$("speed").value;
  lastFrame = ts;
  shown = Math.min(WEEKS, 1 + acc);
  draw(); stats();
  if (shown >= WEEKS) { playing = false; buttons(); return; }
  requestAnimationFrame(frame);
}
function buttons() {
  $("startBtn").disabled = playing;
  $("pauseBtn").disabled = !playing;
}
$("startBtn").addEventListener("click", () => {
  if (shown >= WEEKS) { shown = 1; acc = 0; }
  playing = true; lastFrame = 0; buttons(); requestAnimationFrame(frame);
});
$("pauseBtn").addEventListener("click", () => { playing = false; buttons(); });
$("resetBtn").addEventListener("click", () => {
  playing = false; shown = 1; acc = 0; buttons(); draw(); stats();
});
$("fittedBtn").addEventListener("click", () => {
  for (const [k, v] of Object.entries(DEFAULTS)) if (k !== "speed") $(k).value = v;
  touched = { re: false, amp: false, phase: false };
  recompute();
});
sliders.forEach((id) => $(id).addEventListener("input", () => {
  if (id in touched) touched[id] = true;
  recompute();
}));

for (const [k, v] of Object.entries(DEFAULTS)) $(k).value = v;
buttons();
recompute();

const check = referenceCheck();
$("refDemo").textContent = check.demo.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
$("refNotebook").textContent = check.notebook.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
$("refDiff").textContent = check.relDiff < 1e-6 ? "< 0.0001%" : `${(100 * check.relDiff).toFixed(4)}%`;
$("refStatus").textContent = check.ok ? "✓ within the 1% tolerance" : "✗ outside the 1% tolerance";
$("refStatus").className = check.ok ? "ok" : "bad";

// Exposed for checking against the notebook (see README): run simulate() with any parameters.
window.dengueModel = { FIT, NOTEBOOK_REFERENCE, simulate, referenceCheck };
