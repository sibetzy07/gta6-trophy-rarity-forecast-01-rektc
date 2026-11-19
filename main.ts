// Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.
//
// A trophy's rate is earners / players-tracked, and the two counts move for
// different reasons, so this program models them separately and divides at
// the target instead of extrapolating the rate itself:
//
//   players: new players join with no trophies, so growth dilutes every
//     rate. When the history shows the count moving, fit N ~ t^alpha (t =
//     time since release) over the last day and extrapolate it. When the
//     history is perfectly flat, the page is stale since launch: Exophase
//     has not refreshed it yet, and the next refresh catches up a launch
//     night's worth of joins, about one frozen count per hour of staleness.
//   earners: players earn a trophy at a hazard that decays like
//     1 / (hours since launch) — the easy trophies go early and the rate of
//     new earners slows. The hazard's scale is read off the history when it
//     moves (a person-time estimate over the last day, shrunk toward a
//     prior) and off the trophy's current rate and tier when the history is
//     flat (a trophy almost nobody has is earned slowly; the platinum,
//     which needs the whole game finished, slowest of all). Earners are
//     stepped forward in small steps, so dilution and progression combine
//     instead of one of them being assumed away. A trophy at ~100% stays
//     there: every new tracked player already has it.
//   spread: a logit-scale band around the median, wider when the page is
//     stale (a refresh may or may not land before the target, so the upper
//     tail reaches back toward the last read), wider for rare trophies (a
//     handful of earners is noisy), and wider the longer the horizon.
//
// No randomness anywhere, no clock reads, no network: two runs write the
// same bytes. Input: every *.in in $MAKEFUN_INPUT_DIR (else /input).
// Output: one <id>.out per case in $MAKEFUN_OUTPUT_DIR (else /output),
// exactly {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The five quantile levels, in the order each row lists them.
const QLEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// Standard-normal quantiles at those levels.
const ZNORM = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
// Hazard timescale: the earning hazard t hours after launch is lam0 / (t + 1).
const HAZARD_C_HOURS = 1;
// Prior weight for the hazard, in player-hours of non-earner exposure.
const PRIOR_EXPOSURE = 20000;
// Steps in the earners' walk from the last read to the target.
const ODE_STEPS = 200;

interface TrophyIn { id: number; tier?: string }
interface ReadIn { at: string; players_tracked?: number | null; rates?: Array<number | null> }
interface CaseIn {
  id?: string; at: string; target: string;
  game?: { release_date?: string };
  trophies: TrophyIn[];
  history?: ReadIn[];
}

const clampPct = (x: number): number => (x < 0.01 ? 0.01 : x > 99.99 ? 99.99 : x);
const logitPct = (p: number): number => { const c = clampPct(p); return Math.log(c / (100 - c)); };
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const toMs = (s: string | undefined, fallback: number): number => {
  if (s === undefined) return fallback;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : fallback;
};
// How fast a trophy of this tier is earned, relative to bronze, when the
// history says nothing. The platinum needs the whole game finished, so
// almost nobody earns it in the first hours; bronze is the everyday kind.
const tierSpeed = (tier: string | undefined): number =>
  tier === 'platinum' ? 0.1 : tier === 'gold' ? 0.4 : tier === 'silver' ? 0.8 : 1;

// Least squares of ln n on ln t; 0 when the fit is degenerate.
const powerLawAlpha = (points: Array<{ t: number; n: number }>): number => {
  let sx = 0, sy = 0, sxx = 0, sxy = 0, k = 0;
  for (const p of points) {
    const x = Math.log(Math.max(1, p.t));
    const y = Math.log(Math.max(1, p.n));
    sx += x; sy += y; sxx += x * x; sxy += x * y; k++;
  }
  if (k < 2) return 0;
  const den = k * sxx - sx * sx;
  if (den === 0) return 0;
  const slope = (k * sxy - sx * sy) / den;
  if (!Number.isFinite(slope)) return 0;
  return Math.min(1.5, Math.max(0, slope));
};

const forecastCase = (c: CaseIn): { trophies: Array<{ id: number; q: number[] }> } => {
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const history = (Array.isArray(c.history) ? c.history : [])
    .slice()
    .sort((a, b) => toMs(a.at, 0) - toMs(b.at, 0));
  const checkpointMs = toMs(c.at, 0);
  const targetMs = toMs(c.target, checkpointMs);
  const releaseMs = toMs(c.game?.release_date, checkpointMs);
  const last = history.length > 0 ? history[history.length - 1] : undefined;
  const lastMs = last !== undefined ? toMs(last.at, checkpointMs) : checkpointMs;
  const horizonMs = Math.max(0, targetMs - lastMs);
  const horizonDays = horizonMs / 86400000;

  // --- players tracked, per read (forward-filled where a read lacks it) ---
  const timesMs: number[] = [];
  const players: number[] = [];
  let lastPlayers = 1000;
  for (const h of history) {
    timesMs.push(toMs(h.at, checkpointMs));
    const n = typeof h.players_tracked === 'number' && h.players_tracked > 0 ? h.players_tracked : NaN;
    if (Number.isFinite(n)) lastPlayers = n;
    players.push(lastPlayers);
  }
  const nLast = players.length > 0 ? players[players.length - 1] : 1000;
  const nMin = players.length > 0 ? Math.min(...players) : nLast;
  const nMax = players.length > 0 ? Math.max(...players) : nLast;
  // A page that has not moved at all is stale, not settled: Exophase has
  // not refreshed it since launch, so the counts it shows are old.
  const flat = players.length === 0 || nMax < nMin * 1.02;
  const firstMs = timesMs.length > 0 ? timesMs[0] : lastMs;
  const nearLaunch = (firstMs - releaseMs) / 3600000 <= 48;

  // --- players at the target ---
  let nTarget: number;
  if (flat) {
    const staleHours = Math.max(0, (targetMs - firstMs) / 3600000);
    // Near launch the frozen count is the pre-launch trickle and each hour
    // after launch adds about that many again; far from launch a stale
    // page is a slow drift, not a launch pile-up.
    const ratio = nearLaunch ? 1 + staleHours : 1 + 0.05 * staleHours;
    nTarget = nLast * Math.min(ratio, nearLaunch ? 20 : 2);
  } else {
    const windowStart = lastMs - 24 * 3600000;
    const pts: Array<{ t: number; n: number }> = [];
    for (let k = 0; k < history.length; k++) {
      if (timesMs[k] < windowStart) continue;
      pts.push({ t: Math.max(1, (timesMs[k] - releaseMs) / 60000), n: players[k] });
    }
    const alpha = powerLawAlpha(pts);
    const tLast = Math.max(1, (lastMs - releaseMs) / 60000);
    const tTarget = Math.max(tLast, (targetMs - releaseMs) / 60000);
    const ratio = Math.pow(tTarget / tLast, alpha);
    nTarget = nLast * Math.min(Math.max(ratio, 1), 20);
  }
  if (!Number.isFinite(nTarget) || nTarget < nLast) nTarget = nLast;

  // --- rates per read, forward-filled, and earners per read ---
  // effRate[k][i] is trophy i's rate at read k, carrying the last read that
  // had it forward over reads that lacked it (null where no read had it).
  const effRate: Array<Array<number | null>> = [];
  const effEarners: Array<Array<number | null>> = [];
  const carry: Array<number | null> = trophies.map(() => null);
  for (let k = 0; k < history.length; k++) {
    const row: Array<number | null> = [];
    const erow: Array<number | null> = [];
    for (let i = 0; i < trophies.length; i++) {
      const r = history[k].rates !== undefined ? history[k].rates[i] : undefined;
      if (typeof r === 'number' && Number.isFinite(r)) carry[i] = r;
      row.push(carry[i]);
      erow.push(carry[i] === null ? null : (carry[i] / 100) * players[k]);
    }
    effRate.push(row);
    effEarners.push(erow);
  }
  const lastRateOf = (i: number): number => {
    for (let k = effRate.length - 1; k >= 0; k--) {
      if (effRate[k][i] !== null) return effRate[k][i] as number;
    }
    return 0.01; // never seen: the smallest printable rate
  };

  // --- the last day's reads, for the empirical hazard ---
  const windowIdx: number[] = [];
  for (let k = 0; k < history.length; k++) {
    if (timesMs[k] >= lastMs - 24 * 3600000) windowIdx.push(k);
  }
  const windowMidH = windowIdx.length > 0
    ? ((timesMs[windowIdx[0]] + timesMs[windowIdx[windowIdx.length - 1]]) / 2 - releaseMs) / 3600000
    : Math.max(1, (lastMs - releaseMs) / 3600000);
  // At least an hour: a read minutes after release (or before it, when the
  // tracked players already had the game early) must not imply the rates
  // were reached in minutes, which would blow the hazard up.
  const tLastH = Math.max(1, (lastMs - releaseMs) / 3600000);

  const rows = trophies.map((t, i) => {
    const pLast = Math.min(0.9999, Math.max(0, lastRateOf(i) / 100));
    const earners = pLast * nLast;

    // Hazard scale from the trophy itself: the rate it reached by the last
    // read, spread over the log-time since launch, slowed by its tier.
    const exposureLast = Math.log(1 + tLastH / HAZARD_C_HOURS);
    const lam0prior = tierSpeed(t.tier) * (-Math.log(1 - pLast) / Math.max(exposureLast, 0.05));
    let lam0 = lam0prior;

    if (!flat) {
      // Person-time hazard over the last day: new earners divided by the
      // non-earners who could have earned it, shrunk toward the prior so
      // a trophy with no observed earning keeps a small hazard instead of
      // none (rates are printed to 2 decimals, so small earning is hidden).
      let events = 0;
      let exposure = 0;
      for (let w = 0; w + 1 < windowIdx.length; w++) {
        const a = windowIdx[w];
        const b = windowIdx[w + 1];
        const ea = effEarners[a][i];
        const eb = effEarners[b][i];
        if (ea === null || eb === null) continue;
        const dtH = (timesMs[b] - timesMs[a]) / 3600000;
        if (dtH <= 0) continue;
        exposure += Math.max(0, ((players[a] - ea) + (players[b] - eb)) / 2) * dtH;
        if (eb > ea) events += eb - ea;
      }
      const lamPriorAtMid = lam0prior / (windowMidH + HAZARD_C_HOURS);
      const lamEmp = (events + lamPriorAtMid * PRIOR_EXPOSURE) / (exposure + PRIOR_EXPOSURE);
      lam0 = lamEmp * (windowMidH + HAZARD_C_HOURS);
    }
    lam0 = Math.min(5, Math.max(1e-6, lam0));

    // --- step the earners from the last read to the target ---
    let earnersTarget: number;
    if (pLast >= 0.99) {
      // A trophy (nearly) everyone has is earned by new players too, so
      // its rate sticks instead of diluting.
      earnersTarget = pLast * nTarget;
    } else {
      const dtH = horizonMs / 3600000 / ODE_STEPS;
      let e = earners;
      for (let s = 0; s < ODE_STEPS; s++) {
        const frac = (s + 0.5) / ODE_STEPS;
        const tH = tLastH + dtH * (s + 0.5);
        const lam = lam0 / (tH + HAZARD_C_HOURS);
        const nNow = flat ? nLast : nLast * Math.pow(nTarget / nLast, frac);
        // On a stale page the refresh's new players are just starting, so
        // they add almost no earners: only the old cohort earns. Once the
        // page is moving, the hazard already prices the mixed pool, so the
        // whole current player count is the pool.
        const pool = flat ? Math.max(0, nLast - e) : Math.max(0, nNow - e);
        e = pool - (pool - e) * Math.exp(-lam * dtH);
        if (e < 0) e = 0;
      }
      earnersTarget = e;
    }
    const median = clampPct((earnersTarget / nTarget) * 100);

    // --- spread: honest uncertainty on the logit scale ---
    // On a stale page the answer is a mix: mostly the refresh (a big drop,
    // pinned fairly tightly by the dilution) and sometimes no refresh at
    // all (the rate stays near the last read). So the lower tail is short
    // and the upper tail is long, reaching back to the last read. Once the
    // page is moving the band is symmetric and grows with the horizon.
    const sigCount = 0.7 / Math.sqrt(earners + 1);
    const sigLowBase = flat ? 0.45 : 0.55 + 0.2 * Math.sqrt(horizonDays);
    const sigHighBase = flat ? 0.9 : sigLowBase;
    const sigLow = Math.sqrt(sigLowBase * sigLowBase + sigCount * sigCount);
    const sigHigh = Math.sqrt(sigHighBase * sigHighBase + sigCount * sigCount);
    const lMed = logitPct(median);
    let q: number[];
    if (pLast >= 0.9999) {
      // The one trophy everyone has: it stays there, with a small hedge
      // in case the answer's page shows a hair under 100%.
      q = [99.9, 99.99, 99.99, 99.99, 99.99];
    } else {
      q = ZNORM.map((z) => clampPct(invLogitPct(lMed + z * (z >= 0 ? sigHigh : sigLow))));
    }
    const qr = q.map(round4);
    for (let j = 1; j < qr.length; j++) if (qr[j] < qr[j - 1]) qr[j] = qr[j - 1];
    for (let j = 0; j < qr.length; j++) qr[j] = clampPct(qr[j]);
    return { id: t.id, q: qr };
  });

  return { trophies: rows };
};

mkdirSync(OUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(IN_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(IN_DIR, file), 'utf8')) as CaseIn;
  const out = forecastCase(c);
  if (out.trophies.length !== c.trophies.length) throw new Error(`${file}: wrote ${out.trophies.length} rows for ${c.trophies.length} trophies`);
  writeFileSync(join(OUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
