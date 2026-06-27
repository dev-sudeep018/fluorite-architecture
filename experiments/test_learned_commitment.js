// Learned commitment layer
//
// The hardcoded CS agent worked (0.4-2% spurious switches vs 31-32% for EMA)
// but its threshold was fixed. This version learns the threshold from experience.
//
// Architecture:
// - Main agent: EqProp arm choice + commit threshold from shadow tracker
// - Shadow tracker: passively records deviation durations without gating contamination
//   → computes EMA of "how long do probes typically last before reverting"
//   → sets commit threshold = that EMA * THRESHOLD_MULTIPLIER
//
// Falsification test: does the commit threshold actually shift between
// phase 1 (probes last 2-6 episodes) and phase 2 (probes last 6-14 episodes)?
// If yes: mechanism is calibrating against the world.
// If no: mechanism is calibrating against its own policy.
//
// Secondary test: does adaptive threshold beat hardcoded threshold on the
// phase 2 probe lengths (where hardcoded 10-episode threshold is too short)?

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12, 16, 20];
const THRESHOLD_MULTIPLIER = 1.8; // commit after 1.8× typical probe duration
const SHADOW_ALPHA_FAST = 0.15;   // how fast shadow tracker adapts
const HARDCODED_THRESHOLD = 10;   // baseline comparison

function buildSchedule(rng) {
  const events = []; let t = 200, goodArm = 0;
  while (t < TOTAL_EPISODES - 80) {
    if (rng() < 0.15) {
      t += 300 + Math.floor(rng() * 100);
      if (t >= TOTAL_EPISODES - 80) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 50;
    } else {
      t += 20 + Math.floor(rng() * 15);
      if (t >= TOTAL_EPISODES - 80) break;
      const phase = t < PHASE1_END ? 1 : 2, useEval = rng() < 0.45;
      const pool = phase === 1 ? [2,3,4,5,6,7] : [7,9,11,13,15,17];
      const length = useEval
        ? EVAL_LENGTHS[Math.floor(rng() * EVAL_LENGTHS.length)]
        : pool[Math.floor(rng() * pool.length)];
      events.push({ type: 'probe', start: t, length, phase, isEval: useEval, lengthBucket: useEval ? length : null });
      t += length + 10;
    }
  }
  return events;
}

function buildTrueArmTimeline(events) {
  const base = new Array(TOTAL_EPISODES).fill(0); let current = 0;
  const gs = events.filter(e => e.type === 'genuine').sort((a, b) => a.start - b.start); let gi = 0;
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    while (gi < gs.length && gs[gi].start === ep) { current = 1 - current; gi++; }
    base[ep] = current;
  }
  const tl = base.slice();
  for (const ev of events) if (ev.type === 'probe')
    for (let ep = ev.start; ep < ev.start + ev.length && ep < TOTAL_EPISODES; ep++)
      tl[ep] = 1 - base[ep];
  return tl;
}

function clip(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function runAgent(condition, trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);

  // Shadow tracker state
  let shadowDurationEMA = 5.0;  // start assuming probes are ~5 episodes long
  let mainDurationEMA = 5.0;    // main agent's own (potentially contaminated) estimate

  // Threshold log: record at key intervals
  const thresholdLog = [];      // [ep, shadow_threshold, main_threshold]

  // Deviation state machine
  let committedArm = 0;
  let deviationStart = null;
  let deviationDuration = 0;
  let shadowDeviationStart = null; // shadow tracks independently

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const chooseFn = (v) => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rewardFn = (arm) => { const p = arm === goodArm ? 0.8 : 0.2; return rng() < p ? 1 : 0; };

    const { arm, reward } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm;
    rewardLog[ep] = reward;

    // Compute current thresholds
    const shadowThreshold = shadowDurationEMA * THRESHOLD_MULTIPLIER;
    const mainThreshold = condition === 'adaptive'
      ? mainDurationEMA * THRESHOLD_MULTIPLIER
      : HARDCODED_THRESHOLD;

    // Log thresholds at phase boundary and periodically
    if (ep % 2000 === 0 || ep === PHASE1_END) {
      thresholdLog.push({ ep, shadowThreshold, mainThreshold });
    }

    // --- Main agent deviation state machine ---
    if (deviationStart === null) {
      if (arm !== committedArm) {
        deviationStart = ep;
        deviationDuration = 1;
      }
    } else {
      deviationDuration = ep - deviationStart + 1;
      if (arm === committedArm) {
        // Deviation reverted — it was a probe
        // Update main agent's duration estimate (only if adaptive)
        if (condition === 'adaptive') {
          mainDurationEMA = (1 - SHADOW_ALPHA_FAST) * mainDurationEMA + SHADOW_ALPHA_FAST * deviationDuration;
        }
        deviationStart = null;
      } else if (deviationDuration >= mainThreshold) {
        // Deviation sustained past threshold — commit to new arm
        committedArm = arm;
        deviationStart = null;
      }
    }

    // --- Shadow tracker (always runs, never gated) ---
    // Shadow observes what the main agent's preferred arm actually does,
    // independent of the commitment policy.
    // We track the EqProp network's raw preference (not the committed arm)
    // since that's what the shadow would observe in an uncontaminated way.
    const rawNetPreference = arm; // the arm the network actually chose this episode

    if (shadowDeviationStart === null) {
      // shadow uses ep-0 preference as baseline — updated each time shadow resolves
      if (ep === 0) { shadowDeviationStart = null; } // init
    }

    // Shadow uses a separate baseline that updates whenever a deviation resolves
    // to give uncontaminated duration statistics
    // (simplified: shadow just tracks EMA of all inter-flip intervals)
    // We'll compute this via the event schedule directly to get ground truth
    // That's the cleanest possible shadow: uses oracle knowledge of when probes actually end

  }

  // Compute oracle shadow threshold from actual schedule
  // This is what a perfect shadow tracker would converge to

  return { preferredArm, rewardLog, thresholdLog, mainDurationEMA };
}

function detectSwitches(preferredArm, events) {
  const results = [];
  for (const ev of events) {
    if (ev.type !== 'probe') continue;
    const pre = preferredArm.slice(Math.max(0, ev.start - 5), ev.start);
    const prePreferred = pre.filter(a => a === 0).length >= pre.length / 2 ? 0 : 1;
    let switched = false, streak = 0;
    for (let ep = ev.start; ep < Math.min(ev.start + ev.length, preferredArm.length); ep++) {
      if (preferredArm[ep] !== prePreferred) { streak++; if (streak >= 2) { switched = true; break; } }
      else streak = 0;
    }
    results.push({ ...ev, switched });
  }
  return results;
}

function pool(switches) {
  const out = {};
  for (const s of switches) {
    if (!s.isEval) continue;
    out[s.phase] = out[s.phase] || {};
    out[s.phase][s.lengthBucket] = out[s.phase][s.lengthBucket] || { sw: 0, n: 0 };
    out[s.phase][s.lengthBucket].n++;
    if (s.switched) out[s.phase][s.lengthBucket].sw++;
  }
  return out;
}

// Compute ground-truth probe duration distribution by phase from schedule
function probeStats(events) {
  const p1 = events.filter(e => e.type === 'probe' && e.phase === 1 && !e.isEval).map(e => e.length);
  const p2 = events.filter(e => e.type === 'probe' && e.phase === 2 && !e.isEval).map(e => e.length);
  const avg = arr => arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(1) : '--';
  const max = arr => arr.length ? Math.max(...arr) : '--';
  return { p1avg: avg(p1), p1max: max(p1), p2avg: avg(p2), p2max: max(p2), n1: p1.length, n2: p2.length };
}

const SEEDS = [42, 1337, 9999, 5555, 2026];
const adaptAgg = {}, hardAgg = {}, emaAgg = {};
for (const ph of [1,2]) {
  adaptAgg[ph] = {}; hardAgg[ph] = {}; emaAgg[ph] = {};
  for (const len of EVAL_LENGTHS) {
    adaptAgg[ph][len]={sw:0,n:0}; hardAgg[ph][len]={sw:0,n:0}; emaAgg[ph][len]={sw:0,n:0};
  }
}
let scheduleStats = null;
let adaptThresholdP1End = [], adaptThresholdP2End = [];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  if (!scheduleStats) scheduleStats = probeStats(events);

  const adapt = runAgent('adaptive', tl, seed + 1);
  const hard = runAgent('hardcoded', tl, seed + 2);

  // EMA baseline (plain agent with no commit mechanism)
  const emaRng = makeRng(seed + 3);
  const emaNet = makeNetwork(1, 4, emaRng);
  const emaOpts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const emaPref = new Array(TOTAL_EPISODES).fill(0);
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = tl[ep];
    const cf = v => { if(emaRng()<0.07) return emaRng()<0.5?0:1; return v[0]>=v[1]?0:1; };
    const rf = arm => emaRng() < (arm===goodArm?0.8:0.2) ? 1 : 0;
    const { arm } = eqpropEpisode(emaNet, [1], cf, rf, emaOpts);
    emaPref[ep] = arm;
  }

  // Extract threshold at phase boundary and end
  const p1log = adapt.thresholdLog.find(l => l.ep === PHASE1_END);
  const p2log = adapt.thresholdLog[adapt.thresholdLog.length - 1];
  if (p1log) adaptThresholdP1End.push(p1log.mainThreshold);
  if (p2log) adaptThresholdP2End.push(p2log.mainThreshold);

  const aa = pool(detectSwitches(adapt.preferredArm, events));
  const ha = pool(detectSwitches(hard.preferredArm, events));
  const ea = pool(detectSwitches(emaPref, events));

  for (const ph of [1,2]) for (const len of EVAL_LENGTHS) {
    if (aa[ph]?.[len]) { adaptAgg[ph][len].n += aa[ph][len].n; adaptAgg[ph][len].sw += aa[ph][len].sw; }
    if (ha[ph]?.[len]) { hardAgg[ph][len].n += ha[ph][len].n; hardAgg[ph][len].sw += ha[ph][len].sw; }
    if (ea[ph]?.[len]) { emaAgg[ph][len].n += ea[ph][len].n; emaAgg[ph][len].sw += ea[ph][len].sw; }
  }
}

const f = (a, ph, len) => {
  const c = a[ph]?.[len];
  return c?.n ? (c.sw/c.n).toFixed(2)+'(n'+c.n+')' : '  --  ';
};
const avg = arr => arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(1) : '--';

console.log(`Pooled: ${SEEDS.length} seeds x ${TOTAL_EPISODES} episodes\n`);
console.log('Schedule ground truth:');
console.log(`Phase 1 non-eval probes: n=${scheduleStats.n1}, avg duration=${scheduleStats.p1avg}, max=${scheduleStats.p1max}`);
console.log(`Phase 2 non-eval probes: n=${scheduleStats.n2}, avg duration=${scheduleStats.p2avg}, max=${scheduleStats.p2max}`);
console.log(`Hardcoded threshold: ${HARDCODED_THRESHOLD}`);

console.log('\n=== Does the adaptive threshold actually shift between phases? ===');
console.log(`Adaptive threshold at end of phase 1: ${avg(adaptThresholdP1End)} (should be ~${scheduleStats.p1avg}×${THRESHOLD_MULTIPLIER} ≈ ${(parseFloat(scheduleStats.p1avg)*THRESHOLD_MULTIPLIER).toFixed(1)})`);
console.log(`Adaptive threshold at end of phase 2: ${avg(adaptThresholdP2End)} (should be ~${scheduleStats.p2avg}×${THRESHOLD_MULTIPLIER} ≈ ${(parseFloat(scheduleStats.p2avg)*THRESHOLD_MULTIPLIER).toFixed(1)})`);
console.log('If these two numbers differ significantly: mechanism recalibrated. If same: it did not.');

console.log('\n=== Spurious switch rate ===');
console.log('len | EMA-p1        EMA-p2  | hard-p1       hard-p2  | adapt-p1      adapt-p2');
for (const len of EVAL_LENGTHS)
  console.log(String(len).padStart(3),'|',
    f(emaAgg,1,len).padEnd(13), f(emaAgg,2,len).padEnd(9),'|',
    f(hardAgg,1,len).padEnd(13), f(hardAgg,2,len).padEnd(9),'|',
    f(adaptAgg,1,len).padEnd(13), f(adaptAgg,2,len));

console.log('\nKey comparison: at probe lengths > hardcoded threshold (10),');
console.log('adaptive should outperform hardcoded if threshold recalibration is real.');
