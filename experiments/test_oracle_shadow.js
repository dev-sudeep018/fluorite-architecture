// Oracle shadow tracker experiment
//
// Previous run showed: adaptive threshold barely moved (4.0 → 4.4) even though
// phase 2 probe avg duration was 11.7 vs phase 1's 4.3. Cause: self-entanglement.
// The commitment policy cuts off evidence — agent commits at episode 10, never
// records that the probe lasted 14 more episodes before reverting.
//
// This experiment decouples evidence collection from commitment:
// Oracle shadow knows the full duration of every probe from the schedule itself.
// Sets threshold = oracle_ema × MULTIPLIER at each episode.
// If oracle-adaptive beats EMA and hardcoded across schedule shapes → mechanism is sound,
// problem was purely evidence contamination.
// If oracle-adaptive still loses → the commit-threshold mechanism itself is wrong.
//
// Three conditions:
// 1. EMA (plain EqProp, no commit threshold)
// 2. Hardcoded (commit threshold = 10, fixed)
// 3. Oracle-adaptive (commit threshold from uncontaminated oracle EMA)

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12, 16, 20];
const THRESHOLD_MULTIPLIER = 1.8;
const ORACLE_ALPHA = 0.12;
const HARDCODED_THRESHOLD = 10;
const MIN_THRESHOLD = 3; // never commit in fewer than 3 episodes regardless

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

// Pre-compute oracle threshold at every episode from the actual schedule.
// At each episode, oracle knows the durations of all probes fully completed so far.
// threshold[ep] = EMA of completed probe durations up to ep × MULTIPLIER
function buildOracleThresholds(events) {
  const thresholds = new Array(TOTAL_EPISODES).fill(HARDCODED_THRESHOLD);
  let ema = 5.0; // starting prior
  
  // Sort probes by their end episode
  const probes = events
    .filter(e => e.type === 'probe')
    .map(e => ({ endEp: e.start + e.length, duration: e.length }))
    .sort((a, b) => a.endEp - b.endEp);
  
  let pi = 0;
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    // Absorb all probes that completed before or at this episode
    while (pi < probes.length && probes[pi].endEp <= ep) {
      ema = (1 - ORACLE_ALPHA) * ema + ORACLE_ALPHA * probes[pi].duration;
      pi++;
    }
    thresholds[ep] = Math.max(MIN_THRESHOLD, ema * THRESHOLD_MULTIPLIER);
  }
  return thresholds;
}

function runCommitAgent(oracleThresholds, trueArmTimeline, hardcoded, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);

  let committedArm = 0;
  let deviationStart = null;

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rf = arm => rng() < (arm === goodArm ? 0.8 : 0.2) ? 1 : 0;

    const { arm, reward } = eqpropEpisode(net, [1], cf, rf, opts);
    preferredArm[ep] = arm;
    rewardLog[ep] = reward;

    const threshold = hardcoded ? HARDCODED_THRESHOLD : oracleThresholds[ep];

    if (deviationStart === null) {
      if (arm !== committedArm) deviationStart = ep;
    } else {
      const dur = ep - deviationStart + 1;
      if (arm === committedArm) {
        deviationStart = null; // reverted, stays committed
      } else if (dur >= threshold) {
        committedArm = arm;   // genuine, commit
        deviationStart = null;
      }
    }
  }

  return { preferredArm, rewardLog };
}

function runEMAAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rf = arm => rng() < (arm === goodArm ? 0.8 : 0.2) ? 1 : 0;
    const { arm, reward } = eqpropEpisode(net, [1], cf, rf, opts);
    preferredArm[ep] = arm; rewardLog[ep] = reward;
  }
  return { preferredArm, rewardLog };
}

function detectSwitches(preferredArm, events) {
  const results = [];
  for (const ev of events) {
    if (ev.type !== 'probe') continue;
    const pre = preferredArm.slice(Math.max(0, ev.start - 5), ev.start);
    const preP = pre.filter(a => a === 0).length >= pre.length / 2 ? 0 : 1;
    let switched = false, streak = 0;
    for (let ep = ev.start; ep < Math.min(ev.start + ev.length, preferredArm.length); ep++) {
      if (preferredArm[ep] !== preP) { streak++; if (streak >= 2) { switched = true; break; } }
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

// Also measure genuine-change recovery rate — important counterpart to probe resistance
function genuineRecovery(preferredArm, events, tl) {
  let recovered = 0, total = 0;
  for (const ev of events) {
    if (ev.type !== 'genuine') continue;
    // Check if agent recovers to correct arm within 50 episodes of genuine change
    const checkEnd = Math.min(ev.start + 50, TOTAL_EPISODES);
    const correctArm = tl[checkEnd - 1]; // what the true good arm is after change
    const windowPrefs = preferredArm.slice(ev.start + 20, checkEnd); // give 20 ep grace
    const correctCount = windowPrefs.filter(a => a === correctArm).length;
    if (correctCount > windowPrefs.length * 0.7) recovered++;
    total++;
  }
  return total ? (recovered / total) : 0;
}

const SEEDS = [42, 1337, 9999, 5555, 2026];
const oracleAgg = {}, hardAgg = {}, emaAgg = {};
for (const ph of [1,2]) {
  oracleAgg[ph]={}; hardAgg[ph]={}; emaAgg[ph]={};
  for (const len of EVAL_LENGTHS) {
    oracleAgg[ph][len]={sw:0,n:0}; hardAgg[ph][len]={sw:0,n:0}; emaAgg[ph][len]={sw:0,n:0};
  }
}
let oracleRecoveries=[], hardRecoveries=[], emaRecoveries=[];
let oracleThresholdAtP1=[], oracleThresholdAtP2=[];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);
  const oracleT = buildOracleThresholds(events);

  // Record oracle threshold values at phase boundary and end
  oracleThresholdAtP1.push(oracleT[PHASE1_END]);
  oracleThresholdAtP2.push(oracleT[TOTAL_EPISODES - 1]);

  const oracle = runCommitAgent(oracleT, tl, false, seed + 1);
  const hard = runCommitAgent(null, tl, true, seed + 2);
  const ema = runEMAAgent(tl, seed + 3);

  oracleRecoveries.push(genuineRecovery(oracle.preferredArm, events, tl));
  hardRecoveries.push(genuineRecovery(hard.preferredArm, events, tl));
  emaRecoveries.push(genuineRecovery(ema.preferredArm, events, tl));

  const oa = pool(detectSwitches(oracle.preferredArm, events));
  const ha = pool(detectSwitches(hard.preferredArm, events));
  const ea = pool(detectSwitches(ema.preferredArm, events));

  for (const ph of [1,2]) for (const len of EVAL_LENGTHS) {
    if (oa[ph]?.[len]) { oracleAgg[ph][len].n+=oa[ph][len].n; oracleAgg[ph][len].sw+=oa[ph][len].sw; }
    if (ha[ph]?.[len]) { hardAgg[ph][len].n+=ha[ph][len].n; hardAgg[ph][len].sw+=ha[ph][len].sw; }
    if (ea[ph]?.[len]) { emaAgg[ph][len].n+=ea[ph][len].n; emaAgg[ph][len].sw+=ea[ph][len].sw; }
  }
}

const f = (a,ph,len) => { const c=a[ph]?.[len]; return c?.n?(c.sw/c.n).toFixed(2)+'(n'+c.n+')':'  --  '; };
const avg = arr => (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(2);

// Schedule stats
const schedRng = makeRng(SEEDS[0]);
const sampleEvents = buildSchedule(schedRng);
const p1probes = sampleEvents.filter(e=>e.type==='probe'&&e.phase===1&&!e.isEval);
const p2probes = sampleEvents.filter(e=>e.type==='probe'&&e.phase===2&&!e.isEval);
const pavg = arr => arr.length?(arr.reduce((a,b)=>a+b.length,0)/arr.length).toFixed(1):'--';

console.log(`Pooled: ${SEEDS.length} seeds\n`);
console.log('=== Schedule shape ===');
console.log(`Phase 1 probes: avg=${pavg(p1probes)} ep, n=${p1probes.length} (sample seed)`);
console.log(`Phase 2 probes: avg=${pavg(p2probes)} ep, n=${p2probes.length} (sample seed)`);

console.log('\n=== Oracle threshold recalibration ===');
console.log(`Threshold at end of phase 1: ${avg(oracleThresholdAtP1)} (target ≈ ${pavg(p1probes)}×${THRESHOLD_MULTIPLIER})`);
console.log(`Threshold at end of phase 2: ${avg(oracleThresholdAtP2)} (target ≈ ${pavg(p2probes)}×${THRESHOLD_MULTIPLIER})`);
console.log('(These should differ by ~2-3x if oracle is working)');

console.log('\n=== Genuine change recovery rate (within 50 episodes) ===');
console.log('EMA:    ', avg(emaRecoveries));
console.log('Hard:   ', avg(hardRecoveries));
console.log('Oracle: ', avg(oracleRecoveries));

console.log('\n=== Spurious switch rate ===');
console.log('len | EMA-p1        EMA-p2   | hard-p1       hard-p2  | oracle-p1     oracle-p2');
for (const len of EVAL_LENGTHS)
  console.log(String(len).padStart(3),'|',
    f(emaAgg,1,len).padEnd(13),f(emaAgg,2,len).padEnd(10),'|',
    f(hardAgg,1,len).padEnd(13),f(hardAgg,2,len).padEnd(10),'|',
    f(oracleAgg,1,len).padEnd(13),f(oracleAgg,2,len));

console.log('\nThe key test: does oracle beat hard at probe lengths > hardcoded threshold (10)?');
console.log('And does oracle-p2 show LOWER switch rates than oracle-p1 at longer lengths,');
console.log('reflecting the higher threshold it should have learned by phase 2?');
