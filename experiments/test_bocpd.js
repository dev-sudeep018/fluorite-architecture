// Bayesian Online Changepoint Detection (BOCPD) for probe resistance
//
// The oracle experiment showed: scalar commit-threshold fails when probe durations
// overlap with genuine change detection times. No single cutoff works.
//
// BOCPD maintains P(genuine change has occurred since last commitment) at each episode,
// updated via Bayes rule from the reward signal. It accumulates evidence probabilistically
// rather than counting episodes. A probe generates a burst of bad rewards that pushes
// P(change) up, then reverts — P(change) drops back. A genuine change pushes
// P(change) up and keeps it there. The posterior does what the scalar threshold can't:
// integrate both duration AND reward consistency, weighted by stochastic uncertainty.
//
// Three conditions:
// 1. EMA: plain EqProp, no commit mechanism
// 2. Hardcoded: scalar threshold at 10 episodes
// 3. BOCPD: full posterior, commits when P(genuine change) > COMMIT_THRESHOLD

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12, 16, 20];
const HAZARD = 1.6e-3;          // ~72 genuine changes in 45000 episodes
const P_GOOD = 0.8;             // reward prob for correct arm
const P_BAD  = 0.2;             // reward prob for wrong arm
const COMMIT_THRESHOLD = 0.85;  // commit when P(change) exceeds this
const HARDCODED_THRESHOLD = 10;

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

// BOCPD posterior: single scalar P(change has occurred since last commitment)
// Update:
//   Time step: pChange = pChange*(1-h) + (1-pChange)*h
//   After observing reward from committed arm:
//     pChange = posterior via Bayes rule with P_GOOD / P_BAD likelihood model
function bocpdUpdate(pChange, choseCommittedArm, reward) {
  // Time evolution (genuine change can happen at any episode)
  pChange = pChange * (1 - HAZARD) + (1 - pChange) * HAZARD;

  // Only update on informative observations: chose the committed arm
  if (!choseCommittedArm) return pChange;

  // Likelihood of this reward under each hypothesis
  const pObs_noChange = reward ? P_GOOD : (1 - P_GOOD); // committed arm still good
  const pObs_change   = reward ? P_BAD  : (1 - P_BAD);  // committed arm now bad

  // Bayesian update
  const unnorm_noChange = (1 - pChange) * pObs_noChange;
  const unnorm_change   = pChange * pObs_change;
  const norm = unnorm_noChange + unnorm_change;
  if (norm < 1e-10) return pChange; // numerical guard
  return unnorm_change / norm;
}

function runBOCPDAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog    = new Array(TOTAL_EPISODES).fill(0);
  const pChangeLog   = []; // for diagnostics

  let committedArm = 0;
  let pChange = 0.0;

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rf = arm => rng() < (arm === goodArm ? P_GOOD : P_BAD) ? 1 : 0;

    const { arm, reward } = eqpropEpisode(net, [1], cf, rf, opts);
    preferredArm[ep] = arm;
    rewardLog[ep] = reward;

    // Update BOCPD posterior
    pChange = bocpdUpdate(pChange, arm === committedArm, reward);
    if (ep % 500 === 0) pChangeLog.push(pChange);

    // Commit to new arm if posterior crosses threshold
    if (pChange > COMMIT_THRESHOLD) {
      // The EqProp network currently prefers whichever arm it's been rewarded for
      // Commit to current network preference as the new "stable" arm
      committedArm = arm;
      pChange = 0.0; // reset posterior — we've acknowledged the change
    }
  }

  return { preferredArm, rewardLog, pChangeLog };
}

function runHardcodedAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog    = new Array(TOTAL_EPISODES).fill(0);
  let committedArm = 0, deviationStart = null;
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rf = arm => rng() < (arm === goodArm ? P_GOOD : P_BAD) ? 1 : 0;
    const { arm, reward } = eqpropEpisode(net, [1], cf, rf, opts);
    preferredArm[ep] = arm; rewardLog[ep] = reward;
    if (deviationStart === null) { if (arm !== committedArm) deviationStart = ep; }
    else {
      if (arm === committedArm) { deviationStart = null; }
      else if (ep - deviationStart + 1 >= HARDCODED_THRESHOLD) { committedArm = arm; deviationStart = null; }
    }
  }
  return { preferredArm, rewardLog };
}

function runEMAAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog    = new Array(TOTAL_EPISODES).fill(0);
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rf = arm => rng() < (arm === goodArm ? P_GOOD : P_BAD) ? 1 : 0;
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

function genuineRecovery(preferredArm, events, tl) {
  let rec = 0, tot = 0;
  for (const ev of events) {
    if (ev.type !== 'genuine') continue;
    const correctArm = tl[Math.min(ev.start + 49, TOTAL_EPISODES - 1)];
    const win = preferredArm.slice(ev.start + 20, Math.min(ev.start + 50, TOTAL_EPISODES));
    if (win.filter(a => a === correctArm).length > win.length * 0.7) rec++;
    tot++;
  }
  return tot ? rec / tot : 0;
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

// Run a quick diagnostic to understand what BOCPD actually does during a probe
function bocpdDiagnostic() {
  let pC = 0;
  console.log('=== BOCPD posterior during a 10-episode probe then revert ===');
  console.log('ep | P(change) | phase');
  for (let ep = 0; ep < 25; ep++) {
    const inProbe = ep >= 3 && ep < 13;
    // Simulate: in probe, committed arm gets reward=0 most of the time
    // Outside probe, committed arm gets reward=1 most of the time
    const reward = inProbe ? (Math.random() < P_BAD ? 1 : 0) : (Math.random() < P_GOOD ? 1 : 0);
    pC = bocpdUpdate(pC, true, reward); // always chose committed arm for clean signal
    const phase = ep < 3 ? 'stable' : ep < 13 ? 'PROBE' : 'reverted';
    console.log(String(ep).padStart(3), '|', pC.toFixed(4), '|', phase);
  }
  console.log('(BOCPD should climb during probe, stay below', COMMIT_THRESHOLD, ', drop after revert)');
}

bocpdDiagnostic();

// Main experiment
const SEEDS = [42, 1337, 9999, 5555, 2026];
const bocpdAgg = {}, hardAgg = {}, emaAgg = {};
for (const ph of [1,2]) {
  bocpdAgg[ph]={}; hardAgg[ph]={}; emaAgg[ph]={};
  for (const len of EVAL_LENGTHS) {
    bocpdAgg[ph][len]={sw:0,n:0}; hardAgg[ph][len]={sw:0,n:0}; emaAgg[ph][len]={sw:0,n:0};
  }
}
let bocpdRec=[], hardRec=[], emaRec=[];
let avgRewards = { bocpd:[], hard:[], ema:[] };

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const bocpd = runBOCPDAgent(tl, seed + 1);
  const hard  = runHardcodedAgent(tl, seed + 2);
  const ema   = runEMAAgent(tl, seed + 3);

  bocpdRec.push(genuineRecovery(bocpd.preferredArm, events, tl));
  hardRec.push(genuineRecovery(hard.preferredArm, events, tl));
  emaRec.push(genuineRecovery(ema.preferredArm, events, tl));

  avgRewards.bocpd.push(bocpd.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);
  avgRewards.hard.push(hard.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);
  avgRewards.ema.push(ema.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);

  const ba = pool(detectSwitches(bocpd.preferredArm, events));
  const ha = pool(detectSwitches(hard.preferredArm, events));
  const ea = pool(detectSwitches(ema.preferredArm, events));

  for (const ph of [1,2]) for (const len of EVAL_LENGTHS) {
    if (ba[ph]?.[len]) { bocpdAgg[ph][len].n+=ba[ph][len].n; bocpdAgg[ph][len].sw+=ba[ph][len].sw; }
    if (ha[ph]?.[len]) { hardAgg[ph][len].n+=ha[ph][len].n; hardAgg[ph][len].sw+=ha[ph][len].sw; }
    if (ea[ph]?.[len]) { emaAgg[ph][len].n+=ea[ph][len].n; emaAgg[ph][len].sw+=ea[ph][len].sw; }
  }
}

const f = (a,ph,len) => { const c=a[ph]?.[len]; return c?.n?(c.sw/c.n).toFixed(2)+'(n'+c.n+')':'  --  '; };
const avg = arr => (arr.reduce((a,b)=>a+b,0)/arr.length);

console.log(`\nPooled: ${SEEDS.length} seeds x ${TOTAL_EPISODES} episodes`);
console.log(`BOCPD hazard=${HAZARD}, commit threshold=${COMMIT_THRESHOLD}\n`);

console.log('=== Overall avg reward ===');
console.log('EMA:  ', avg(avgRewards.ema).toFixed(3));
console.log('Hard: ', avg(avgRewards.hard).toFixed(3));
console.log('BOCPD:', avg(avgRewards.bocpd).toFixed(3));

console.log('\n=== Genuine change recovery rate (within 50 episodes) ===');
console.log('EMA:  ', avg(emaRec).toFixed(2));
console.log('Hard: ', avg(hardRec).toFixed(2));
console.log('BOCPD:', avg(bocpdRec).toFixed(2));

console.log('\n=== Spurious switch rate ===');
console.log('len | EMA-p1        EMA-p2  | hard-p1       hard-p2  | bocpd-p1      bocpd-p2');
for (const len of EVAL_LENGTHS)
  console.log(String(len).padStart(3),'|',
    f(emaAgg,1,len).padEnd(13), f(emaAgg,2,len).padEnd(9),'|',
    f(hardAgg,1,len).padEnd(13), f(hardAgg,2,len).padEnd(9),'|',
    f(bocpdAgg,1,len).padEnd(13), f(bocpdAgg,2,len));

console.log('\nIf BOCPD beats EMA on spurious switches AND matches EMA on recovery:');
console.log('probabilistic inference over duration works where scalar threshold does not.');
console.log('If BOCPD still loses to EMA: the whole commit-mechanism family is wrong here.');
