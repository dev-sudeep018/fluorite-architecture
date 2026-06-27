// Critical period + energy-based gate
//
// The mutual dependency problem from last run:
//   gate needs E_slow structure to anchor to
//   E_slow structure needs gate to avoid probe contamination
//   → can't bootstrap either from scratch in one shot
//
// Biological resolution: critical period
//   1. Short high-plasticity burn-in: EqProp runs, we record co-activation
//   2. One-shot pre-wiring: Hebbian weights set from co-activation data
//      (this is the like-to-like wiring from MICrONS, applied empirically)
//   3. Main run: energy-based gate (not reward-based) controls further E_slow
//      Gate = |v[arm0] - v[arm1]| = how deep in an attractor we are right now
//      High certainty → gate open → Hebbian reinforces current structure
//      Low certainty → gate closed → structure preserved during transitions/probes
//
// The energy gate can't be fooled by probes because:
//   - During a probe, even if the network starts flipping, certainty drops
//   - Gate closes before Hebbian can deepen the probe-induced state
//   - EqProp alone (still running at full lr) eventually either resists or commits
//   - If it resists (short probe), certainty recovers in the original attractor
//   - If it commits (genuine change), certainty builds in the new attractor, gate reopens

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const BURN_IN = 800;          // critical period length
const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12];
const HEBBIAN_K = 50;
const HEBBIAN_LR = 0.004;
const HEBBIAN_DECAY = 0.015;
const ENERGY_GATE_SMOOTH = 0.1; // EMA smoothing for gate signal

function buildSchedule(rng) {
  const events = []; let t = 200, goodArm = 0;
  while (t < TOTAL_EPISODES - 80) {
    if (rng() < 0.15) {
      t += 300 + Math.floor(rng() * 100); if (t >= TOTAL_EPISODES - 80) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 50;
    } else {
      t += 20 + Math.floor(rng() * 15); if (t >= TOTAL_EPISODES - 80) break;
      const phase = t < PHASE1_END ? 1 : 2, useEval = rng() < 0.45;
      const pool = phase === 1 ? [2,3,4,5,6] : [6,8,10,12,14];
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

// Pre-wire Hebbian weights from co-activation data collected during burn-in
function preWireFromCoactivation(net, coactivations) {
  const { nClamp, n, W } = net;
  const K = coactivations.length;
  if (!K) return;

  // Compute mean and pairwise correlation of unit activations
  const means = new Array(n).fill(0);
  for (const act of coactivations) for (let i = 0; i < n; i++) means[i] += act[i] / K;

  const covs = Array.from({length: n}, () => new Array(n).fill(0));
  for (const act of coactivations) {
    for (let i = nClamp; i < n; i++) {
      for (let j = nClamp; j < n; j++) {
        covs[i][j] += (act[i] - means[i]) * (act[j] - means[j]) / K;
      }
    }
  }

  // Apply pre-wiring: strengthen co-activating pairs, weaken anti-correlated pairs
  const PREWIRE_STRENGTH = 0.4; // how much to shift weights based on co-activation
  for (let i = nClamp; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // Normalize by product of standard deviations for a correlation coefficient
      const si = Math.sqrt(covs[i][i] + 1e-8);
      const sj = Math.sqrt(covs[j][j] + 1e-8);
      const corr = covs[i][j] / (si * sj);
      // Add co-activation-based bias to existing weights
      W[i][j] = clip(W[i][j] + PREWIRE_STRENGTH * corr, -4, 4);
      W[j][i] = W[i][j];
    }
  }
}

function applyHebbianUpdate(net, activations, gateStrength) {
  if (!activations.length || gateStrength < 0.05) return;
  const { nClamp, n, W } = net;
  const K = activations.length;
  const mean = new Array(n).fill(0);
  for (const act of activations) for (let i = 0; i < n; i++) mean[i] += act[i] / K;
  for (let i = nClamp; i < n; i++) {
    for (let j = nClamp; j < n; j++) {
      if (j === i) continue;
      const dW = HEBBIAN_LR * gateStrength * (mean[i] * mean[j] - HEBBIAN_DECAY * W[i][j]);
      W[i][j] = clip(W[i][j] + dW, -4, 4);
      W[j][i] = W[i][j];
    }
  }
}

function getSettledState(net, clampVals) {
  // Run a brief settle and return the full activation vector
  const { nClamp, n, W, b } = net;
  let s = new Array(n).fill(0);
  for (let i = 0; i < nClamp; i++) s[i] = clampVals[i];
  for (let t = 0; t < 22; t++) {
    const next = s.slice();
    for (let i = nClamp; i < n; i++) {
      let inp = b[i];
      for (let j = 0; j < n; j++) inp += W[i][j] * s[j];
      next[i] = Math.tanh(inp);
    }
    for (let i = nClamp; i < n; i++) s[i] = next[i];
  }
  return s;
}

function runAgent(condition, trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  const depthLog = [], gateLog = [];
  let depthWindow = [];

  let activationWindow = [];
  let burnInCoactivations = [];
  let criticalPeriodDone = false;
  let smoothedGate = 0.5;

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const chooseFn = (v) => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rewardFn = (arm) => { const p = arm === goodArm ? 0.8 : 0.2; return rng() < p ? 1 : 0; };

    const { arm, reward, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm;
    rewardLog[ep] = reward;

    const depth = Math.abs(values[0] - values[1]);
    depthWindow.push(depth);
    if (depthWindow.length > 100) depthWindow.shift();
    if (ep % 500 === 0) depthLog.push(depthWindow.reduce((a,b)=>a+b,0)/depthWindow.length);

    if (condition === 'critical_period') {
      if (ep < BURN_IN) {
        // Critical period: collect co-activation data only
        const s = getSettledState(net, [1]);
        burnInCoactivations.push(s);
      } else if (!criticalPeriodDone) {
        // End of critical period: pre-wire from co-activation
        preWireFromCoactivation(net, burnInCoactivations);
        criticalPeriodDone = true;
        burnInCoactivations = []; // free memory
      } else {
        // Main run: Hebbian with energy-based gate
        // Gate = smoothed attractor depth (certainty about current state)
        // High depth = deep in attractor = gate open
        // Low depth = near bifurcation = gate closed
        smoothedGate = (1 - ENERGY_GATE_SMOOTH) * smoothedGate + ENERGY_GATE_SMOOTH * depth;

        // Normalize gate: typical depth after burn-in is ~0.6-0.8
        // Gate open when depth > 0.4, fully open at depth > 0.7
        const gateStrength = Math.max(0, Math.min(1, (smoothedGate - 0.35) / 0.45));
        if (ep % 500 === 0) gateLog.push(gateStrength);

        const s = getSettledState(net, [1]);
        activationWindow.push(s);

        if (activationWindow.length >= HEBBIAN_K) {
          const avgGate = activationWindow.length > 0 ? gateStrength : 0;
          applyHebbianUpdate(net, activationWindow, avgGate);
          activationWindow = [];
        }
      }
    }
  }

  return { preferredArm, rewardLog, depthLog, gateLog };
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

const SEEDS = [42, 1337, 9999, 5555, 2026];
const cpAgg = {}, plainAgg = {};
for (const ph of [1,2]) {
  cpAgg[ph] = {}; plainAgg[ph] = {};
  for (const len of EVAL_LENGTHS) { cpAgg[ph][len]={sw:0,n:0}; plainAgg[ph][len]={sw:0,n:0}; }
}
let cpDepths=[], plainDepths=[], cpGates=[];
let cpRewards=[], plainRewards=[];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const cp = runAgent('critical_period', tl, seed + 1);
  const plain = runAgent('plain', tl, seed + 2);

  cpDepths.push(cp.depthLog);
  plainDepths.push(plain.depthLog);
  cpGates.push(cp.gateLog);
  cpRewards.push(cp.rewardLog.reduce((a,b)=>a+b,0)/cp.rewardLog.length);
  plainRewards.push(plain.rewardLog.reduce((a,b)=>a+b,0)/plain.rewardLog.length);

  const ca = pool(detectSwitches(cp.preferredArm, events));
  const pa = pool(detectSwitches(plain.preferredArm, events));
  for (const ph of [1,2]) for (const len of EVAL_LENGTHS) {
    if (ca[ph]?.[len]) { cpAgg[ph][len].n += ca[ph][len].n; cpAgg[ph][len].sw += ca[ph][len].sw; }
    if (pa[ph]?.[len]) { plainAgg[ph][len].n += pa[ph][len].n; plainAgg[ph][len].sw += pa[ph][len].sw; }
  }
}

const f = (a, ph, len) => {
  const c = a[ph]?.[len];
  return c?.n ? (c.sw/c.n).toFixed(2)+'(n'+c.n+')' : ' -- ';
};
const avgD = (logs, idx) => {
  const vals = logs.map(l => l[Math.min(idx, l.length-1)]).filter(x => x != null);
  return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3) : '--';
};

console.log(`Pooled: ${SEEDS.length} seeds, ${BURN_IN}-episode critical period\n`);

console.log('=== Overall avg reward ===');
const avgCPR = cpRewards.reduce((a,b)=>a+b,0)/cpRewards.length;
const avgPR = plainRewards.reduce((a,b)=>a+b,0)/plainRewards.length;
console.log('plain  :', avgPR.toFixed(3));
console.log('crit_p :', avgCPR.toFixed(3));

console.log('\n=== Spurious switch rate ===');
console.log('len | plain-p1       plain-p2  | critp-p1       critp-p2');
for (const len of EVAL_LENGTHS)
  console.log(String(len).padStart(3),'|',f(plainAgg,1,len).padEnd(14),f(plainAgg,2,len).padEnd(10),'|',f(cpAgg,1,len).padEnd(14),f(cpAgg,2,len));

console.log('\n=== Attractor depth over time (ep500 / ep11000 / ep22000 / ep44000) ===');
console.log('plain  :', avgD(plainDepths,1), avgD(plainDepths,22), avgD(plainDepths,44), avgD(plainDepths,88));
console.log('crit_p :', avgD(cpDepths,1),   avgD(cpDepths,22),   avgD(cpDepths,44),   avgD(cpDepths,88));

console.log('\n=== Gate strength over time (energy gate, 0=closed 1=open) ===');
console.log('crit_p :', avgD(cpGates,1), avgD(cpGates,22-BURN_IN/500|0), avgD(cpGates,44-BURN_IN/500|0), avgD(cpGates,88-BURN_IN/500|0));
console.log('(gate opens when network is deep in an attractor, closes near bifurcation)');
