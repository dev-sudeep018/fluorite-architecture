// E_slow + precision gate
// Previous run showed: Hebbian deepens attractors (correct) but makes system
// MORE vulnerable to probes (wrong). The reason: deeper attractors commit harder
// to probe-induced wrong states too.
//
// Missing piece: a gate on WHEN E_slow is allowed to update.
// Biological analog: dopamine codes for precision of prediction error, not error alone.
// High precision (reward matches current model) → allow structural update.
// Low precision (reward inconsistent with current model) → suppress structural update.
//
// Implementation: track EMA of reward rate when choosing current preferred arm.
// High → gate open → Hebbian deepens current attractor.
// Low (probe pressure, or early genuine change) → gate closes → attractor depth preserved.
// Eventually on genuine change: EqProp alone shifts preference (it always runs at full lr),
// new preferred arm starts rewarding consistently, gate reopens, Hebbian builds new attractor.

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12];
const HEBBIAN_K = 50;
const HEBBIAN_LR = 0.003;
const HEBBIAN_DECAY = 0.02;
const GATE_ALPHA = 0.06;    // how fast the precision gate adapts
const GATE_THRESHOLD = 0.45; // reward rate above which gate opens — calibrated to 0.8/0.2 bandit reality

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

function applyHebbianUpdate(net, activationHistory, gateStrength) {
  if (!activationHistory.length || gateStrength < 0.01) return;
  const { nClamp, n, W } = net;
  const K = activationHistory.length;
  const meanAct = new Array(n).fill(0);
  for (const act of activationHistory) for (let i = 0; i < n; i++) meanAct[i] += act[i] / K;
  for (let i = nClamp; i < n; i++) {
    for (let j = nClamp; j < n; j++) {
      if (j === i) continue;
      const coact = meanAct[i] * meanAct[j];
      const dW = HEBBIAN_LR * gateStrength * (coact - HEBBIAN_DECAY * W[i][j]);
      W[i][j] = clip(W[i][j] + dW, -4, 4);
      W[j][i] = W[i][j];
    }
  }
}

function runAgent(condition, trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  const gateLog = [];     // track gate values over time
  const depthLog = [];    // track attractor depth over time

  let activationWindow = [];
  let preferredArmRewardEMA = 0.5; // precision gate signal
  let depthWindow = [];

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];

    const chooseFn = (values) => {
      if (rng() < 0.07) return rng() < 0.5 ? 0 : 1;
      return values[0] >= values[1] ? 0 : 1;
    };
    const rewardFn = (arm) => {
      const p = arm === goodArm ? 0.8 : 0.2;
      return rng() < p ? 1 : 0;
    };

    const { arm, reward, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm;
    rewardLog[ep] = reward;

    // Attractor depth proxy
    depthWindow.push(Math.abs(values[0] - values[1]));
    if (depthWindow.length > 100) depthWindow.shift();
    if (ep % 500 === 0) depthLog.push(depthWindow.reduce((a,b)=>a+b,0)/depthWindow.length);

    if (condition === 'gated_hebbian') {
      // Update precision gate: track reward rate when choosing current preferred arm
      const currentPreferredArm = values[0] >= values[1] ? 0 : 1;
      if (arm === currentPreferredArm) {
        // Only update gate when the agent chose its preferred arm — that's when
        // the reward tells us whether the environment supports the current attractor
        preferredArmRewardEMA = (1 - GATE_ALPHA) * preferredArmRewardEMA + GATE_ALPHA * reward;
      }

      // Gate strength: 0 when reward rate = 0.5 (chance), 1 when reward rate = 1.0
      // Clipped to 0 below threshold to fully suppress Hebbian during probes
      const rawGate = (preferredArmRewardEMA - GATE_THRESHOLD) / (1 - GATE_THRESHOLD);
      const gateStrength = Math.max(0, Math.min(1, rawGate));

      if (ep % 500 === 0) gateLog.push(gateStrength);

      // Record activation for Hebbian window
      const syntheticAct = new Array(net.n).fill(0);
      syntheticAct[0] = 1;
      syntheticAct[net.n - 2] = values[0];
      syntheticAct[net.n - 1] = values[1];
      activationWindow.push({ act: syntheticAct, gate: gateStrength });

      if (activationWindow.length >= HEBBIAN_K) {
        // Use average gate strength over the window
        const avgGate = activationWindow.reduce((s, x) => s + x.gate, 0) / activationWindow.length;
        applyHebbianUpdate(net, activationWindow.map(x => x.act), avgGate);
        activationWindow = [];
      }
    }
  }

  return { preferredArm, rewardLog, gateLog, depthLog };
}

function detectSwitches(preferredArm, events) {
  const results = [];
  for (const ev of events) {
    if (ev.type !== 'probe') continue;
    const preWindow = preferredArm.slice(Math.max(0, ev.start - 5), ev.start);
    const prePreferred = preWindow.filter(a => a === 0).length >= preWindow.length / 2 ? 0 : 1;
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
const gatedAgg = {}, plainAgg = {}, hebbAgg = {};
for (const ph of [1,2]) {
  gatedAgg[ph] = {}; plainAgg[ph] = {}; hebbAgg[ph] = {};
  for (const len of EVAL_LENGTHS) {
    gatedAgg[ph][len]={sw:0,n:0}; plainAgg[ph][len]={sw:0,n:0}; hebbAgg[ph][len]={sw:0,n:0};
  }
}

let gatedDepths = [], plainDepths = [], gatedGates = [];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const gated = runAgent('gated_hebbian', tl, seed + 1);
  const plain = runAgent('plain', tl, seed + 2);

  gatedDepths.push(gated.depthLog);
  plainDepths.push(plain.depthLog);
  gatedGates.push(gated.gateLog);

  const ga = pool(detectSwitches(gated.preferredArm, events));
  const pa = pool(detectSwitches(plain.preferredArm, events));

  for (const ph of [1,2]) for (const len of EVAL_LENGTHS) {
    if (ga[ph]?.[len]) { gatedAgg[ph][len].n += ga[ph][len].n; gatedAgg[ph][len].sw += ga[ph][len].sw; }
    if (pa[ph]?.[len]) { plainAgg[ph][len].n += pa[ph][len].n; plainAgg[ph][len].sw += pa[ph][len].sw; }
  }
}

const f = (a, ph, len) => {
  const c = a[ph]?.[len];
  return c?.n ? (c.sw/c.n).toFixed(2)+'(n'+c.n+')' : ' -- ';
};
const avgD = (depths, idx) => {
  const vals = depths.map(d => d[Math.min(idx, d.length-1)]).filter(x => x != null);
  return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3) : '--';
};

console.log(`Pooled across ${SEEDS.length} seeds\n`);
console.log('=== Spurious switch rate ===');
console.log('len | plain-p1       plain-p2  | gated_hebb-p1  gated_hebb-p2');
for (const len of EVAL_LENGTHS)
  console.log(String(len).padStart(3),'|',f(plainAgg,1,len).padEnd(14),f(plainAgg,2,len).padEnd(10),'|',f(gatedAgg,1,len).padEnd(14),f(gatedAgg,2,len));

console.log('\n=== Attractor depth over time ===');
console.log('            ep500   ep11000  ep22000  ep44000');
console.log('plain  :  ', avgD(plainDepths,1), ' ', avgD(plainDepths,22), ' ', avgD(plainDepths,44), ' ', avgD(plainDepths,88));
console.log('gated  :  ', avgD(gatedDepths,1), ' ', avgD(gatedDepths,22), ' ', avgD(gatedDepths,44), ' ', avgD(gatedDepths,88));

const avgG = (gates, idx) => {
  const vals = gates.map(g => g[Math.min(idx, g.length-1)]).filter(x => x != null);
  return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3) : '--';
};
console.log('\n=== Gate strength over time (1=fully open, 0=closed) ===');
console.log('gated  :  ', avgG(gatedGates,1), ' ', avgG(gatedGates,22), ' ', avgG(gatedGates,44), ' ', avgG(gatedGates,88));
console.log('(gate should be high during stable periods and drop during probes)');

console.log('\n=== Overall avg reward ===');
// compute from raw logs — need to save them
console.log('(see switch rates above — lower spurious switching at same genuine-change recovery = better)');
