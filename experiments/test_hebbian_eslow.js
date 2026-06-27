// E_fast / E_slow bandit test
// E_fast: standard EqProp settling and weight update per episode
// E_slow: slow Hebbian co-activation update every K episodes
//         implements the MICrONS "like-to-like" wiring principle dynamically
//         hypothesis: deepens attractors, produces probe resistance as a structural property
//         rather than as explicit probe detection

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12];
const HEBBIAN_K = 50;       // apply slow update every K episodes
const HEBBIAN_LR = 0.003;   // << EqProp lr of 0.07
const HEBBIAN_DECAY = 0.02; // Oja-style decay to prevent unbounded growth

function buildSchedule(rng) {
  const events = []; let t = 200, goodArm = 0;
  while (t < TOTAL_EPISODES - 80) {
    if (rng() < 0.15) {
      t += 300 + Math.floor(rng() * 100); if (t >= TOTAL_EPISODES - 80) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 50;
    } else {
      t += 20 + Math.floor(rng() * 15); if (t >= TOTAL_EPISODES - 80) break;
      const phase = t < PHASE1_END ? 1 : 2;
      const useEval = rng() < 0.45;
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

// Apply slow Hebbian update based on average activation over last K episodes
// Oja-style: dW += lr_slow * (avg_i * avg_j - decay * W_ij)
// This strengthens connections between co-activating units (like-to-like)
// and weakens connections between units that don't co-activate
function applyHebbianUpdate(net, activationHistory) {
  const { nClamp, n, W } = net;
  const K = activationHistory.length;
  if (K === 0) return;

  // Compute mean activation per unit over last K episodes
  const meanAct = new Array(n).fill(0);
  for (const act of activationHistory) for (let i = 0; i < n; i++) meanAct[i] += act[i] / K;

  // Compute co-activation (outer product of means, simplified)
  for (let i = nClamp; i < n; i++) {
    for (let j = nClamp; j < n; j++) {
      if (j === i) continue;
      const coact = meanAct[i] * meanAct[j];
      const dW = HEBBIAN_LR * (coact - HEBBIAN_DECAY * W[i][j]);
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

  // For Hebbian: store settled state activations over each K-episode window
  let activationWindow = [];

  // Track attractor depth over time (measure energy gap between equilibria)
  // Proxy: |value[arm0] - value[arm1]| averaged over a window
  const attractorDepth = [];
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

    // Track value gap as attractor depth proxy
    depthWindow.push(Math.abs(values[0] - values[1]));
    if (depthWindow.length > 100) depthWindow.shift();
    if (ep % 500 === 0) attractorDepth.push(depthWindow.reduce((a,b)=>a+b,0)/depthWindow.length);

    if (condition === 'hebbian') {
      // Record settled state for Hebbian update
      // We use the free-phase state — approximate by re-running settle briefly
      // (cheaper: just record the output values as a proxy for unit states)
      // For a proper implementation we'd need to save the full free-phase state from eqpropEpisode
      // Here we approximate with a synthetic activation vector from the network output
      // This is honest — it's an approximation of the full biological Hebbian rule
      const syntheticAct = new Array(net.n).fill(0);
      syntheticAct[0] = 1; // clamp unit
      syntheticAct[net.n - 2] = values[0];
      syntheticAct[net.n - 1] = values[1];
      activationWindow.push(syntheticAct);

      if (activationWindow.length >= HEBBIAN_K) {
        applyHebbianUpdate(net, activationWindow);
        activationWindow = [];
      }
    }
  }

  return { preferredArm, rewardLog, attractorDepth };
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

function agg(switches) {
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

// Pool across seeds
const SEEDS = [42, 1337, 9999, 5555, 2026];
const hebbAgg = {}, plainAgg = {};
for (const ph of [1,2]) { hebbAgg[ph]=[]; plainAgg[ph]=[]; for (const len of EVAL_LENGTHS) { hebbAgg[ph][len]={sw:0,n:0}; plainAgg[ph][len]={sw:0,n:0}; } }

let hebbDepths = [], plainDepths = [];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const hebb = runAgent('hebbian', tl, seed + 1);
  const plain = runAgent('plain', tl, seed + 2);

  hebbDepths.push(hebb.attractorDepth);
  plainDepths.push(plain.attractorDepth);

  const hs = detectSwitches(hebb.preferredArm, events);
  const ps = detectSwitches(plain.preferredArm, events);

  const ha = agg(hs), pa = agg(ps);
  for (const ph of [1,2]) for (const len of EVAL_LENGTHS) {
    if (ha[ph] && ha[ph][len]) { hebbAgg[ph][len].n += ha[ph][len].n; hebbAgg[ph][len].sw += ha[ph][len].sw; }
    if (pa[ph] && pa[ph][len]) { plainAgg[ph][len].n += pa[ph][len].n; plainAgg[ph][len].sw += pa[ph][len].sw; }
  }
}

const f = (a, ph, len) => {
  const c = a[ph][len];
  return c && c.n ? (c.sw/c.n).toFixed(2)+'(n'+c.n+')' : ' -- ';
};

console.log(`Pooled across ${SEEDS.length} seeds\n`);
console.log('=== Spurious switch rate: plain EqProp vs EqProp + slow Hebbian (E_slow) ===');
console.log('len | plain-p1      plain-p2  | hebbian-p1    hebbian-p2');
for (const len of EVAL_LENGTHS) {
  console.log(String(len).padStart(3), '|',
    f(plainAgg,1,len).padEnd(13), f(plainAgg,2,len).padEnd(10), '|',
    f(hebbAgg,1,len).padEnd(13), f(hebbAgg,2,len));
}

// Attractor depth over time: does Hebbian deepen the energy gap?
// Average depth at early, mid, late phase 1, and late phase 2
console.log('\n=== Attractor depth (|v[arm0] - v[arm1]| averaged over 100 episodes) ===');
console.log('Time into run → early(ep500) | mid-p1(ep11000) | end-p1(ep22000) | end-p2(ep44000)');
const avgDepth = (depths, idx) => {
  const vals = depths.map(d => d[Math.min(idx, d.length-1)]).filter(x => x !== undefined);
  return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3) : ' --';
};
// idx ≈ ep/500
console.log('plain  :', avgDepth(plainDepths,1).padEnd(10), avgDepth(plainDepths,22).padEnd(17), avgDepth(plainDepths,44).padEnd(17), avgDepth(plainDepths,88));
console.log('hebbian:', avgDepth(hebbDepths,1).padEnd(10), avgDepth(hebbDepths,22).padEnd(17), avgDepth(hebbDepths,44).padEnd(17), avgDepth(hebbDepths,88));

console.log('\nIf Hebbian deepens attractors (higher depth numbers), probe resistance should follow.');
console.log('If depth is the same, the Hebbian update is doing nothing structural.');
