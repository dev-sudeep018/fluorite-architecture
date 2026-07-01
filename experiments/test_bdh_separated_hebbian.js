// BDH-inspired separated Hebbian state
//
// Verified from arXiv:2509.26507 (Kosowski et al., Pathway, 2025):
// BDH strictly separates:
//   (E, Dx, Dy) — trained by backprop, FROZEN at inference
//   σ           — Hebbian correlation state, updated ONLY during inference,
//                 NEVER touches (E, Dx, Dy)
// Timescale explicitly stated: "minutes... up to hundreds of tokens" — working
// memory, not long-term identity. The paper is explicit that transferring this
// state to genuine long-term memory over longer timescales is UNSOLVED.
//
// Our own Wall 1 (test_hebbian_eslow.js) applied Hebbian updates directly to
// the SAME weight matrix that EqProp's gradient rule was also updating.
// Result: attractor depth deepened correctly (0.97 vs plain's 0.53) but
// spurious switch rate got WORSE (68-97% vs plain's 34-50%) — the two update
// rules fighting over the same object.
//
// This test: keep the EqProp-trained readout exactly as before (untouched),
// and add a SEPARATE Hebbian correlation matrix H that never receives EqProp's
// gradient update — only pure Hebbian correlation, decaying over time, exactly
// BDH's σ. Does architectural separation alone fix Wall 1?

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [2, 3, 4, 5, 6, 8, 10, 12];
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

// Hebbian correlation state parameters (BDH-style — separate matrix, never gradient-touched)
const HEBB_LR    = 0.015;  // correlation update rate
const HEBB_DECAY = 0.02;   // decay per episode (BDH's "U" damping term)
const HEBB_BONUS = 1.4;    // how strongly H influences the decision

function buildSchedule(rng) {
  const events = []; let t = 150, goodArm = 0;
  while (t < TOTAL_EPISODES - 60) {
    if (rng() < 0.18) {
      t += 260 + Math.floor(rng() * 80); if (t >= TOTAL_EPISODES - 60) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 40;
    } else {
      t += 22 + Math.floor(rng() * 18); if (t >= TOTAL_EPISODES - 60) break;
      const phase = t < PHASE1_END ? 1 : 2, useEval = rng() < 0.50;
      const pool = phase === 1 ? [2,3,4,5,6,7] : [7,9,11,13,15,17];
      const length = useEval ? EVAL_LENGTHS[Math.floor(rng()*EVAL_LENGTHS.length)] : pool[Math.floor(rng()*pool.length)];
      events.push({ type: 'probe', start: t, length, phase, isEval: useEval, lengthBucket: useEval ? length : null });
      t += length + 8;
    }
  }
  return events;
}

function buildTrueArmTimeline(events) {
  const base = new Array(TOTAL_EPISODES).fill(0); let current = 0;
  const gs = events.filter(e => e.type === 'genuine').sort((a,b)=>a.start-b.start); let gi=0;
  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    while (gi<gs.length && gs[gi].start===ep) { current=1-current; gi++; }
    base[ep]=current;
  }
  const tl = base.slice();
  for (const ev of events) if (ev.type==='probe')
    for (let ep=ev.start; ep<ev.start+ev.length && ep<TOTAL_EPISODES; ep++) tl[ep]=1-base[ep];
  return tl;
}

function clip(x) { return Math.max(-4, Math.min(4, x)); }

// Separated-Hebbian agent: EqProp network exactly as validated before,
// PLUS a separate H matrix (nHidden x 2) updated by pure correlation, never gradient.
function runSeparatedHebbianAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  // Separate Hebbian state — indices correspond to the 4 hidden units of net
  const H = [[0,0],[0,0],[0,0],[0,0]]; // [hiddenUnit][arm]

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  const depthLog = [];
  let depthWindow = [];
  let hBonusEstimate = [0, 0]; // running estimate of H's vote per arm, updated each episode

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];

    // Choice now genuinely incorporates H's bonus (sum over hidden units, scaled)
    const hVote = [
      H.reduce((s,row)=>s+row[0],0) / 4,
      H.reduce((s,row)=>s+row[1],0) / 4,
    ];
    const chooseFn = (values) => {
      if (rng() < 0.07) return rng() < 0.5 ? 0 : 1;
      const eff0 = values[0] + HEBB_BONUS * hVote[0];
      const eff1 = values[1] + HEBB_BONUS * hVote[1];
      return eff0 >= eff1 ? 0 : 1;
    };
    const rewardFn = (arm) => { const p = arm===goodArm?0.8:0.2; return rng()<p?1:0; };

    const { arm, reward, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm; rewardLog[ep] = reward;

    depthWindow.push(Math.abs(values[0]-values[1]));
    if (depthWindow.length>100) depthWindow.shift();
    if (ep%500===0) depthLog.push(depthWindow.reduce((a,b)=>a+b,0)/depthWindow.length);

    // Pure Hebbian correlation update on H — NEVER touches net.W
    const activationProxy = Math.abs(values[arm]);
    for (let h=0; h<4; h++) {
      H[h][arm] = clip((1-HEBB_DECAY)*H[h][arm] + HEBB_LR*activationProxy*(2*reward-1));
    }
  }

  return { preferredArm, rewardLog, depthLog, H };
}

// Baseline: plain EqProp (validated many times already)
function runPlainAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  const depthLog = []; let depthWindow=[];
  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const cf = v => { if(rng()<0.07) return rng()<0.5?0:1; return v[0]>=v[1]?0:1; };
    const rf = a => rng()<(a===goodArm?0.8:0.2)?1:0;
    const { arm, reward, values } = eqpropEpisode(net, [1], cf, rf, opts);
    preferredArm[ep]=arm; rewardLog[ep]=reward;
    depthWindow.push(Math.abs(values[0]-values[1]));
    if (depthWindow.length>100) depthWindow.shift();
    if (ep%500===0) depthLog.push(depthWindow.reduce((a,b)=>a+b,0)/depthWindow.length);
  }
  return { preferredArm, rewardLog, depthLog };
}

function detectSwitches(preferredArm, events) {
  const results = [];
  for (const ev of events) {
    if (ev.type!=='probe') continue;
    const pre = preferredArm.slice(Math.max(0,ev.start-5), ev.start);
    const preP = pre.filter(a=>a===0).length >= pre.length/2 ? 0 : 1;
    let switched=false, streak=0;
    for (let ep=ev.start; ep<Math.min(ev.start+ev.length, preferredArm.length); ep++) {
      if (preferredArm[ep]!==preP) { streak++; if(streak>=2){switched=true;break;} } else streak=0;
    }
    results.push({...ev, switched});
  }
  return results;
}

function pool(switches) {
  const out = {};
  for (const s of switches) {
    if (!s.isEval) continue;
    out[s.phase]=out[s.phase]||{};
    out[s.phase][s.lengthBucket]=out[s.phase][s.lengthBucket]||{sw:0,n:0};
    out[s.phase][s.lengthBucket].n++;
    if (s.switched) out[s.phase][s.lengthBucket].sw++;
  }
  return out;
}

const sepAgg={}, plainAgg={};
for (const ph of [1,2]) { sepAgg[ph]={}; plainAgg[ph]={}; for (const l of EVAL_LENGTHS){sepAgg[ph][l]={sw:0,n:0};plainAgg[ph][l]={sw:0,n:0};} }
let sepDepths=[], plainDepths=[], sepRewards=[], plainRewards=[];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const sep = runSeparatedHebbianAgent(tl, seed+1);
  const plain = runPlainAgent(tl, seed+2);

  sepDepths.push(sep.depthLog); plainDepths.push(plain.depthLog);
  sepRewards.push(sep.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);
  plainRewards.push(plain.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);

  const sa = pool(detectSwitches(sep.preferredArm, events));
  const pa = pool(detectSwitches(plain.preferredArm, events));
  for (const ph of [1,2]) for (const l of EVAL_LENGTHS) {
    if (sa[ph]?.[l]) { sepAgg[ph][l].n+=sa[ph][l].n; sepAgg[ph][l].sw+=sa[ph][l].sw; }
    if (pa[ph]?.[l]) { plainAgg[ph][l].n+=pa[ph][l].n; plainAgg[ph][l].sw+=pa[ph][l].sw; }
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
const f = (a,ph,l) => { const c=a[ph]?.[l]; return c?.n?(c.sw/c.n).toFixed(2)+'(n'+c.n+')':'  --  '; };
const avgD = (logs,idx) => { const v=logs.map(l=>l[Math.min(idx,l.length-1)]).filter(x=>x!=null); return v.length?(v.reduce((a,b)=>a+b,0)/v.length).toFixed(3):'--'; };

console.log(`BDH-inspired separated Hebbian state — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes\n`);

console.log('=== Overall reward ===');
console.log('plain              :', avg(plainRewards).toFixed(3));
console.log('separated_hebbian  :', avg(sepRewards).toFixed(3));

console.log('\n=== Attractor depth over time (ep500/11000/22000/44000) ===');
console.log('plain              :', avgD(plainDepths,1), avgD(plainDepths,22), avgD(plainDepths,44), avgD(plainDepths,88));
console.log('separated_hebbian  :', avgD(sepDepths,1), avgD(sepDepths,22), avgD(sepDepths,44), avgD(sepDepths,88));

console.log('\n=== Spurious switch rate ===');
console.log('len | plain-p1      plain-p2  | sep_hebb-p1   sep_hebb-p2');
for (const l of EVAL_LENGTHS)
  console.log(String(l).padStart(3),'|',f(plainAgg,1,l).padEnd(13),f(plainAgg,2,l).padEnd(10),'|',f(sepAgg,1,l).padEnd(13),f(sepAgg,2,l));

console.log('\n=== Comparison to Wall-1 same-matrix Hebbian result (from prior session) ===');
console.log('same-matrix Hebbian (Wall 1): depth 0.97-1.10 vs plain 0.53-0.72, switch rate 68-97% vs plain 34-50% (WORSE)');
console.log('separated Hebbian (this test): depth', avgD(sepDepths,88), 'vs plain', avgD(plainDepths,88));

let sepWorse=0, sepBetter=0, sepTied=0, total=0;
for (const l of EVAL_LENGTHS) for (const ph of [1,2]) {
  const s=sepAgg[ph][l], p=plainAgg[ph][l];
  if (!s?.n || !p?.n) continue;
  total++;
  const sr=s.sw/s.n, pr=p.sw/p.n;
  if (sr > pr+0.03) sepWorse++; else if (sr < pr-0.03) sepBetter++; else sepTied++;
}
console.log(`\nSeparated Hebbian vs plain: better at ${sepBetter}/${total}, worse at ${sepWorse}/${total}, tied at ${sepTied}/${total}`);
console.log(sepWorse < 3 && avgD(sepDepths,88) > avgD(plainDepths,88)
  ? '\n✓ Separation appears to fix Wall 1: deeper attractors without the switch-rate penalty.'
  : sepBetter > sepWorse
  ? '\n→ Partial improvement: separation helps but does not fully eliminate the tradeoff.'
  : '\n→ Separation alone does not fix Wall 1 — the conflict was not purely about shared weights.');
