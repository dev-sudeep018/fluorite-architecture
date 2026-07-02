// Oja-stabilized pure co-activation Hebbian state
//
// Verified: pure Hebbian learning is unstable because postsynaptic output y
// is itself w*x, so dw ~ x*y ~ w*x^2 -> exponential runaway. This is true
// with or without a reward term - "wiring causes firing causes more wiring."
// Oja (1982) fixes this with a stabilizing term: dw = lr*(x*y - y^2*w).
//
// Four conditions, same schedule, same seeds:
//   1. plain            - baseline, no Hebbian state at all
//   2. reward_coupled    - H updated by activation*(2*reward-1), no normalization
//                          (this is what we tested last turn - failed, 15/16 worse)
//   3. pure_unnormalized - H updated by pure co-activation (arm chosen * activation
//                          strength), NO reward term, NO Oja normalization
//                          PREDICTION per verified theory: also unstable/runaway
//   4. pure_oja          - same as (3) but with Oja's stabilizing term added
//                          PREDICTION: bounded, may actually help

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [2, 3, 4, 5, 6, 8, 10, 12];
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

const HEBB_LR    = 0.02;
const HEBB_BONUS = 1.2;
const OJA_LR      = 0.02; // separate rate for the stabilizing term

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

function runAgent(condition, trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const H = [0, 0]; // scalar Hebbian channel per arm (honest simplification, as before)

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  const depthLog = [];
  const hMagLog = []; // track |H| over time to directly see runaway vs bounded
  let depthWindow = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];

    const chooseFn = (values) => {
      if (rng() < 0.07) return rng() < 0.5 ? 0 : 1;
      if (condition === 'plain') return values[0] >= values[1] ? 0 : 1;
      const eff0 = values[0] + HEBB_BONUS * H[0];
      const eff1 = values[1] + HEBB_BONUS * H[1];
      return eff0 >= eff1 ? 0 : 1;
    };
    const rewardFn = (arm) => { const p = arm===goodArm?0.8:0.2; return rng()<p?1:0; };

    const { arm, reward, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm; rewardLog[ep] = reward;

    depthWindow.push(Math.abs(values[0]-values[1]));
    if (depthWindow.length>100) depthWindow.shift();
    if (ep%500===0) { depthLog.push(depthWindow.reduce((a,b)=>a+b,0)/depthWindow.length); hMagLog.push(Math.max(Math.abs(H[0]),Math.abs(H[1]))); }

    const activationProxy = Math.abs(values[arm]); // "how strongly did the network favor this choice" - no reward here

    if (condition === 'reward_coupled') {
      // H updated by activation correlated with REWARD (the version tested last turn)
      H[arm] = clip(H[arm]*0.98 + HEBB_LR * activationProxy * (2*reward-1));
    } else if (condition === 'pure_unnormalized') {
      // Pure co-activation - x*y with NO reward, NO stabilization
      // x = activationProxy (did the network fire strongly for this choice)
      // y = H[arm] itself (postsynaptic proxy)
      const x = activationProxy, y = H[arm];
      H[arm] = clip(H[arm] + HEBB_LR * x * (y===0?0.1:y)); // seed with small value to escape y=0 fixed point
    } else if (condition === 'pure_oja') {
      // Oja-stabilized: dw = lr*(x*y) - oja_lr*(y^2 * w)
      const x = activationProxy, y = H[arm]===0?0.1:H[arm];
      const potentiation = HEBB_LR * x * y;
      const stabilization = OJA_LR * y * y * H[arm];
      H[arm] = clip(H[arm] + potentiation - stabilization);
    }
    // 'plain' does nothing to H
  }

  return { preferredArm, rewardLog, depthLog, hMagLog };
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

const conditions = ['plain', 'reward_coupled', 'pure_unnormalized', 'pure_oja'];
const agg = {}; const rewards = {}; const depths = {}; const hMags = {};
for (const c of conditions) {
  agg[c] = {1:{},2:{}};
  for (const l of EVAL_LENGTHS) { agg[c][1][l]={sw:0,n:0}; agg[c][2][l]={sw:0,n:0}; }
  rewards[c]=[]; depths[c]=[]; hMags[c]=[];
}

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  for (let ci=0; ci<conditions.length; ci++) {
    const c = conditions[ci];
    const r = runAgent(c, tl, seed + ci*100 + 1);
    rewards[c].push(r.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);
    depths[c].push(r.depthLog);
    hMags[c].push(r.hMagLog);
    const sw = pool(detectSwitches(r.preferredArm, events));
    for (const ph of [1,2]) for (const l of EVAL_LENGTHS) {
      if (sw[ph]?.[l]) { agg[c][ph][l].n+=sw[ph][l].n; agg[c][ph][l].sw+=sw[ph][l].sw; }
    }
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
const f = (c,ph,l) => { const x=agg[c][ph][l]; return x?.n?(x.sw/x.n).toFixed(2)+'(n'+x.n+')':'  --  '; };
const avgD = (logs,idx) => { const v=logs.map(l=>l[Math.min(idx,l.length-1)]).filter(x=>x!=null); return v.length?(v.reduce((a,b)=>a+b,0)/v.length).toFixed(3):'--'; };

console.log(`Oja-stabilized Hebbian test — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes\n`);

console.log('=== Overall reward ===');
for (const c of conditions) console.log(c.padEnd(20), ':', avg(rewards[c]).toFixed(3));

console.log('\n=== |H| magnitude over time (ep500/22000/44000) — is it bounded or runaway? ===');
for (const c of conditions.slice(1)) {
  console.log(c.padEnd(20), ':', avgD(hMags[c],1), avgD(hMags[c],44), avgD(hMags[c],88));
}

console.log('\n=== Attractor depth over time ===');
for (const c of conditions) console.log(c.padEnd(20), ':', avgD(depths[c],1), avgD(depths[c],44), avgD(depths[c],88));

console.log('\n=== Spurious switch rate: plain vs pure_oja ===');
console.log('len | plain-p1      plain-p2  | pure_oja-p1   pure_oja-p2');
for (const l of EVAL_LENGTHS)
  console.log(String(l).padStart(3),'|',f('plain',1,l).padEnd(13),f('plain',2,l).padEnd(10),'|',f('pure_oja',1,l).padEnd(13),f('pure_oja',2,l));

console.log('\n=== Spurious switch rate: reward_coupled vs pure_unnormalized (both predicted unstable) ===');
console.log('len | reward_c-p1   reward_c-p2 | pure_unn-p1   pure_unn-p2');
for (const l of EVAL_LENGTHS)
  console.log(String(l).padStart(3),'|',f('reward_coupled',1,l).padEnd(13),f('reward_coupled',2,l).padEnd(10),'|',f('pure_unnormalized',1,l).padEnd(13),f('pure_unnormalized',2,l));

let ojaBetter=0, ojaWorse=0, ojaTied=0, total=0;
for (const l of EVAL_LENGTHS) for (const ph of [1,2]) {
  const p=agg['plain'][ph][l], o=agg['pure_oja'][ph][l];
  if (!p?.n || !o?.n) continue;
  total++;
  const pr=p.sw/p.n, or=o.sw/o.n;
  if (or < pr-0.03) ojaBetter++; else if (or > pr+0.03) ojaWorse++; else ojaTied++;
}
console.log(`\n=== Verdict: pure_oja vs plain — better ${ojaBetter}/${total}, worse ${ojaWorse}/${total}, tied ${ojaTied}/${total} ===`);
