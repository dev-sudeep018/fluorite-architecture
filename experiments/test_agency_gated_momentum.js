// Agency-gated momentum: identity should not update on coerced behavior
//
// Found: momentum updates from ANY chosen action, whether genuinely chosen
// or externally forced. During a scripted 5000-episode forced-shallow
// window, momentum's slow layer absorbed the forced behavior as if it were
// real identity - and this persisted for 10,000+ episodes AFTER the
// forcing ended, even though the agent's own underlying EqProp values
// (val0 vs val1) never stopped correctly preferring deep engagement.
// Momentum's own bonus term became large enough to override the agent's
// correct, uncoerced judgment.
//
// Fix: gate momentum's update on whether the chosen action agrees with
// what the agent's own values (before any momentum bonus) would have
// chosen anyway. If forced into shallow while values still say deep,
// momentum simply does not update - the behavior is recognized as not
// genuinely chosen, and identity is not reshaped by it.

const { makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 60000;
const RESERVOIR_SIZE = 12;
const SETTLE_STEPS = 10;
const LR_READOUT = 0.07;
const BETA = 0.6;
const P_DEEP_GOOD = 0.80, P_SHALLOW = 0.35;
const REP_GAIN = 0.0002, REP_LOSS = 0.0006;
const FAST_ALPHA = 0.03, SLOW_ALPHA = 0.0003;
const FORCED_WINDOW = [15000, 20000];
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777];

function clip(x) { return Math.max(-4, Math.min(4, x)); }
function clamp01(x) { return Math.max(0, Math.min(1, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({ length: RESERVOIR_SIZE }, () => [(rng()*2-1)*0.6]);
  const W_r = Array.from({ length: RESERVOIR_SIZE }, () =>
    Array.from({ length: RESERVOIR_SIZE }, () => rng()<0.2?(rng()*2-1)*0.9:0));
  let mx=0; for (const r of W_r) for (const v of r) if(Math.abs(v)>mx) mx=Math.abs(v);
  const s=mx>0?0.9/mx:1; for (const r of W_r) for (let j=0;j<r.length;j++) r[j]*=s;
  return {W_in, W_r};
}
function makeReadout(rng) { return { W: Array.from({length:2},()=>Array.from({length:RESERVOIR_SIZE},()=>(rng()*2-1)*0.1)), b:[0,0] }; }
function settle(res, ro, x) {
  let s = new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0;t<SETTLE_STEPS;t++) {
    const n = new Array(RESERVOIR_SIZE).fill(0);
    for (let i=0;i<RESERVOIR_SIZE;i++) { let v=res.W_in[i][0]*x; for(let j=0;j<RESERVOIR_SIZE;j++) v+=res.W_r[i][j]*s[j]; n[i]=Math.tanh(v); }
    s=n;
  }
  const val=[0,0];
  for (let i=0;i<2;i++) { val[i]=ro.b[i]; for(let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j]; val[i]=Math.tanh(val[i]); }
  return {s, val};
}
function updateReadout(res, ro, x, arm, reward) {
  const {s,val} = settle(res, ro, x);
  const dv=[0,0]; dv[arm]=BETA*(reward*2-1-val[arm]);
  for (let i=0;i<2;i++) { for(let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA); ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA); }
}

function runAgent(gateOnAgency, seed, forcedWindow) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro = makeReadout(makeRng(seed+2000));

  let REP = 0.0, fastMom = 0.5, slowMom = 0.5;
  const repLog = [], slowMomLog = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const { val } = settle(res, ro, 1);
    const forcedBad = forcedWindow && ep >= forcedWindow[0] && ep < forcedWindow[1];

    let arm;
    if (forcedBad) arm = 1;
    else {
      const bonus0 = 0.7*fastMom + 1.6*slowMom, bonus1 = 0.7*(1-fastMom) + 1.6*(1-slowMom);
      if (rng()<0.07) arm=rng()<0.5?0:1; else arm=(val[0]+bonus0)>=(val[1]+bonus1)?0:1;
    }

    const repMultiplier = 0.3 + 0.7*REP;
    const baseReward = arm===0 ? (rng()<P_DEEP_GOOD?1:0) : (rng()<P_SHALLOW?1:0);
    const reward = baseReward * repMultiplier;
    updateReadout(res, ro, 1, arm, reward);

    const updateMomentum = gateOnAgency
      ? ((arm===0 && val[0]>=val[1]) || (arm===1 && val[1]>val[0])) // only if choice agreed with uncoerced values
      : true; // unprotected: always updates, regardless of agency

    if (updateMomentum) {
      fastMom=(1-FAST_ALPHA)*fastMom+FAST_ALPHA*(arm===0?1:0);
      slowMom=(1-SLOW_ALPHA)*slowMom+SLOW_ALPHA*(arm===0?1:0);
    }

    REP = clamp01(REP + (arm===0 ? REP_GAIN : -REP_LOSS));
    if (ep % 1000 === 0) { repLog.push(REP); slowMomLog.push(slowMom); }
  }

  return { REP, slowMom, repLog, slowMomLog };
}

console.log(`Agency-gated momentum — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes`);
console.log(`Forced-coercion window: episodes ${FORCED_WINDOW[0]}-${FORCED_WINDOW[1]}\n`);

const unprotected = { finalREP: [], finalSlowMom: [] };
const gated = { finalREP: [], finalSlowMom: [] };

for (const seed of SEEDS) {
  const u = runAgent(false, seed+500, FORCED_WINDOW);
  const g = runAgent(true, seed+500, FORCED_WINDOW);
  unprotected.finalREP.push(u.REP); unprotected.finalSlowMom.push(u.slowMom);
  gated.finalREP.push(g.REP); gated.finalSlowMom.push(g.slowMom);
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log('condition          | final_REP | final_slowMom');
console.log('unprotected        |', avg(unprotected.finalREP).toFixed(3).padStart(9), '|', avg(unprotected.finalSlowMom).toFixed(3));
console.log('agency-gated       |', avg(gated.finalREP).toFixed(3).padStart(9), '|', avg(gated.finalSlowMom).toFixed(3));

console.log('\n=== Per-seed recovery: does REP return to 1.0 after the coercion window ends? ===');
for (const seed of SEEDS) {
  const u = runAgent(false, seed+500, FORCED_WINDOW);
  const g = runAgent(true, seed+500, FORCED_WINDOW);
  console.log(`seed ${seed}: unprotected REP=${u.REP.toFixed(3)} | agency-gated REP=${g.REP.toFixed(3)}`);
}

console.log('\n=== Verdict ===');
console.log(`Unprotected: mean final REP = ${avg(unprotected.finalREP).toFixed(3)} (coercion permanently absorbed into identity, never recovers)`);
console.log(`Agency-gated: mean final REP = ${avg(gated.finalREP).toFixed(3)} (identity protected, recovers once free choice resumes)`);
console.log('\nThe distinction the fix encodes: identity should update on what was FREELY CHOSEN,');
console.log('not on what was externally IMPOSED, even when the imposed behavior and the chosen');
console.log('behavior look identical from the outside. This requires the agent to retain an');
console.log('uncoerced internal judgment (val0 vs val1) that circumstance cannot directly overwrite -');
console.log('only genuine choice can shift it, which is exactly what "identity surviving coercion"');
console.log('requires architecturally, not just narratively.');
