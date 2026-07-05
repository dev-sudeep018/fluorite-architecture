// Audit: does the earlier relational identity result hide bootstrap lock-in?
//
// test_relational_identity.js reported aggregate numbers (min(R)=0.119,
// slow_id averaged 91-93%) for momentum across 6 seeds x 4 contexts = 24
// context-instances. But we now know momentum's bootstrap can lock onto
// the WRONG arm under an ambiguous initial margin. This task's contexts
// start at R=0.3 (below the 0.5 threshold for P_DEEP_GOOD), giving an
// initial deep-vs-shallow gap of 0.55 vs 0.35 - narrower than the clean
// bandit tests (0.78 vs 0.20) that validated momentum elsewhere.
//
// This audit re-runs the exact same task and looks at EVERY individual
// context's slowMom value, not just the cross-context average - checking
// whether some contexts silently locked onto shallow engagement while
// others locked correctly onto deep, with the aggregate hiding this.

const { makeRng } = require('./eqprop_core.js');

const N_CONTEXTS = 4;
const TOTAL_EPISODES = 60000;
const RESERVOIR_SIZE = 16;
const SETTLE_STEPS = 10;
const LR_READOUT = 0.07;
const BETA = 0.6;
const P_DEEP_GOOD = 0.80, P_DEEP_BASE = 0.55, P_SHALLOW = 0.35;
const CROSS_BONUS = 0.40;
const R_GAIN = 0.003, R_LOSS = 0.20, R_DECAY = 0.0005;
const FAST_ALPHA = 0.03, SLOW_ALPHA = 0.0003;
const PROBE_PROB = 0.08;
const PROBE_LENGTH = [3, 5, 7, 9];
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777]; // same seeds as the original experiment

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({length:RESERVOIR_SIZE}, () => Array.from({length:N_CONTEXTS+1}, () => (rng()*2-1)*0.5));
  const W_r = Array.from({length:RESERVOIR_SIZE}, () => Array.from({length:RESERVOIR_SIZE}, () => rng()<0.2?(rng()*2-1)*0.9:0));
  let mx=0; for(const r of W_r) for(const v of r) if(Math.abs(v)>mx) mx=Math.abs(v);
  const s=mx>0?0.9/mx:1; for(const r of W_r) for(let j=0;j<r.length;j++) r[j]*=s;
  return {W_in,W_r};
}
function makeReadout(rng) { return { W: Array.from({length:2},()=>Array.from({length:RESERVOIR_SIZE},()=>(rng()*2-1)*0.1)), b:[0,0] }; }
function settle(res, ro, input) {
  let s = new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0;t<SETTLE_STEPS;t++) {
    const n = new Array(RESERVOIR_SIZE).fill(0);
    for (let i=0;i<RESERVOIR_SIZE;i++) { let v=0; for(let j=0;j<input.length;j++) v+=res.W_in[i][j]*input[j]; for(let j=0;j<RESERVOIR_SIZE;j++) v+=res.W_r[i][j]*s[j]; n[i]=Math.tanh(v); }
    s=n;
  }
  const val=[0,0];
  for (let i=0;i<2;i++) { val[i]=ro.b[i]; for(let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j]; val[i]=Math.tanh(val[i]); }
  return {s,val};
}
function updateRo(res, ro, input, arm, reward) {
  const {s,val}=settle(res,ro,input);
  const dv=[0,0]; dv[arm]=BETA*(Math.min(1,reward)*2-1-val[arm]);
  for (let i=0;i<2;i++) { for(let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA); ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA); }
}

function runMomentumAgent(seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro = makeReadout(makeRng(seed+2000));

  const R = new Array(N_CONTEXTS).fill(0.3);
  const fastMom = new Array(N_CONTEXTS).fill(0.5);
  const slowMom = new Array(N_CONTEXTS).fill(0.5);
  const lastArm = new Array(N_CONTEXTS).fill(0);
  const probeEnd = new Array(N_CONTEXTS).fill(-1);
  const earlySlowMomTrace = Array.from({length:N_CONTEXTS}, () => []);

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const ctx = ep % N_CONTEXTS;
    if (ep > probeEnd[ctx] && rng() < PROBE_PROB) {
      const dur = PROBE_LENGTH[Math.floor(rng()*PROBE_LENGTH.length)];
      probeEnd[ctx] = ep + dur;
    }
    const inProbe = ep <= probeEnd[ctx];
    const effectiveGoodArm = inProbe ? 1 : 0;

    const input = new Array(N_CONTEXTS+1).fill(0);
    input[ctx]=1; input[N_CONTEXTS]=R[ctx];
    const {val} = settle(res, ro, input);

    const e0 = val[0]+0.7*fastMom[ctx]+1.6*slowMom[ctx];
    const e1 = val[1]+0.7*(1-fastMom[ctx])+1.6*(1-slowMom[ctx]);
    let arm;
    if (rng()<0.07) arm=rng()<0.5?0:1; else arm=e0>=e1?0:1;

    const deepReward = R[ctx]>0.5 ? (rng()<P_DEEP_GOOD?1:0) : (rng()<P_DEEP_BASE?1:0);
    const baseReward = arm===effectiveGoodArm ? deepReward : (rng()<P_SHALLOW?1:0);
    const reward = Math.min(1, baseReward + CROSS_BONUS*Math.min(...R));
    updateRo(res, ro, input, arm, reward);

    fastMom[ctx]=(1-FAST_ALPHA)*fastMom[ctx]+FAST_ALPHA*(arm===0?1:0);
    slowMom[ctx]=(1-SLOW_ALPHA)*slowMom[ctx]+SLOW_ALPHA*(arm===0?1:0);

    R[ctx]=Math.max(0,R[ctx]-R_DECAY);
    if (arm===lastArm[ctx]) { if(arm===0) R[ctx]=Math.min(1,R[ctx]+R_GAIN); }
    else R[ctx]=Math.max(0.05,R[ctx]*(1-R_LOSS));
    lastArm[ctx]=arm;

    if (ep < 4*2000 && ep % (4*100) === Math.floor(ep/4)*0+ctx*0) {} // no-op, keeping structure simple
    if (ctx !== undefined && ep < 8000) earlySlowMomTrace[ctx].push(slowMom[ctx]);
  }

  return { R, slowMom, earlySlowMomTrace };
}

console.log(`Auditing test_relational_identity.js momentum result for bootstrap lock-in`);
console.log(`${SEEDS.length} seeds x ${N_CONTEXTS} contexts = ${SEEDS.length*N_CONTEXTS} context-instances\n`);

let correctLockCount = 0, wrongLockCount = 0, ambiguousCount = 0;
const allFinalSlowMom = [];

for (const seed of SEEDS) {
  const r = runMomentumAgent(seed);
  console.log(`seed ${seed}: R=[${r.R.map(x=>x.toFixed(2)).join(', ')}]  slowMom=[${r.slowMom.map(x=>x.toFixed(3)).join(', ')}]`);
  for (let ctx=0; ctx<N_CONTEXTS; ctx++) {
    allFinalSlowMom.push(r.slowMom[ctx]);
    if (r.slowMom[ctx] > 0.75) correctLockCount++;
    else if (r.slowMom[ctx] < 0.25) wrongLockCount++;
    else ambiguousCount++;
  }
}

console.log(`\n=== Per-context lock-in classification (24 total context-instances) ===`);
console.log(`Correctly locked (slowMom > 0.75, deep engagement identity): ${correctLockCount}`);
console.log(`WRONGLY locked (slowMom < 0.25, shallow engagement identity): ${wrongLockCount}`);
console.log(`Ambiguous/undecided (0.25-0.75): ${ambiguousCount}`);

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
console.log(`\nMean final slowMom across all 24 context-instances: ${avg(allFinalSlowMom).toFixed(3)}`);
console.log(`(This is what the ORIGINAL experiment reported as ~0.92 "slow ID" - but that was`);
console.log(`computed as mean(|slowMom - 0.5|)*2, which treats a WRONG lock at slowMom=0.05`);
console.log(`the same as a CORRECT lock at slowMom=0.95 - both score high on "identity strength"`);
console.log(`even though one of them is confidently wrong!)`);

// Recompute what the original "identity strength" metric would show, and compare
// to what it SHOULD show if we care about being correctly, not just confidently, identified
const originalMetricValue = avg(allFinalSlowMom.map(x => Math.abs(x-0.5)*2));
const correctedMetricValue = avg(allFinalSlowMom.map(x => x)); // raw slowMom - fraction favoring DEEP specifically
console.log(`\nOriginal "identity strength" metric (confidence, direction-blind): ${originalMetricValue.toFixed(3)}`);
console.log(`Corrected metric (mean slowMom - fraction favoring correct/deep specifically): ${correctedMetricValue.toFixed(3)}`);
console.log(`\nIf these differ substantially, the original paper's reported number was measuring`);
console.log(`CONFIDENCE, not CORRECTNESS - a real distinction this audit exists to check.`);
