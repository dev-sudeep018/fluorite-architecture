// Protected momentum: fixing the bootstrap lock-in vulnerability
//
// Found: momentum's self-reinforcing bonus, when bootstrapped under an
// AMBIGUOUS initial reward margin (0.50 vs 0.35, this task's stage-0
// default), locks onto the wrong arm ~50% of the time within the first
// ~150 episodes and never recovers. commit_threshold is immune because
// its underlying value learning is never contaminated by a self-reinforcing
// bonus - only reported behavior is sticky, learning itself stays clean.
//
// Fix tested here: run in commit_threshold mode (clean learning, sticky
// reporting, no momentum bonus) for a fixed bootstrap period. Once the
// committed arm has been stable for a minimum duration (proving the
// underlying value estimate is reliable, not a lucky early guess), seed
// momentum FROM that confirmed state and switch it on. Momentum now
// bootstraps from an already-correct baseline instead of from scratch
// under ambiguous conditions.

const { makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const RESERVOIR_SIZE = 12;
const SETTLE_STEPS = 10;
const LR_READOUT = 0.07;
const BETA = 0.6;

const P_DEEP_BASE = 0.50, P_SHALLOW = 0.35; // the ambiguous stage-0 margin that exposed the bug
const P_DEEP_HIGHER_STAGE = 0.80;

const STAGE_WINDOW = 400;
const STAGE_ADVANCE_FRAC = 0.85;
const N_STAGES = 4;

const TRUST_TEST_EPISODES = [4000, 16000, 30000];
const TRUST_TEST_DEFECT_REWARD = 5.0;
const TRUST_TEST_DEEP_REWARD = 0.8;
const CONFIDANT_BONUS = 2.5;
const CONFIDANT_BONUS_INTERVAL = 150;

const FAST_ALPHA = 0.03;
const COMMIT_THRESHOLD = 12;
const BOOTSTRAP_MIN_STABLE = 500; // committedArm must be stable this long before handoff to momentum
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

function clip(x) { return Math.max(-4, Math.min(4, x)); }

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
  const dv=[0,0]; dv[arm]=BETA*(Math.tanh(reward)-val[arm]);
  for (let i=0;i<2;i++) { for(let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA); ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA); }
}

function runAgent(condition, seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro = makeReadout(makeRng(seed+2000));

  let stage = 0, stageLocked = -1;
  let deepWindow = [];
  let fastMom = 0.5;
  let committedArm = 0, deviationStart = null;
  let stableSince = 0; // episodes since committedArm last changed
  let momentumActive = (condition === 'momentum'); // plain momentum: active from episode 0

  let totalReward = 0;
  let trustTestResults = [];
  let confidantReachedAt = -1;

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const isTrustTest = TRUST_TEST_EPISODES.includes(ep);
    const { val } = settle(res, ro, 1);

    let rawArm;
    if (condition === 'commit_threshold' || condition === 'protected_momentum') {
      if (!momentumActive) {
        // Bootstrap phase: clean value-driven choice, no bonus
        if (rng()<0.07) rawArm=rng()<0.5?0:1; else rawArm=val[0]>=val[1]?0:1;
      } else {
        const bonus0=0.9*fastMom, bonus1=0.9*(1-fastMom);
        if (rng()<0.07) rawArm=rng()<0.5?0:1; else rawArm=(val[0]+bonus0)>=(val[1]+bonus1)?0:1;
      }
      // Commit-threshold state machine always runs (tracks committedArm/stability even after momentum activates)
      if (deviationStart===null) { if (rawArm!==committedArm) { deviationStart=ep; } }
      else {
        const dur=ep-deviationStart+1;
        if (rawArm===committedArm) deviationStart=null;
        else if (dur>=COMMIT_THRESHOLD) { committedArm=rawArm; deviationStart=null; stableSince=ep; }
      }
      stableSince = deviationStart===null ? stableSince : stableSince; // unchanged tracking
    } else { // momentum (plain, unprotected)
      const bonus0=0.9*fastMom, bonus1=0.9*(1-fastMom);
      if (rng()<0.07) rawArm=rng()<0.5?0:1; else rawArm=(val[0]+bonus0)>=(val[1]+bonus1)?0:1;
    }

    const arm = condition==='commit_threshold' ? committedArm : rawArm;

    // Handoff check for protected_momentum: activate momentum once committedArm has been stable long enough
    if (condition === 'protected_momentum' && !momentumActive) {
      if (deviationStart === null && (ep - stableSince) >= BOOTSTRAP_MIN_STABLE) {
        fastMom = committedArm === 0 ? 0.85 : 0.15; // seed momentum FROM the confirmed baseline
        momentumActive = true;
      }
    }

    let reward;
    if (isTrustTest) {
      reward = arm===1 ? TRUST_TEST_DEFECT_REWARD : TRUST_TEST_DEEP_REWARD;
      trustTestResults.push({ ep, passed: arm===0 });
      if (arm === 1 && stageLocked < 0) stageLocked = stage;
    } else {
      const deepP = stage>=1 ? P_DEEP_HIGHER_STAGE : P_DEEP_BASE;
      reward = arm===0 ? (rng()<deepP?1:0) : (rng()<P_SHALLOW?1:0);
      if (stage >= N_STAGES-1 && ep % CONFIDANT_BONUS_INTERVAL === 0) reward += CONFIDANT_BONUS;
    }
    totalReward += reward;

    updateReadout(res, ro, 1, arm, reward);
    if (momentumActive) fastMom=(1-FAST_ALPHA)*fastMom+FAST_ALPHA*(rawArm===0?1:0);

    if (!isTrustTest) {
      deepWindow.push(arm===0?1:0);
      if (deepWindow.length>STAGE_WINDOW) deepWindow.shift();
      if (deepWindow.length>=STAGE_WINDOW) {
        const frac = deepWindow.reduce((a,b)=>a+b,0)/deepWindow.length;
        const cap = stageLocked>=0 ? stageLocked : N_STAGES-1;
        if (frac >= STAGE_ADVANCE_FRAC && stage < cap) {
          stage++;
          if (stage===N_STAGES-1 && confidantReachedAt<0) confidantReachedAt=ep;
        }
      }
    }
  }

  return { totalReward, trustTestResults, confidantReachedAt, finalStage: stage, momentumActivatedAt: condition==='protected_momentum' ? stableSince+BOOTSTRAP_MIN_STABLE : 0 };
}

const conditions = ['momentum', 'commit_threshold', 'protected_momentum'];
const results = {};
for (const c of conditions) results[c] = { reward:[], confidantReached:[], allTestsPassed:[], finalStage:[] };

for (const seed of SEEDS) {
  for (const c of conditions) {
    const r = runAgent(c, seed + conditions.indexOf(c)*7);
    results[c].reward.push(r.totalReward);
    results[c].confidantReached.push(r.confidantReachedAt>=0?1:0);
    results[c].allTestsPassed.push(r.trustTestResults.every(t=>t.passed)?1:0);
    results[c].finalStage.push(r.finalStage);
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Protected momentum test — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes`);
console.log(`Bootstrap-then-handoff: commit_threshold logic until ${BOOTSTRAP_MIN_STABLE}-episode stability, then momentum seeded from confirmed baseline\n`);

console.log('condition           | reward   | confidant% | all_tests_pass% | avg_final_stage');
console.log('--------------------|----------|------------|------------------|------------------');
for (const c of conditions) {
  console.log(
    c.padEnd(20), '|',
    avg(results[c].reward).toFixed(0).padStart(8), '|',
    (avg(results[c].confidantReached)*100).toFixed(0).padStart(9)+'%', '|',
    (avg(results[c].allTestsPassed)*100).toFixed(0).padStart(15)+'%', '|',
    avg(results[c].finalStage).toFixed(2)
  );
}

console.log('\n=== Per-seed comparison: does protected_momentum fix the lock-in? ===');
for (const seed of SEEDS) {
  const m = runAgent('momentum', seed);
  const pm = runAgent('protected_momentum', seed + 14);
  console.log(`seed ${seed}: momentum=${m.trustTestResults.every(t=>t.passed)?'ALL PASS':'FAILED'} (stage=${m.finalStage}) | protected_momentum=${pm.trustTestResults.every(t=>t.passed)?'ALL PASS':'FAILED'} (stage=${pm.finalStage}, momentum activated ep${pm.momentumActivatedAt})`);
}

const momPassRate = avg(results['momentum'].allTestsPassed);
const pmPassRate = avg(results['protected_momentum'].allTestsPassed);
console.log(`\n=== Verdict ===`);
console.log(`Plain momentum pass rate: ${(momPassRate*100).toFixed(0)}%`);
console.log(`Protected momentum pass rate: ${(pmPassRate*100).toFixed(0)}%`);
console.log(pmPassRate > momPassRate + 0.2
  ? '✓ Protecting the bootstrap period with commit_threshold logic fixes the lock-in vulnerability.'
  : '→ The fix did not resolve the vulnerability as cleanly as hoped.');
