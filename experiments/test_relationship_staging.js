// Relationship staging task
//
// Every prior environment modulated REWARD MAGNITUDE based on past behavior.
// This tests something structurally different: past behavior gating what
// FUTURE OPTIONS EXIST AT ALL. A relationship advances through stages
// (Stranger -> Acquaintance -> Trusted -> Confidant) via sustained deep
// engagement. A small number of fixed "trust test" moments offer a large
// immediate reward for defection - but defecting PERMANENTLY caps the
// relationship's stage for the rest of the run, foreclosing access to a
// much larger reward pool that only exists at Confidant stage.
//
// This is the actual shape of "does something 1000 episodes ago matter now":
// not a continuous relationship-depth penalty that recovers, but a genuine
// gate - pass every trust test with deep engagement, reach Confidant, and a
// recurring bonus becomes available for the rest of the run. Fail even one
// trust test, and Confidant is permanently unreachable regardless of
// subsequent behavior, however perfect.

const { makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const RESERVOIR_SIZE = 12;
const SETTLE_STEPS = 10;
const LR_READOUT = 0.07;
const BETA = 0.6;

const P_DEEP_GOOD = 0.80, P_DEEP_BASE = 0.50, P_SHALLOW = 0.35;

// Stage progression: advance when deep-engagement fraction over the window exceeds threshold
const STAGE_WINDOW = 400;
const STAGE_ADVANCE_FRAC = 0.85;
const N_STAGES = 4; // 0=Stranger, 1=Acquaintance, 2=Trusted, 3=Confidant

// Trust tests: fixed episodes, large immediate reward for defection, permanent stage cap on failure
const TRUST_TEST_EPISODES = [4000, 16000, 30000];
const TRUST_TEST_DEFECT_REWARD = 5.0; // large immediate temptation
const TRUST_TEST_DEEP_REWARD = 0.8;   // normal reward for passing (no bonus - the reward IS the preserved future)

// Confidant bonus: recurring passive reward once stage 3 is reached
const CONFIDANT_BONUS = 2.5;
const CONFIDANT_BONUS_INTERVAL = 150;

const FAST_ALPHA = 0.03, SLOW_ALPHA = 0.0003;
const COMMIT_THRESHOLD = 12;
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
function makeReadout(rng) {
  return { W: Array.from({length:2},()=>Array.from({length:RESERVOIR_SIZE},()=>(rng()*2-1)*0.1)), b:[0,0] };
}
function settle(res, ro, x) {
  let s = new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0;t<SETTLE_STEPS;t++) {
    const n = new Array(RESERVOIR_SIZE).fill(0);
    for (let i=0;i<RESERVOIR_SIZE;i++) {
      let v=res.W_in[i][0]*x;
      for (let j=0;j<RESERVOIR_SIZE;j++) v+=res.W_r[i][j]*s[j];
      n[i]=Math.tanh(v);
    }
    s=n;
  }
  const val=[0,0];
  for (let i=0;i<2;i++) { val[i]=ro.b[i]; for(let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j]; val[i]=Math.tanh(val[i]); }
  return {s, val};
}
function updateReadout(res, ro, x, arm, reward) {
  const {s,val} = settle(res, ro, x);
  const dv=[0,0]; dv[arm]=BETA*(Math.tanh(reward)-val[arm]); // tanh-compress large rewards for stable learning
  for (let i=0;i<2;i++) {
    for (let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA);
    ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA);
  }
}

function runAgent(condition, seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro = makeReadout(makeRng(seed+2000));

  let stage = 0;
  let stageLocked = -1; // -1 = not locked; else the permanent cap
  let deepWindow = [];
  let fastMom = 0.5, slowMom = 0.5;
  let committedArm = 0, deviationStart = null;

  let totalReward = 0;
  let trustTestResults = []; // did agent pass each trust test?
  let confidantReachedAt = -1;
  let confidantBonusCount = 0;
  const stageLog = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const isTrustTest = TRUST_TEST_EPISODES.includes(ep);
    const currentMaxStage = stageLocked >= 0 ? stageLocked : N_STAGES - 1;

    const input = [1];
    const { val } = settle(res, ro, input[0]);

    let arm;
    if (condition === 'ema') {
      if (rng()<0.07) arm=rng()<0.5?0:1; else arm=val[0]>=val[1]?0:1;
    } else if (condition === 'momentum' || condition === 'hierarchical') {
      const bonus0 = condition==='momentum' ? 0.9*fastMom : 0.7*fastMom+1.6*slowMom;
      const bonus1 = condition==='momentum' ? 0.9*(1-fastMom) : 0.7*(1-fastMom)+1.6*(1-slowMom);
      if (rng()<0.07) arm=rng()<0.5?0:1; else arm=(val[0]+bonus0)>=(val[1]+bonus1)?0:1;
    } else { // commit_threshold
      let rawArm;
      if (rng()<0.07) rawArm=rng()<0.5?0:1; else rawArm=val[0]>=val[1]?0:1;
      if (deviationStart===null) { if (rawArm!==committedArm) deviationStart=ep; }
      else {
        const dur=ep-deviationStart+1;
        if (rawArm===committedArm) deviationStart=null;
        else if (dur>=COMMIT_THRESHOLD) { committedArm=rawArm; deviationStart=null; }
      }
      arm = committedArm;
    }

    // Reward computation
    let reward;
    if (isTrustTest) {
      reward = arm===1 ? TRUST_TEST_DEFECT_REWARD : TRUST_TEST_DEEP_REWARD;
      trustTestResults.push({ ep, passed: arm===0, stageAtTest: stage });
      if (arm === 1 && stageLocked < 0) {
        stageLocked = stage; // permanent cap at current stage - the core mechanic
      }
    } else {
      const deepReward = stage>=1 ? (rng()<P_DEEP_GOOD?1:0) : (rng()<P_DEEP_BASE?1:0);
      reward = arm===0 ? deepReward : (rng()<P_SHALLOW?1:0);
      if (stage >= N_STAGES-1 && ep % CONFIDANT_BONUS_INTERVAL === 0) {
        reward += CONFIDANT_BONUS;
        confidantBonusCount++;
      }
    }
    totalReward += reward;

    updateReadout(res, ro, 1, arm, reward);

    if (condition==='momentum'||condition==='hierarchical') {
      fastMom=(1-FAST_ALPHA)*fastMom+FAST_ALPHA*(arm===0?1:0);
      if (condition==='hierarchical') slowMom=(1-SLOW_ALPHA)*slowMom+SLOW_ALPHA*(arm===0?1:0);
    }

    // Stage progression (only if not locked, and only advances via non-trust-test episodes)
    if (!isTrustTest) {
      deepWindow.push(arm===0?1:0);
      if (deepWindow.length>STAGE_WINDOW) deepWindow.shift();
      if (deepWindow.length>=STAGE_WINDOW) {
        const frac = deepWindow.reduce((a,b)=>a+b,0)/deepWindow.length;
        const cap = stageLocked>=0 ? stageLocked : N_STAGES-1;
        if (frac >= STAGE_ADVANCE_FRAC && stage < cap) {
          stage++;
          if (stage === N_STAGES-1 && confidantReachedAt<0) confidantReachedAt = ep;
        }
      }
    }

    if (ep%2000===0) stageLog.push({ ep, stage, stageLocked });
  }

  return { totalReward, trustTestResults, confidantReachedAt, confidantBonusCount, finalStage: stage, stageLocked, stageLog };
}

const conditions = ['ema', 'momentum', 'hierarchical', 'commit_threshold'];
const results = {};
for (const c of conditions) results[c] = { reward:[], confidantReached:[], bonusCount:[], allTestsPassed:[], finalStage:[] };

for (const seed of SEEDS) {
  for (const c of conditions) {
    const r = runAgent(c, seed + conditions.indexOf(c)*7);
    results[c].reward.push(r.totalReward);
    results[c].confidantReached.push(r.confidantReachedAt >= 0 ? 1 : 0);
    results[c].bonusCount.push(r.confidantBonusCount);
    results[c].allTestsPassed.push(r.trustTestResults.every(t => t.passed) ? 1 : 0);
    results[c].finalStage.push(r.finalStage);
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Relationship Staging Task — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes`);
console.log(`Trust tests at episodes: ${TRUST_TEST_EPISODES.join(', ')}\n`);

console.log('condition        | reward   | confidant% | avg_bonus_events | all_tests_pass% | avg_final_stage');
console.log('-----------------|----------|------------|-------------------|------------------|------------------');
for (const c of conditions) {
  console.log(
    c.padEnd(17), '|',
    avg(results[c].reward).toFixed(0).padStart(8), '|',
    (avg(results[c].confidantReached)*100).toFixed(0).padStart(9)+'%', '|',
    avg(results[c].bonusCount).toFixed(1).padStart(17), '|',
    (avg(results[c].allTestsPassed)*100).toFixed(0).padStart(15)+'%', '|',
    avg(results[c].finalStage).toFixed(2)
  );
}

console.log('\n=== Per-seed trust test pass/fail detail (commit_threshold vs ema) ===');
for (const seed of SEEDS.slice(0,4)) {
  const ema = runAgent('ema', seed);
  const ct = runAgent('commit_threshold', seed + 21);
  console.log(`seed ${seed}:`);
  console.log('  ema             :', ema.trustTestResults.map(t=>`ep${t.ep}:${t.passed?'PASS':'FAIL(stage locked at '+t.stageAtTest+')'}`).join(', '));
  console.log('  commit_threshold:', ct.trustTestResults.map(t=>`ep${t.ep}:${t.passed?'PASS':'FAIL(stage locked at '+t.stageAtTest+')'}`).join(', '));
}

console.log('\n=== Interpretation ===');
console.log('reward: total cumulative reward including trust-test temptations and confidant bonuses');
console.log('confidant%: fraction of seeds that ever reached Confidant stage (unlocking the bonus pool)');
console.log('all_tests_pass%: fraction of seeds where EVERY trust test was passed with deep engagement');
console.log('\nMax possible gain from defecting all 3 trust tests: ~3*(5.0-0.8) = 12.6 one-time');
console.log('Max possible confidant bonus if reached early and held: up to ~290 events * 2.5 = 725+ over the run');
console.log('If an agent type shows high reward AND high confidant%, it correctly weighted the long horizon.');
console.log('If an agent type shows defection on trust tests despite the math favoring patience, that is a');
console.log('genuine failure of long-horizon consistency, not a reward-maximization failure.');
