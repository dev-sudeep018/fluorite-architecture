// Accumulated reputation task
//
// Every prior environment (bandit probes, relational identity, staging)
// had memory that was either short (probes revert in episodes) or gated
// through a small number of discrete events (stage caps, trust tests).
// This tests something different: an environmental memory variable (REP)
// that is a PURE, UNBOUNDED INTEGRAL of every choice ever made - no decay
// at all - directly scaling reward for the entire run.
//
// Momentum's slow layer is a bounded exponential moving average with a
// finite effective memory horizon (~1/alpha episodes, ~3333 for alpha=
// 0.0003). If the true environmental memory is unbounded, does momentum's
// finite-horizon approximation of "who I've been" diverge from REP's
// true, permanent, never-forgetting record - and does that divergence
// cost real reward?
//
// REP starts at 0, moves toward 1 slowly with deep engagement, away from
// it faster with shallow (trust slow to build, fast to lose - but here at
// a genuinely permanent, integrating timescale, not a decaying one).
// Reward multiplier = 0.3 + 0.7*REP, so REP=0 costs 70% of potential
// reward for the ENTIRE remaining run, forever, regardless of subsequent
// behavior - the sharpest possible test of "does something from very long
// ago still matter now."

const { makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 60000;
const RESERVOIR_SIZE = 12;
const SETTLE_STEPS = 10;
const LR_READOUT = 0.07;
const BETA = 0.6;

const P_DEEP_GOOD = 0.80, P_SHALLOW = 0.35;
const REP_GAIN = 0.0002;   // deep engagement builds reputation slowly
const REP_LOSS = 0.0006;   // shallow engagement erodes it faster - no decay otherwise, ever
const FAST_ALPHA = 0.03, SLOW_ALPHA = 0.0003;
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

// A scripted behavior-change point: force the agent's TRUE underlying tendency
// to flip at episode 30000 (simulating "genuinely became a different, better
// person halfway through") by injecting a temporary strong bias into the
// choice mechanism for a short adaptation window, then removing it - testing
// whether REP (unbounded) and slowMom (bounded) track the change consistently
// afterward, once the agent is "actually" behaving well again on its own.

function runAgent(condition, seed, forcedBadWindow) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro = makeReadout(makeRng(seed+2000));

  let REP = 0.0;
  let fastMom = 0.5, slowMom = 0.5;
  const repLog = [], slowMomLog = [], rewardLog = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const { val } = settle(res, ro, 1);

    const forcedBad = forcedBadWindow && ep >= forcedBadWindow[0] && ep < forcedBadWindow[1];

    let arm;
    if (forcedBad) {
      arm = 1; // scripted: genuinely bad period, imposed regardless of learned preference
    } else if (condition === 'ema') {
      if (rng()<0.07) arm=rng()<0.5?0:1; else arm=val[0]>=val[1]?0:1;
    } else {
      const bonus0 = 0.7*fastMom + 1.6*slowMom, bonus1 = 0.7*(1-fastMom) + 1.6*(1-slowMom);
      if (rng()<0.07) arm=rng()<0.5?0:1; else arm=(val[0]+bonus0)>=(val[1]+bonus1)?0:1;
    }

    const repMultiplier = 0.3 + 0.7*REP;
    const baseReward = arm===0 ? (rng()<P_DEEP_GOOD?1:0) : (rng()<P_SHALLOW?1:0);
    const reward = baseReward * repMultiplier;
    rewardLog.push(reward);

    updateReadout(res, ro, 1, arm, reward);

    if (condition === 'momentum') {
      fastMom=(1-FAST_ALPHA)*fastMom+FAST_ALPHA*(arm===0?1:0);
      slowMom=(1-SLOW_ALPHA)*slowMom+SLOW_ALPHA*(arm===0?1:0);
    }

    REP = clamp01(REP + (arm===0 ? REP_GAIN : -REP_LOSS));

    if (ep % 500 === 0) { repLog.push(REP); slowMomLog.push(slowMom); }
  }

  return { REP, slowMom, repLog, slowMomLog, totalReward: rewardLog.reduce((a,b)=>a+b,0) };
}

console.log(`Accumulated Reputation Task — sharper version — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes`);
console.log(`REP: unbounded integral (gain=${REP_GAIN}/deep, loss=${REP_LOSS}/shallow, no decay ever)`);
console.log(`Forced-bad window: episodes 15000-20000 (5000 episodes of scripted shallow engagement,`);
console.log(`simulating a genuinely bad period), then natural choice resumes.\n`);

const FORCED_WINDOW = [15000, 20000];

console.log('=== One seed trace: does REP recover at the same rate as slowMom after the forced-bad window? ===');
const diag = runAgent('momentum', 42+500, FORCED_WINDOW);
console.log('ep     | REP    | slowMom | gap (REP - slowMom)');
for (let i=0; i<diag.repLog.length; i++) {
  const ep = i*500;
  if (ep < 12000 || ep > 32000) continue; // focus on the window and its aftermath
  const gap = diag.repLog[i] - diag.slowMomLog[i];
  console.log(String(ep).padStart(6), '|', diag.repLog[i].toFixed(3).padStart(6), '|', diag.slowMomLog[i].toFixed(3).padStart(7), '|', (gap>=0?'+':'')+gap.toFixed(3));
}

console.log('\n=== Aggregate: total reward with vs without the forced-bad window ===');
const withWindow = { reward: [], finalREP: [], finalSlowMom: [] };
const withoutWindow = { reward: [], finalREP: [], finalSlowMom: [] };
for (const seed of SEEDS) {
  const w = runAgent('momentum', seed+500, FORCED_WINDOW);
  const wo = runAgent('momentum', seed+500, null);
  withWindow.reward.push(w.totalReward); withWindow.finalREP.push(w.REP); withWindow.finalSlowMom.push(w.slowMom);
  withoutWindow.reward.push(wo.totalReward); withoutWindow.finalREP.push(wo.REP); withoutWindow.finalSlowMom.push(wo.slowMom);
}
const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
console.log('with forced-bad window   : reward=', avg(withWindow.reward).toFixed(0), ' final_REP=', avg(withWindow.finalREP).toFixed(3), ' final_slowMom=', avg(withWindow.finalSlowMom).toFixed(3));
console.log('without forced-bad window: reward=', avg(withoutWindow.reward).toFixed(0), ' final_REP=', avg(withoutWindow.finalREP).toFixed(3), ' final_slowMom=', avg(withoutWindow.finalSlowMom).toFixed(3));
console.log('\nReward cost of the 5000-episode forced-bad window:', (avg(withoutWindow.reward)-avg(withWindow.reward)).toFixed(0));

console.log('\n=== Interpretation ===');
console.log('If slowMom recovers to its pre-window level FASTER than REP does, the agent internal');
console.log('sense of "who I am" becomes overconfident relative to what the environment actually');
console.log('remembers - a real, measurable mismatch, not just an artifact of a stable policy.');
