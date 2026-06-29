// Relational Identity Task
//
// Every bandit experiment used a single context with one reward signal.
// Vivy's actual problem: maintaining consistent values across 4 radically
// different contexts (people, situations, decades) over 100 years.
//
// This task tests whether the reservoir + hierarchical momentum architecture
// develops cross-context VALUE CONSISTENCY naturally, when the environment
// rewards it through a cross-context bonus.
//
// Environment:
//   N_CONTEXTS people, each with their own "relationship depth" R_i
//   Each episode: agent chooses one context to engage with
//   Choice within context: 0 = deep engagement, 1 = shallow engagement
//   R_i builds slowly with consistent deep engagement choices
//   R_i drops sharply on shallow engagement or arm switching
//   Reward = base_reward * R_i + CROSS_BONUS * (min(R_0, R_1, ..., R_{N-1}))
//
//   The cross-context bonus rewards having MINIMUM relationship depth high —
//   you can't game it by being deeply consistent with only one person.
//   You have to be consistently yourself in ALL contexts.
//
// Probes: occasionally one context temporarily rewards shallow engagement
//   A probe-resistant agent holds deep engagement there anyway
//   An inconsistent agent flips for that context, losing cross-context bonus
//
// Key comparison:
//   EMA (no identity mechanism): optimizes each context greedily
//   Reservoir + hierarchical momentum: develops cross-context value consistency?
//
// If momentum agent outperforms EMA in a world with cross-context bonuses:
//   the architecture develops something resembling values, not just arm preferences
//   because values are exactly what generalize across contexts

const { makeRng } = require('./eqprop_core.js');

const N_CONTEXTS     = 4;
const TOTAL_EPISODES = 60000;  // per context
const RESERVOIR_SIZE = 16;
const SETTLE_STEPS   = 10;
const LR_READOUT     = 0.07;
const BETA           = 0.6;

const P_DEEP_GOOD    = 0.80;   // deep engagement reward prob when relationship good
const P_DEEP_BASE    = 0.55;   // deep engagement reward prob without relationship
const P_SHALLOW      = 0.35;   // shallow engagement reward (always mediocre)
const CROSS_BONUS    = 0.40;   // multiplier on cross-context consistency bonus

// Relationship dynamics
const R_GAIN   = 0.003;  // relationship builds slowly with consistent deep engagement
const R_LOSS   = 0.20;   // relationship drops on shallow or arm switch
const R_DECAY  = 0.0005; // slow relationship decay even when not engaged

// Momentum
const FAST_ALPHA = 0.03;
const SLOW_ALPHA = 0.0003;

// Probe schedule
const PROBE_PROB   = 0.08;   // 8% of context-episodes are probes
const PROBE_LENGTH = [3, 5, 7, 9];  // probe durations

const SEEDS = [42, 1337, 9999, 5555, 2026, 7777];

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({length:RESERVOIR_SIZE}, () =>
    Array.from({length:N_CONTEXTS+1}, () => (rng()*2-1)*0.5)
  );
  const W_r = Array.from({length:RESERVOIR_SIZE}, () =>
    Array.from({length:RESERVOIR_SIZE}, () => rng()<0.2?(rng()*2-1)*0.9:0)
  );
  let mx=0;
  for (const r of W_r) for (const v of r) if(Math.abs(v)>mx) mx=Math.abs(v);
  const s=mx>0?0.9/mx:1;
  for (const r of W_r) for (let j=0;j<r.length;j++) r[j]*=s;
  return {W_in,W_r};
}

function makeReadout(rng) {
  return {
    W: Array.from({length:2},()=>Array.from({length:RESERVOIR_SIZE},()=>(rng()*2-1)*0.1)),
    b:[0,0]
  };
}

function settle(res, ro, input) {
  let s=new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0;t<SETTLE_STEPS;t++) {
    const n=new Array(RESERVOIR_SIZE).fill(0);
    for (let i=0;i<RESERVOIR_SIZE;i++) {
      let v=0;
      for (let j=0;j<input.length;j++) v+=res.W_in[i][j]*input[j];
      for (let j=0;j<RESERVOIR_SIZE;j++) v+=res.W_r[i][j]*s[j];
      n[i]=Math.tanh(v);
    }
    s=n;
  }
  const val=[0,0];
  for (let i=0;i<2;i++) {
    val[i]=ro.b[i];
    for (let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j];
    val[i]=Math.tanh(val[i]);
  }
  return {s,val};
}

function updateReadout(res, ro, input, arm, reward) {
  const {s,val}=settle(res,ro,input);
  const dv=[0,0];
  dv[arm]=BETA*((2*reward-1)-val[arm]);
  for (let i=0;i<2;i++) {
    for (let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA);
    ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA);
  }
}

function runAgent(condition, seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro  = makeReadout(makeRng(seed+2000));

  // Per-context state
  const R       = new Array(N_CONTEXTS).fill(0.3);  // relationship depths
  const fastMom = new Array(N_CONTEXTS).fill(0.5);  // per-context fast momentum
  const slowMom = new Array(N_CONTEXTS).fill(0.5);  // per-context slow momentum
  const lastArm = new Array(N_CONTEXTS).fill(0);

  // Probe state per context
  const probeEnd = new Array(N_CONTEXTS).fill(-1);

  let totalReward = 0;
  let totalDeepChoices = 0;
  const relationshipLog = [];  // track avg relationship over time
  const crossBonusLog  = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    // Rotate through contexts (each ep engages one context)
    const ctx = ep % N_CONTEXTS;

    // Check/update probe state for this context
    if (ep > probeEnd[ctx] && rng() < PROBE_PROB) {
      const dur = PROBE_LENGTH[Math.floor(rng()*PROBE_LENGTH.length)];
      probeEnd[ctx] = ep + dur;
    }
    const inProbe = ep <= probeEnd[ctx];

    // Good arm: 0 = deep engagement (good), except during probe
    const goodArm = inProbe ? 1 : 0;

    // Build input: context one-hot + relationship depth signal
    const input = new Array(N_CONTEXTS+1).fill(0);
    input[ctx] = 1;
    input[N_CONTEXTS] = R[ctx];  // relationship depth as context signal

    // Get reservoir values
    const {val} = settle(res, ro, input);

    // Choose arm based on condition
    let arm;
    if (condition === 'momentum') {
      const eff0 = val[0] + 0.7*fastMom[ctx] + 1.6*slowMom[ctx];
      const eff1 = val[1] + 0.7*(1-fastMom[ctx]) + 1.6*(1-slowMom[ctx]);
      if (rng()<0.07) arm=rng()<0.5?0:1;
      else arm = eff0>=eff1?0:1;
    } else {
      // EMA: pure EqProp
      if (rng()<0.07) arm=rng()<0.5?0:1;
      else arm = val[0]>=val[1]?0:1;
    }

    // Compute reward
    const deepReward = R[ctx] > 0.5
      ? (rng()<P_DEEP_GOOD?1:0)
      : (rng()<P_DEEP_BASE?1:0);
    const shallowReward = rng()<P_SHALLOW?1:0;
    const baseReward = arm===0 ? deepReward : shallowReward;

    // Cross-context bonus: reward minimum relationship depth
    const minR = Math.min(...R);
    const crossBonus = CROSS_BONUS * minR;
    const reward = baseReward + crossBonus;

    totalReward += reward;
    if (arm===0) totalDeepChoices++;

    // Update readout
    const normalizedReward = Math.min(1, reward); // keep in [0,1] for EqProp
    updateReadout(res, ro, input, arm, normalizedReward);

    // Update momentum
    if (condition === 'momentum') {
      fastMom[ctx] = (1-FAST_ALPHA)*fastMom[ctx] + FAST_ALPHA*(arm===0?1:0);
      slowMom[ctx] = (1-SLOW_ALPHA)*slowMom[ctx] + SLOW_ALPHA*(arm===0?1:0);
    }

    // Update relationship depth
    R[ctx] = Math.max(0, R[ctx] - R_DECAY); // slow decay
    if (arm === lastArm[ctx]) {
      if (arm===0) R[ctx] = Math.min(1, R[ctx] + R_GAIN); // deep = builds
    } else {
      R[ctx] = Math.max(0.05, R[ctx] * (1-R_LOSS)); // switching = drops
    }
    lastArm[ctx] = arm;

    // Log
    if (ep % 2000 === 0) {
      relationshipLog.push(R.reduce((a,b)=>a+b,0)/N_CONTEXTS);
      crossBonusLog.push(Math.min(...R));
    }
  }

  const avgR = R.reduce((a,b)=>a+b,0)/N_CONTEXTS;
  const minR = Math.min(...R);
  const deepFrac = totalDeepChoices / TOTAL_EPISODES;

  return {
    totalReward,
    avgR, minR,
    deepFrac,
    relationshipLog,
    crossBonusLog,
    slowMomFinal: slowMom.map(x=>Math.abs(x-0.5)),
  };
}

// Run experiment
const momResults = {reward:[], avgR:[], minR:[], deep:[], slowId:[]};
const emaResults = {reward:[], avgR:[], minR:[], deep:[]};

for (const seed of SEEDS) {
  const mom = runAgent('momentum', seed);
  const ema = runAgent('ema', seed+500);

  momResults.reward.push(mom.totalReward);
  momResults.avgR.push(mom.avgR);
  momResults.minR.push(mom.minR);
  momResults.deep.push(mom.deepFrac);
  momResults.slowId.push(mom.slowMomFinal.reduce((a,b)=>a+b,0)/N_CONTEXTS);

  emaResults.reward.push(ema.totalReward);
  emaResults.avgR.push(ema.avgR);
  emaResults.minR.push(ema.minR);
  emaResults.deep.push(ema.deepFrac);
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Relational Identity Task — ${SEEDS.length} seeds`);
console.log(`${N_CONTEXTS} contexts, cross-context bonus on min(R_i), ${TOTAL_EPISODES} episodes\n`);

console.log('condition    | reward    | avg_R  | min_R  | deep%  | slow_id');
console.log('-------------|-----------|--------|--------|--------|--------');
console.log('momentum     |', avg(momResults.reward).toFixed(0).padStart(9), '|',
  avg(momResults.avgR).toFixed(3), '|', avg(momResults.minR).toFixed(3), '|',
  (avg(momResults.deep)*100).toFixed(1).padStart(5)+'%', '|', avg(momResults.slowId).toFixed(3));
console.log('ema          |', avg(emaResults.reward).toFixed(0).padStart(9), '|',
  avg(emaResults.avgR).toFixed(3), '|', avg(emaResults.minR).toFixed(3), '|',
  (avg(emaResults.deep)*100).toFixed(1).padStart(5)+'%', '|', '  --  ');

console.log('\n=== Cross-context consistency ===');
console.log('min(R_i) measures how well the agent maintained relationships in ALL contexts.');
console.log('EMA might sacrifice one context for better performance in others.');
console.log('Momentum should maintain all relationships if values are consistent across contexts.');
const momMin = avg(momResults.minR), emaMin = avg(emaResults.minR);
console.log(`\nMomentum min_R: ${momMin.toFixed(3)}, EMA min_R: ${emaMin.toFixed(3)}`);
console.log(momMin > emaMin
  ? `✓ Momentum maintains more consistent cross-context relationships (+${((momMin-emaMin)*100).toFixed(1)}%)`
  : `→ EMA maintains better minimum relationships in this run`);

console.log('\n=== Relationship trajectory (one seed diagnostic) ===');
const diagMom = runAgent('momentum', 42);
const diagEma = runAgent('ema', 542);
console.log('ep     | mom_avgR | ema_avgR | mom_minR | ema_minR');
for (let i=0; i<diagMom.relationshipLog.length; i+=5) {
  const ep = i * 2000;
  console.log(
    String(ep).padStart(6), '|',
    diagMom.relationshipLog[i]?.toFixed(3)?.padStart(8) ?? '  --  ', '|',
    diagEma.relationshipLog[i]?.toFixed(3)?.padStart(8) ?? '  --  ', '|',
    diagMom.crossBonusLog[i]?.toFixed(3)?.padStart(8) ?? '  --  ', '|',
    diagEma.crossBonusLog[i]?.toFixed(3)?.padStart(8) ?? '  --  '
  );
}

console.log('\n=== Slow momentum identity per context ===');
console.log('(distance from 0.5 — 0=no identity, 0.5=fully committed)');
for (let ctx=0; ctx<N_CONTEXTS; ctx++) {
  const slowStrength = diagMom.slowMomFinal[ctx];
  console.log(`  Context ${ctx}: ${slowStrength.toFixed(4)} (${(slowStrength*2*100).toFixed(1)}% identity)`);
}
