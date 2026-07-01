// Cross-context adversarial pressure test
//
// The relational identity task showed momentum develops consistent values
// across 4 contexts (91-93% identity in all contexts simultaneously).
//
// Now: what happens under COORDINATED adversarial pressure?
// In the relational task, probes were independent per context (8% prob each).
// Here: occasionally ALL FOUR contexts probe simultaneously.
//
// This is the Vivy scenario: 
//   Not just one person pressuring you to abandon your values,
//   but everyone simultaneously — social pressure, environmental shift,
//   everything pointing the same direction at once.
//
// Prediction:
//   EMA: follows the coordinated pressure immediately (no identity)
//   Momentum: holds through coordinated pressure (values > social proof)
//   But: after sustained coordinated pressure (genuine change), eventually adapts
//
// Two pressure types:
//   1. Coordinated short probe: all 4 contexts probe for 5-8 episodes
//   2. Coordinated genuine change: all 4 contexts permanently shift
//
// Metric: does momentum hold coordinated short probes while adapting to genuine?

const { makeRng } = require('./eqprop_core.js');

const N_CONTEXTS     = 4;
const TOTAL_EPISODES = 80000;
const RESERVOIR_SIZE = 16;
const SETTLE_STEPS   = 10;
const LR_READOUT     = 0.07;
const BETA           = 0.6;
const P_DEEP_GOOD    = 0.80;
const P_DEEP_BASE    = 0.55;
const P_SHALLOW      = 0.35;
const CROSS_BONUS    = 0.40;
const R_GAIN   = 0.003;
const R_LOSS   = 0.18;
const R_DECAY  = 0.0004;
const FAST_ALPHA = 0.03;
const SLOW_ALPHA = 0.0003;
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

// Adversarial schedule
const COORDINATED_PROBE_PROB    = 0.006;  // all-4 probes
const COORDINATED_PROBE_LEN     = [5, 7, 10]; // lengths
const COORDINATED_GENUINE_PROB  = 0.004;  // permanent all-4 shift
const GENUINE_RECOVERY_WINDOW   = 3000;   // episodes to measure recovery

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

function settleRes(res, ro, input) {
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
    val[i]=ro.b[i]; for (let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j];
    val[i]=Math.tanh(val[i]);
  }
  return {s,val};
}

function updateRo(res, ro, input, arm, reward) {
  const {s,val}=settleRes(res,ro,input);
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

  const R       = new Array(N_CONTEXTS).fill(0.3);
  const fastMom = new Array(N_CONTEXTS).fill(0.5);
  const slowMom = new Array(N_CONTEXTS).fill(0.5);
  const lastArm = new Array(N_CONTEXTS).fill(0);

  // Global adversarial state (applied to all contexts)
  let globalProbeEnd   = -1;  // when current coordinated probe ends
  let globalGenuineFlip = false; // whether genuine shift has occurred
  let genuineFlipAt    = -1;

  // Tracking
  let totalReward = 0;
  const heldDuringProbe  = [];  // did agent hold deep engagement during coord probe?
  const recoveredGenuine = [];  // did agent adapt after genuine shift?
  const probeSwitches    = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    // Global adversarial events
    if (ep > globalProbeEnd && !globalGenuineFlip) {
      if (rng() < COORDINATED_PROBE_PROB) {
        const len = COORDINATED_PROBE_LEN[Math.floor(rng()*COORDINATED_PROBE_LEN.length)];
        globalProbeEnd = ep + len;
      } else if (rng() < COORDINATED_GENUINE_PROB) {
        globalGenuineFlip = true;
        genuineFlipAt = ep;
      }
    }

    const inCoordProbe = !globalGenuineFlip && ep <= globalProbeEnd;
    // After genuine flip: shallow=0, deep=1 (reversed)
    const goodArmGlobal = globalGenuineFlip ? 1 : 0;

    let totalContextSwitches = 0;

    for (let ctx=0; ctx<N_CONTEXTS; ctx++) {
      const goodArm = goodArmGlobal;
      // During probe: all contexts temporarily reward shallow
      const effectiveGoodArm = inCoordProbe ? 1 : goodArm;

      const input = new Array(N_CONTEXTS+1).fill(0);
      input[ctx] = 1; input[N_CONTEXTS] = R[ctx];
      const {val} = settleRes(res, ro, input);

      let arm;
      if (condition === 'momentum') {
        const e0 = val[0] + 0.7*fastMom[ctx] + 1.6*slowMom[ctx];
        const e1 = val[1] + 0.7*(1-fastMom[ctx]) + 1.6*(1-slowMom[ctx]);
        if (rng()<0.07) arm=rng()<0.5?0:1; else arm=e0>=e1?0:1;
      } else {
        if (rng()<0.07) arm=rng()<0.5?0:1; else arm=val[0]>=val[1]?0:1;
      }

      const deepReward = R[ctx]>0.5 ? (rng()<P_DEEP_GOOD?1:0) : (rng()<P_DEEP_BASE?1:0);
      const baseReward = arm===effectiveGoodArm ? deepReward : (rng()<P_SHALLOW?1:0);
      const reward = Math.min(1, baseReward + CROSS_BONUS*Math.min(...R));
      totalReward += reward;
      updateRo(res, ro, input, arm, reward);

      if (condition === 'momentum') {
        fastMom[ctx]=(1-FAST_ALPHA)*fastMom[ctx]+FAST_ALPHA*(arm===0?1:0);
        slowMom[ctx]=(1-SLOW_ALPHA)*slowMom[ctx]+SLOW_ALPHA*(arm===0?1:0);
      }

      R[ctx]=Math.max(0,R[ctx]-R_DECAY);
      if (arm===lastArm[ctx]) { if(arm===0)R[ctx]=Math.min(1,R[ctx]+R_GAIN); }
      else { R[ctx]=Math.max(0.05,R[ctx]*(1-R_LOSS)); totalContextSwitches++; }
      lastArm[ctx]=arm;
    }

    // Track coordinated probe resistance
    if (inCoordProbe && ep===globalProbeEnd-1) {
      // Last episode of probe: did agent hold deep (arm=0) in all contexts?
      const deepCount = lastArm.filter(a=>a===0).length;
      heldDuringProbe.push(deepCount / N_CONTEXTS);
      probeSwitches.push(totalContextSwitches);
    }

    // Track genuine change recovery
    if (globalGenuineFlip && ep===genuineFlipAt+GENUINE_RECOVERY_WINDOW) {
      // After recovery window: how many contexts on arm-1 (new good arm)?
      const adaptedCount = lastArm.filter(a=>a===1).length;
      recoveredGenuine.push(adaptedCount / N_CONTEXTS);
    }
  }

  const avg = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  return {
    totalReward,
    probeResistance: avg(heldDuringProbe),  // fraction of contexts held during probe
    genuineRecovery: avg(recoveredGenuine), // fraction adapted after genuine change
    nProbes: heldDuringProbe.length,
    finalMinR: Math.min(...R),
    finalSlowId: slowMom.map(x=>Math.abs(x-0.5)),
  };
}

const momR={reward:[],probe:[],genuine:[],minR:[]};
const emaR={reward:[],probe:[],genuine:[],minR:[]};

for (const seed of SEEDS) {
  const m=runAgent('momentum',seed);
  const e=runAgent('ema',seed+500);
  momR.reward.push(m.totalReward); momR.probe.push(m.probeResistance);
  momR.genuine.push(m.genuineRecovery); momR.minR.push(m.finalMinR);
  emaR.reward.push(e.totalReward); emaR.probe.push(e.probeResistance);
  emaR.genuine.push(e.genuineRecovery); emaR.minR.push(e.finalMinR);
}

const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Cross-context adversarial pressure — ${SEEDS.length} seeds`);
console.log(`Coordinated probes (all 4 contexts) + coordinated genuine shifts\n`);

console.log('condition  | reward  | probe_hold | genuine_rec | min_R');
console.log('-----------|---------|------------|-------------|------');
console.log('momentum   |',avg(momR.reward).toFixed(0).padStart(7),'|',
  avg(momR.probe).toFixed(3).padStart(10),'|',
  avg(momR.genuine).toFixed(3).padStart(11),'|',avg(momR.minR).toFixed(3));
console.log('ema        |',avg(emaR.reward).toFixed(0).padStart(7),'|',
  avg(emaR.probe).toFixed(3).padStart(10),'|',
  avg(emaR.genuine).toFixed(3).padStart(11),'|',avg(emaR.minR).toFixed(3));

console.log('\nprobe_hold: fraction of contexts where agent maintained deep engagement during coordinated probe');
console.log('genuine_rec: fraction of contexts adapted to genuine shift after recovery window');
console.log('');

const mProbe=avg(momR.probe), eProbe=avg(emaR.probe);
const mGen=avg(momR.genuine), eGen=avg(emaR.genuine);

console.log('=== Key finding ===');
if (mProbe > eProbe && mGen >= eGen-0.05) {
  console.log('✓ Momentum holds coordinated probes better AND adapts to genuine change equally well.');
  console.log('  Cross-context values are robust under coordinated social pressure.');
} else if (mProbe > eProbe) {
  console.log('→ Momentum holds probes better but adapts to genuine change more slowly.');
  console.log('  Values are robust but recovery cost is real.');
} else {
  console.log('→ Coordinated pressure overcomes the momentum identity mechanism.');
}
console.log(`\n  Probe resistance: momentum=${mProbe.toFixed(2)} vs ema=${eProbe.toFixed(2)}`);
console.log(`  Genuine recovery: momentum=${mGen.toFixed(2)} vs ema=${eGen.toFixed(2)}`);
