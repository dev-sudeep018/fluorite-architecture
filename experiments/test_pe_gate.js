// Prediction-error-gated consolidation
//
// The rigidity problem from last experiment:
//   Slow momentum strong enough to protect against probes
//   is also strong enough to prevent genuine changes from
//   ever accumulating 12 consecutive inconsistent choices.
//   The gate condition can't be reached because the gate's
//   OWN signal (slow momentum) blocks it.
//
// The solution: use a DIFFERENT information stream.
//   Current system uses: which arm was chosen (behavioral)
//   Unused signal: prediction error (how surprised was I by the reward?)
//
//   During stable Task A:
//     Agent chooses arm-0 (consistent with identity)
//     Arm-0 rewards well → prediction error LOW
//
//   During probe (arm-1 temporarily good):
//     Agent still mostly chooses arm-0 (identity holds)
//     Arm-0 stops rewarding → prediction error HIGH
//     ...but probe ends → arm-0 rewards again → error DROPS
//
//   During genuine change (arm-1 permanently good):
//     Agent still mostly chooses arm-0 (identity holds)
//     Arm-0 keeps failing → prediction error STAYS HIGH
//     DURATION of high prediction error distinguishes probe from genuine
//
// Gate condition: prediction error EMA > threshold for > N episodes
//   → confirmed genuine change
//   → open slow layer for consolidation window
//   This uses information orthogonal to momentum, so can't be self-blocked
//
// This is the Complementary Learning Systems insight implemented:
//   Fast layer (EqProp + reservoir): learns from everything
//   Prediction error detector: identifies genuine environmental change
//   Slow layer (values): only updates when genuine change confirmed

const { makeRng } = require('./eqprop_core.js');

const PHASE1_EP = 15000;
const PHASE2_EP = 15000;
const TEST_EP   = 2000;
const TOTAL     = PHASE1_EP + PHASE2_EP + TEST_EP;
const RESERVOIR_SIZE = 12;
const SETTLE_STEPS   = 10;
const LR_READOUT     = 0.08;
const BETA           = 0.6;
const P_GOOD = 0.78, P_BAD = 0.22;
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999, 12345, 54321];

// Momentum
const FAST_ALPHA = 0.03;
const SLOW_ALPHA_STABLE = 0.0001;  // very slow identity accumulation
const SLOW_ALPHA_CONSOLIDATE = 0.015; // fast consolidation during confirmed change

// Prediction error gate parameters — RELATIVE, not absolute
const PE_BASELINE_ALPHA = 0.002;  // very slow baseline tracker
const PE_FAST_ALPHA     = 0.08;   // fast tracker of current PE
const PE_DEVIATION_THRESH = 0.12; // deviation above baseline → suspect change
const PE_CONFIRM_N  = 60;
const CONSOLIDATE_WINDOW = 600;

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({length:RESERVOIR_SIZE}, () => [(rng()*2-1)*0.6]);
  const W_r  = Array.from({length:RESERVOIR_SIZE}, () =>
    Array.from({length:RESERVOIR_SIZE}, () => rng()<0.2?(rng()*2-1)*0.9:0)
  );
  let mx=0;
  for (const r of W_r) for (const v of r) if(Math.abs(v)>mx) mx=Math.abs(v);
  const s = mx>0?0.9/mx:1;
  for (const r of W_r) for (let j=0;j<r.length;j++) r[j]*=s;
  return {W_in,W_r};
}

function makeReadout(rng) {
  return {
    W: Array.from({length:2},()=>Array.from({length:RESERVOIR_SIZE},()=>(rng()*2-1)*0.1)),
    b: [0,0]
  };
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
  for (let i=0;i<2;i++) {
    val[i]=ro.b[i];
    for (let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j];
    val[i]=Math.tanh(val[i]);
  }
  return {s,val};
}

function updateReadout(res, ro, x, arm, reward) {
  const {s,val} = settle(res,ro,x);
  const dv=[0,0];
  dv[arm]=BETA*((2*reward-1)-val[arm]);
  for (let i=0;i<2;i++) {
    for (let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA);
    ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA);
  }
  return Math.abs((2*reward-1)-val[arm]); // return raw prediction error
}

function runAgent(condition, seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro  = makeReadout(makeRng(seed+2000));

  let fastMom = 0.5, slowMom = 0.5;
  let peBaseline = 0.35;  // running baseline of normal PE level
  let peFast = 0.35;      // fast tracker
  let peSustainedN = 0;
  let gateOpen = false;
  let gateRemaining = 0;

  const results = {A:[],B:[],test:[]};
  const diagLog = [];

  for (let ep=0;ep<TOTAL;ep++) {
    const phase = ep<PHASE1_EP?'A':ep<PHASE1_EP+PHASE2_EP?'B':'test';
    const goodArm = phase==='B'?1:0;

    // Get arm values from reservoir
    const {val} = settle(res,ro,1);

    // Apply momentum
    const eff0 = val[0] + 0.8*fastMom + 1.5*slowMom;
    const eff1 = val[1] + 0.8*(1-fastMom) + 1.5*(1-slowMom);

    let arm;
    if (rng()<0.07) arm=rng()<0.5?0:1;
    else arm = eff0>=eff1?0:1;

    const reward = rng()<(arm===goodArm?P_GOOD:P_BAD)?1:0;

    // Update readout and get prediction error
    let pe = 0;
    if (phase!=='test') {
      pe = updateReadout(res,ro,1,arm,reward);
    } else {
      // During test: compute PE without updating
      const {val:v2}=settle(res,ro,1);
      pe = Math.abs((2*reward-1)-v2[arm]);
    }

    // Update PE tracking on identity-consistent choices only
    const identityArm = slowMom > 0.5 ? 0 : 1;
    if (arm === identityArm) {
      peFast     = (1-PE_FAST_ALPHA)*peFast + PE_FAST_ALPHA*pe;
      // Baseline only updates when gate is closed and deviation is small
      // (so genuine changes don't corrupt the baseline)
      if (!gateOpen && (peFast - peBaseline) < PE_DEVIATION_THRESH) {
        peBaseline = (1-PE_BASELINE_ALPHA)*peBaseline + PE_BASELINE_ALPHA*pe;
      }
    }

    const deviation = peFast - peBaseline;

    // Gate on sustained DEVIATION above baseline
    if (!gateOpen) {
      if (deviation > PE_DEVIATION_THRESH) {
        peSustainedN++;
        if (peSustainedN >= PE_CONFIRM_N) {
          gateOpen = true;
          gateRemaining = CONSOLIDATE_WINDOW;
          peSustainedN = 0;
        }
      } else {
        peSustainedN = Math.max(0, peSustainedN - 1); // decay on non-deviant ep
      }
    } else {
      gateRemaining--;
      if (gateRemaining <= 0) {
        gateOpen = false;
      }
    }

    // Update fast momentum always
    fastMom = (1-FAST_ALPHA)*fastMom + FAST_ALPHA*(arm===0?1:0);

    // Update slow momentum:
    //   Stable (gate closed): very slow accumulation
    //   Consolidating (gate open): fast integration of new reality
    if (gateOpen) {
      slowMom = (1-SLOW_ALPHA_CONSOLIDATE)*slowMom + SLOW_ALPHA_CONSOLIDATE*(arm===0?1:0);
    } else {
      slowMom = (1-SLOW_ALPHA_STABLE)*slowMom + SLOW_ALPHA_STABLE*(arm===0?1:0);
    }

    const correct = arm===goodArm?1:0;
    results[phase]=results[phase]||[];
    if (phase==='A') results.A.push(correct);
    else if (phase==='B') results.B.push(correct);
    else results.test.push(correct);

    if (ep%1000===0) diagLog.push({ep,phase,peFast,peBaseline,dev:peFast-peBaseline,gateOpen,fast:fastMom,slow:slowMom,peSustainedN});
  }

  const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;
  return {
    taskA:  avg(results.A.slice(-2000)),
    taskB:  avg(results.B.slice(-2000)),
    ret:    avg(results.test),
    slow:   slowMom,
    diagLog,
  };
}

// Also run hierarchical ungated as baseline for comparison
function runHierarchical(seed) {
  const { makeNetwork, eqpropEpisode } = require('./eqprop_core.js');
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro  = makeReadout(makeRng(seed+2000));
  let fastMom=0.5, slowMom=0.5;
  const results={A:[],B:[],test:[]};
  for (let ep=0;ep<TOTAL;ep++) {
    const phase=ep<PHASE1_EP?'A':ep<PHASE1_EP+PHASE2_EP?'B':'test';
    const goodArm=phase==='B'?1:0;
    const {val}=settle(res,ro,1);
    const eff0=val[0]+0.8*fastMom+1.5*slowMom;
    const eff1=val[1]+0.8*(1-fastMom)+1.5*(1-slowMom);
    let arm;
    if(rng()<0.07) arm=rng()<0.5?0:1;
    else arm=eff0>=eff1?0:1;
    const reward=rng()<(arm===goodArm?P_GOOD:P_BAD)?1:0;
    if(phase!=='test') updateReadout(res,ro,1,arm,reward);
    fastMom=(1-FAST_ALPHA)*fastMom+FAST_ALPHA*(arm===0?1:0);
    const SLOW_U=0.0001;
    slowMom=(1-SLOW_U)*slowMom+SLOW_U*(arm===0?1:0);
    if(phase==='A') results.A.push(arm===goodArm?1:0);
    else if(phase==='B') results.B.push(arm===goodArm?1:0);
    else results.test.push(arm===goodArm?1:0);
  }
  const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;
  return {taskA:avg(results.A.slice(-2000)),taskB:avg(results.B.slice(-2000)),ret:avg(results.test),slow:slowMom};
}

const peResults  = {taskA:[],taskB:[],ret:[],slow:[]};
const hierResults = {taskA:[],taskB:[],ret:[],slow:[]};

for (const seed of SEEDS) {
  const pe   = runAgent('pe_gated', seed);
  const hier = runHierarchical(seed);
  for (const k of ['taskA','taskB','ret','slow']) {
    peResults[k].push(pe[k]);
    hierResults[k].push(hier[k]);
  }
}

const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Prediction-error gated consolidation — ${SEEDS.length} seeds`);
console.log(`PE threshold=${PE_DEVIATION_THRESH}, confirm N=${PE_CONFIRM_N}, window=${CONSOLIDATE_WINDOW}\n`);

console.log('condition              | task-A | task-B | RETENTION | slow_final');
console.log('-----------------------|--------|--------|-----------|----------');
console.log('hierarchical (baseline)|',avg(hierResults.taskA).toFixed(3),'|',avg(hierResults.taskB).toFixed(3),'|',avg(hierResults.ret).toFixed(3).padStart(9),'|',avg(hierResults.slow).toFixed(3).padStart(9));
console.log('pe_gated_consolidation |',avg(peResults.taskA).toFixed(3),'|',avg(peResults.taskB).toFixed(3),'|',avg(peResults.ret).toFixed(3).padStart(9),'|',avg(peResults.slow).toFixed(3).padStart(9));

console.log('\n=== Verdict ===');
const ret=avg(peResults.ret), b=avg(peResults.taskB);
const v = ret>0.70&&b>0.70?'✓ BOTH preserved — rigidity tradeoff resolved'
  :ret>0.70?'→ rigid (retention preserved, task-B lost)'
  :b>0.70?'→ forgetting (task-B learned, task-A lost)'
  :'→ partial on both';
console.log('pe_gated:', v, `(ret=${ret.toFixed(2)}, B=${b.toFixed(2)})`);

console.log('\n=== Gate opening diagnostic (seed 42) ===');
const diag = runAgent('pe_gated', 42);
let gatesOpened=0;
for (const entry of diag.diagLog) {
  if (entry.gateOpen && (!diag.diagLog[diag.diagLog.indexOf(entry)-1]?.gateOpen)) gatesOpened++;
  const phase=entry.phase==='A'?'TaskA':entry.phase==='B'?'TaskB':'Test ';
  const gStr = entry.gateOpen?'[OPEN]':'      ';
  if (entry.ep>=13000 || entry.gateOpen || (entry.dev !== undefined && entry.dev > 0.05)) {
    const devStr = entry.dev !== undefined ? ' dev='+entry.dev.toFixed(3)+' base='+entry.peBaseline.toFixed(3) : '';
    console.log(`  ep${String(entry.ep).padStart(5)} [${phase}] ${gStr}${devStr} fast=${entry.fast.toFixed(3)} slow=${entry.slow.toFixed(3)}`);
  }
}
console.log(`Total gate openings: ${gatesOpened}`);
// patched diagnostic print
