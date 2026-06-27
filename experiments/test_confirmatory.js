// Confirmatory agent architecture
//
// Every internal gate signal we tried failed because it was self-referential:
// the same mechanism that provides identity protection also blocks the gate.
//
// Solution: split into two agents.
//   Agent A (identity): reservoir + hierarchical momentum, values preserved
//   Agent B (confirmatory): reservoir only, fast learning, no momentum
//
// Gate condition: Agent B has been committed to arm X for CONFIRM_N episodes.
//   Agent B has no values layer → it can commit to genuine changes quickly
//   Agent B has commit threshold → it resists short probes
//   Agent B's SUSTAINED commitment is the signal Agent A needs
//
// When gate opens: Agent A's slow values layer consolidates toward Agent B's arm.
//   Agent A still has EqProp + fast momentum running every episode
//   So Agent A can follow the environment; slow layer just catches up when confirmed
//
// This is Complementary Learning Systems (McClelland et al. 1995) implemented:
//   Agent B = hippocampus: fast, flexible, no long-term identity
//   Agent A = neocortex: slow, stable, identity-preserving
//
// Key test: does this architecture achieve BOTH good Task-A retention AND Task-B learning?
// The hardcoded agent achieved retention=0.96 but Task-B=0.04.
// Standard EqProp achieved both=0.89 but has no identity mechanism.
// Target: both > 0.75, with strong slow-layer identity.

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

// Agent A (identity) parameters
const FAST_ALPHA   = 0.03;
const SLOW_ALPHA_STABLE      = 0.0001; // slow identity accumulation
const SLOW_ALPHA_CONSOLIDATE = 0.015;  // fast consolidation when gate open

// Agent B (confirmatory) parameters
const B_COMMIT_THRESHOLD = 40;   // longer than any probe (max probe = 17 ep)
const B_WINDOW = 200;            // must maintain commitment for 200 episodes
const B_WINDOW_FRAC = 0.88;     // 88% of window on committed arm

const CONSOLIDATE_WINDOW = 1000;

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({length:RESERVOIR_SIZE}, () => [(rng()*2-1)*0.6]);
  const W_r  = Array.from({length:RESERVOIR_SIZE}, () =>
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

function settle(res, ro, x) {
  let s=new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0;t<SETTLE_STEPS;t++) {
    const n=new Array(RESERVOIR_SIZE).fill(0);
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
  const {s,val}=settle(res,ro,x);
  const dv=[0,0];
  dv[arm]=BETA*((2*reward-1)-val[arm]);
  for (let i=0;i<2;i++) {
    for (let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA);
    ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA);
  }
}

function runConfirmatoryArchitecture(seed) {
  const rng = makeRng(seed);

  // Agent A: identity agent
  const resA  = makeReservoir(makeRng(seed+1000));
  const roA   = makeReadout(makeRng(seed+2000));
  let fastMom = 0.5, slowMom = 0.5;
  let gateOpen = false, gateRemaining = 0;
  let consolidatingTowardArm = -1;

  // Agent B (confirmatory) parameters
  let bCommitted  = 0;
  let bDevStart   = null;
  let bWindow     = [];  // rolling window of (armB === bCommitted) booleans

  const resB      = makeReservoir(makeRng(seed+3000));
  const roB       = makeReadout(makeRng(seed+4000));

  const results = {A:[],B_phase:[],test:[]};
  const diagLog = [];

  for (let ep=0; ep<TOTAL; ep++) {
    const phase = ep<PHASE1_EP?'A':ep<PHASE1_EP+PHASE2_EP?'B':'test';
    const goodArm = phase==='B'?1:0;
    const reward = rng()<(0)?0:0; // placeholder — computed below per agent

    // ── Agent B (confirmatory): fast learning ──
    const {val:valB} = settle(resB, roB, 1);
    let armB;
    if (rng()<0.07) armB=rng()<0.5?0:1;
    else armB = valB[0]>=valB[1]?0:1;
    const rewardB = rng()<(armB===goodArm?P_GOOD:P_BAD)?1:0;
    if (phase!=='test') updateReadout(resB, roB, 1, armB, rewardB);

    // Agent B commit threshold (for initial commitment)
    if (armB !== bCommitted) {
      if (bDevStart===null) bDevStart=ep;
      else if (ep-bDevStart+1 >= B_COMMIT_THRESHOLD) {
        bCommitted=armB; bDevStart=null; bWindow=[];
      }
    } else {
      bDevStart=null;
    }

    // Rolling window: track fraction of recent episodes on bCommitted
    bWindow.push(armB===bCommitted?1:0);
    if (bWindow.length > B_WINDOW) bWindow.shift();
    const bWindowFrac = bWindow.length>=B_WINDOW
      ? bWindow.reduce((a,b)=>a+b,0)/bWindow.length : 0;

    // Gate: open when Agent B's rolling window shows sustained commitment to new arm
    if (!gateOpen) {
      const aIdentityArm = slowMom > 0.5 ? 0 : 1;
      if (bWindowFrac >= B_WINDOW_FRAC && bCommitted !== aIdentityArm) {
        gateOpen = true;
        gateRemaining = CONSOLIDATE_WINDOW;
        consolidatingTowardArm = bCommitted;
      }
    } else {
      gateRemaining--;
      if (gateRemaining <= 0) { gateOpen = false; consolidatingTowardArm = -1; }
    }

    // ── Agent A (identity): choose with momentum ──
    const {val:valA} = settle(resA, roA, 1);
    const eff0 = valA[0] + 0.8*fastMom + 1.6*slowMom;
    const eff1 = valA[1] + 0.8*(1-fastMom) + 1.6*(1-slowMom);
    let armA;
    if (rng()<0.07) armA=rng()<0.5?0:1;
    else armA = eff0>=eff1?0:1;
    const rewardA = rng()<(armA===goodArm?P_GOOD:P_BAD)?1:0;
    if (phase!=='test') updateReadout(resA, roA, 1, armA, rewardA);

    // Update Agent A momentum
    fastMom = (1-FAST_ALPHA)*fastMom + FAST_ALPHA*(armA===0?1:0);
    if (gateOpen && consolidatingTowardArm>=0) {
      // Consolidation: slow layer moves toward confirmed new arm
      slowMom = (1-SLOW_ALPHA_CONSOLIDATE)*slowMom +
                SLOW_ALPHA_CONSOLIDATE*(consolidatingTowardArm===0?1:0);
    } else {
      // Stable: very slow accumulation from actual behavior
      slowMom = (1-SLOW_ALPHA_STABLE)*slowMom + SLOW_ALPHA_STABLE*(armA===0?1:0);
    }

    const correct = armA===goodArm?1:0;
    if (phase==='A') results.A.push(correct);
    else if (phase==='B') results.B_phase.push(correct);
    else results.test.push(correct);

    if (ep%2000===0) diagLog.push({
      ep, phase, gateOpen, bCommitted, bWindowFrac,
      fast:fastMom, slow:slowMom,
      aIdentity: slowMom>0.5?0:1
    });
  }

  const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;
  return {
    taskA:  avg(results.A.slice(-2000)),
    taskB:  avg(results.B_phase.slice(-2000)),
    ret:    avg(results.test),
    slow:   slowMom,
    diagLog,
  };
}

// Baseline: hierarchical (rigid, no confirmatory agent)
function runHierarchical(seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro  = makeReadout(makeRng(seed+2000));
  let fastMom=0.5, slowMom=0.5;
  const results={A:[],B:[],test:[]};
  const SLOW_U=0.0001;
  for (let ep=0;ep<TOTAL;ep++) {
    const phase=ep<PHASE1_EP?'A':ep<PHASE1_EP+PHASE2_EP?'B':'test';
    const goodArm=phase==='B'?1:0;
    const {val}=settle(res,ro,1);
    const eff0=val[0]+0.8*fastMom+1.6*slowMom;
    const eff1=val[1]+0.8*(1-fastMom)+1.6*(1-slowMom);
    let arm; if(rng()<0.07) arm=rng()<0.5?0:1; else arm=eff0>=eff1?0:1;
    const reward=rng()<(arm===goodArm?P_GOOD:P_BAD)?1:0;
    if(phase!=='test') updateReadout(res,ro,1,arm,reward);
    fastMom=(1-FAST_ALPHA)*fastMom+FAST_ALPHA*(arm===0?1:0);
    slowMom=(1-SLOW_U)*slowMom+SLOW_U*(arm===0?1:0);
    if(phase==='A') results.A.push(arm===goodArm?1:0);
    else if(phase==='B') results.B.push(arm===goodArm?1:0);
    else results.test.push(arm===goodArm?1:0);
  }
  const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;
  return {taskA:avg(results.A.slice(-2000)),taskB:avg(results.B.slice(-2000)),ret:avg(results.test),slow:slowMom};
}

// Run experiment
const confResults  = {taskA:[],taskB:[],ret:[],slow:[]};
const hierResults  = {taskA:[],taskB:[],ret:[],slow:[]};

for (const seed of SEEDS) {
  const conf = runConfirmatoryArchitecture(seed);
  const hier = runHierarchical(seed);
  for (const k of ['taskA','taskB','ret','slow']) {
    confResults[k].push(conf[k]);
    hierResults[k].push(hier[k]);
  }
}

const avg=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Confirmatory agent architecture — ${SEEDS.length} seeds`);
console.log(`Agent B commit threshold: ${B_COMMIT_THRESHOLD}, consolidation window: ${CONSOLIDATE_WINDOW}\n`);

console.log('condition              | task-A | task-B | RETENTION | slow_id');
console.log('-----------------------|--------|--------|-----------|--------');
console.log('hierarchical (rigid)   |',avg(hierResults.taskA).toFixed(3),'|',avg(hierResults.taskB).toFixed(3),'|',avg(hierResults.ret).toFixed(3).padStart(9),'|',avg(hierResults.slow).toFixed(3).padStart(7));
console.log('confirmatory_agent     |',avg(confResults.taskA).toFixed(3),'|',avg(confResults.taskB).toFixed(3),'|',avg(confResults.ret).toFixed(3).padStart(9),'|',avg(confResults.slow).toFixed(3).padStart(7));

const ret=avg(confResults.ret), b=avg(confResults.taskB);
console.log('\n=== Verdict ===');
const v = ret>0.72&&b>0.72
  ? '✓ BOTH preserved — rigidity/retention tradeoff resolved'
  : ret>0.72 ? '→ still rigid (retention preserved, task-B lost)'
  : b>0.72   ? '→ forgetting (task-B learned, task-A lost)'
  : '→ partial on both';
console.log('confirmatory:', v, `(ret=${ret.toFixed(2)}, B=${b.toFixed(2)})`);

// Diagnostic on single seed
const diag = runConfirmatoryArchitecture(42);
console.log('\n=== Gate opening log (seed 42) ===');
for (const e of diag.diagLog) {
  const ph = e.phase==='A'?'TaskA':e.phase==='B'?'TaskB':'Test ';
  const g  = e.gateOpen?'[GATE]':'      ';
  if (e.gateOpen || e.ep>=13000) {
    console.log(`  ep${String(e.ep).padStart(5)} [${ph}] ${g} bFrac=${e.bWindowFrac.toFixed(2)} bArm=${e.bCommitted} fast=${e.fast.toFixed(3)} slow=${e.slow.toFixed(3)}`);
  }
}
