// Gated consolidation
//
// The catastrophic forgetting test showed the core tradeoff:
//   fast momentum: learns new tasks, loses old ones (forgetting)
//   slow momentum: preserves old tasks, can't learn new ones (rigidity)
//   hierarchical: preserves perfectly, Task B accuracy 0.036
//
// The missing mechanism: slow momentum should only update when we have
// CONFIRMED a genuine change — not during ordinary task-B experience.
// The commit threshold already detects genuine changes vs probes.
// Gate the slow layer update on the commit threshold signal:
//   - During Task B: commit threshold detects genuine change, opens gate
//   - Slow momentum now updates, gradually accepting the new reality
//   - During probes: gate stays closed, slow momentum protected
//
// Biologically: this is memory consolidation. Hippocampus (fast) captures
// new experience. After confirmation of significance, neocortex (slow)
// consolidates it. Sleep/replay bridges the two timescales.
// Here: commit threshold = hippocampal confirmation,
//       gated slow update = neocortical consolidation.
//
// Prediction: gated consolidation achieves BOTH good retention AND Task B learning.

const { makeRng, makeNetwork, eqpropEpisode } = require('./eqprop_core.js');

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

const FAST_ALPHA = 0.03;
const SLOW_ALPHA_GATED = 0.01; // FASTER when gate is open — consolidation is active
const COMMIT_THRESHOLD = 12;

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({length:RESERVOIR_SIZE}, () => [(rng()*2-1)*0.6]);
  const W_r  = Array.from({length:RESERVOIR_SIZE}, () =>
    Array.from({length:RESERVOIR_SIZE}, () => rng()<0.2 ? (rng()*2-1)*0.9 : 0)
  );
  let maxAbs = 0;
  for (const row of W_r) for (const v of row) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  const s = maxAbs > 0 ? 0.9/maxAbs : 1;
  for (const row of W_r) for (let j=0; j<row.length; j++) row[j]*=s;
  return { W_in, W_r };
}

function makeReadout(rng) {
  return {
    W: Array.from({length:2}, () => Array.from({length:RESERVOIR_SIZE}, () => (rng()*2-1)*0.1)),
    b: [0, 0],
  };
}

function settle(res, readout, x) {
  let s = new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0; t<SETTLE_STEPS; t++) {
    const n = new Array(RESERVOIR_SIZE).fill(0);
    for (let i=0; i<RESERVOIR_SIZE; i++) {
      let v = res.W_in[i][0]*x;
      for (let j=0; j<RESERVOIR_SIZE; j++) v += res.W_r[i][j]*s[j];
      n[i] = Math.tanh(v);
    }
    s = n;
  }
  const val = [0, 0];
  for (let i=0; i<2; i++) {
    val[i] = readout.b[i];
    for (let j=0; j<RESERVOIR_SIZE; j++) val[i] += readout.W[i][j]*s[j];
    val[i] = Math.tanh(val[i]);
  }
  return { s, val };
}

function updateReadout(res, readout, x, arm, reward) {
  const { s, val } = settle(res, readout, x);
  const dv = [0, 0];
  dv[arm] = BETA * ((2*reward-1) - val[arm]);
  for (let i=0; i<2; i++) {
    for (let j=0; j<RESERVOIR_SIZE; j++) readout.W[i][j] = clip(readout.W[i][j] + LR_READOUT*dv[i]*s[j]/BETA);
    readout.b[i] = clip(readout.b[i] + LR_READOUT*dv[i]/BETA);
  }
}

function runAgent(condition, seed) {
  const rng = makeRng(seed);
  const res = condition !== 'standard' ? makeReservoir(makeRng(seed+1000)) : null;
  const readout = condition !== 'standard' ? makeReadout(makeRng(seed+2000)) : null;
  const stdNet = condition === 'standard' ? makeNetwork(1, 6, makeRng(seed+3000)) : null;
  const stdOpts = { freeSteps:SETTLE_STEPS, nudgeSteps:SETTLE_STEPS, beta:BETA, lr:0.07 };

  let fastMom = 0.5, slowMom = 0.5;
  let committedArm = 0, deviationStart = null;
  let gateOpen = false, gateOpenCount = 0;

  const taskACorrect = [], taskBCorrect = [], retentionCorrect = [];
  const gateLog = [];

  for (let ep=0; ep<TOTAL; ep++) {
    const phase = ep < PHASE1_EP ? 'A' : ep < PHASE1_EP+PHASE2_EP ? 'B' : 'test';
    const goodArm = phase === 'B' ? 1 : 0;

    let arm;
    if (condition === 'standard') {
      const cf = v => { if(rng()<0.07) return rng()<0.5?0:1; return v[0]>=v[1]?0:1; };
      const rf = a => rng()<(a===goodArm?P_GOOD:P_BAD)?1:0;
      arm = eqpropEpisode(stdNet, [1], cf, rf, stdOpts).arm;
    } else {
      const { val } = settle(res, readout, 1);

      let bonus0, bonus1;
      if (condition === 'hierarchical_ungated') {
        bonus0 = 0.8*fastMom + 1.8*slowMom;
        bonus1 = 0.8*(1-fastMom) + 1.8*(1-slowMom);
      } else { // gated_consolidation
        // Fast momentum always active
        // Slow momentum only active when gate is CLOSED (protecting stable identity)
        // When gate is open (genuine change confirmed), slow momentum updating — reduce its influence so fast can lead
        const slowInfluence = gateOpen ? 0.3 : 1.8; // reduce slow influence during consolidation
        bonus0 = 0.8*fastMom + slowInfluence*slowMom;
        bonus1 = 0.8*(1-fastMom) + slowInfluence*(1-slowMom);
      }

      if (rng()<0.07) arm = rng()<0.5?0:1;
      else arm = (val[0]+bonus0) >= (val[1]+bonus1) ? 0 : 1;

      if (phase !== 'test') {
        const reward = rng()<(arm===goodArm?P_GOOD:P_BAD)?1:0;
        updateReadout(res, readout, 1, arm, reward);
      }
    }

    // Commit threshold logic
    if (arm !== committedArm) {
      if (deviationStart === null) deviationStart = ep;
      else if (ep - deviationStart + 1 >= COMMIT_THRESHOLD) {
        // GENUINE CHANGE CONFIRMED — open consolidation gate
        committedArm = arm;
        deviationStart = null;
        if (condition === 'gated_consolidation') {
          gateOpen = true;
          gateOpenCount = 0;
        }
      }
    } else {
      deviationStart = null;
    }

    // Gate management: close gate after consolidation period
    if (gateOpen) {
      gateOpenCount++;
      if (gateOpenCount > 500) { // consolidation window: 500 episodes
        gateOpen = false;
      }
    }

    // Momentum updates
    if (condition !== 'standard') {
      fastMom = (1-FAST_ALPHA)*fastMom + FAST_ALPHA*(arm===0?1:0);

      if (condition === 'gated_consolidation') {
        if (gateOpen) {
          // Consolidation window: genuine change confirmed, update fast toward new reality
          slowMom = (1-SLOW_ALPHA_GATED)*slowMom + SLOW_ALPHA_GATED*(arm===0?1:0);
        } else {
          // Stable period: update very slowly — this is where identity BUILDS
          const SLOW_ALPHA_STABLE = 0.0001; // slow but non-zero during stable periods
          slowMom = (1-SLOW_ALPHA_STABLE)*slowMom + SLOW_ALPHA_STABLE*(arm===0?1:0);
        }
      } else {
        // hierarchical ungated: slow always updates
        const SLOW_ALPHA_UNGATED = 0.0003;
        slowMom = (1-SLOW_ALPHA_UNGATED)*slowMom + SLOW_ALPHA_UNGATED*(arm===0?1:0);
      }
    }

    if (ep % 2000 === 0) gateLog.push({ ep, gateOpen, fast: fastMom, slow: slowMom });

    const correct = arm === goodArm ? 1 : 0;
    if (phase === 'A')    taskACorrect.push(correct);
    else if (phase==='B') taskBCorrect.push(correct);
    else                  retentionCorrect.push(correct);
  }

  const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  return {
    taskA:     avg(taskACorrect.slice(-2000)),
    taskB:     avg(taskBCorrect.slice(-2000)),
    retention: avg(retentionCorrect),
    slowFinal: slowMom,
    gateLog,
  };
}

const conditions = ['standard', 'hierarchical_ungated', 'gated_consolidation'];
const res = {};
for (const c of conditions) res[c] = { taskA:[], taskB:[], ret:[], slow:[] };

for (const seed of SEEDS) {
  for (const c of conditions) {
    const r = runAgent(c, seed);
    res[c].taskA.push(r.taskA);
    res[c].taskB.push(r.taskB);
    res[c].ret.push(r.retention);
    res[c].slow.push(r.slowFinal);
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Gated consolidation — ${SEEDS.length} seeds`);
console.log(`Gate opens for 500 episodes when commit threshold (N=${COMMIT_THRESHOLD}) confirms genuine change\n`);

console.log('condition              | task-A | task-B | RETENTION | slow_final');
console.log('-----------------------|--------|--------|-----------|----------');
for (const c of conditions) {
  const r = res[c];
  console.log(
    c.padEnd(23),'|',
    avg(r.taskA).toFixed(3).padStart(6),'|',
    avg(r.taskB).toFixed(3).padStart(6),'|',
    avg(r.ret).toFixed(3).padStart(9),'|',
    avg(r.slow).toFixed(3).padStart(9)
  );
}

console.log('\n=== Verdict ===');
for (const c of conditions) {
  const r = res[c];
  const ret = avg(r.ret), b = avg(r.taskB);
  const v = ret>0.70&&b>0.70 ? '✓ BOTH preserved'
    : ret>0.70 ? '→ retention preserved, task-B lost (rigid)'
    : b>0.70   ? '→ task-B learned, task-A forgotten'
    : '→ partial on both';
  console.log(c.padEnd(23),':', v, `(ret=${ret.toFixed(2)}, B=${b.toFixed(2)})`);
}

console.log('\n=== Gate opening log (gated_consolidation, seed 42) ===');
const diagR = runAgent('gated_consolidation', 42);
for (const entry of diagR.gateLog) {
  const phase = entry.ep<PHASE1_EP?'A':entry.ep<PHASE1_EP+PHASE2_EP?'B':'T';
  const gateStr = entry.gateOpen ? '[GATE OPEN]' : '           ';
  console.log(`  ep${String(entry.ep).padStart(5)} [${phase}] ${gateStr} fast=${entry.fast.toFixed(3)} slow=${entry.slow.toFixed(3)}`);
}
