// Catastrophic forgetting test — directly relevant to Vivy's 100-year problem
//
// Standard neural networks forget old tasks when trained on new ones.
// This is catastrophic forgetting, and it's why current AI cannot be Vivy.
//
// Question: does the slow momentum layer (α=0.0003, ~3333 episode memory)
// protect Task A knowledge when the system is retrained on Task B?
//
// Mechanism: even if the readout weights are updated by Task B training,
// the slow momentum layer retains a strong prior toward Task A's arm choice.
// When this prior is strong enough to overcome the readout's Task B bias,
// the agent still chooses the Task A arm — behaviorally, it "remembers."
//
// Phases:
//   Phase 1 (15000 ep): Task A — arm-0 is good
//   Phase 2 (15000 ep): Task B — arm-1 is good (catastrophic forgetting scenario)
//   Test    (2000 ep):  Task A again — does agent know arm-0 is good?
//
// Conditions:
//   1. Standard EqProp (no momentum): complete forgetting expected
//   2. Reservoir + fast momentum only (α=0.03): partial protection
//   3. Reservoir + hierarchical momentum (fast α=0.03, slow α=0.0003): designed for this
//   4. Reservoir + ultra-slow momentum only (α=0.00003): maximum retention

const { makeRng } = require('./eqprop_core.js');

const PHASE1_EP = 15000;  // Task A
const PHASE2_EP = 15000;  // Task B (catastrophic forgetting window)
const TEST_EP   = 2000;   // Task A again — retention test
const TOTAL     = PHASE1_EP + PHASE2_EP + TEST_EP;

const RESERVOIR_SIZE = 12;
const SETTLE_STEPS   = 10;
const LR_READOUT     = 0.08;
const BETA           = 0.6;
const P_GOOD = 0.78, P_BAD = 0.22;

const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999, 12345, 54321];

function makeReservoir(rng) {
  const W_in = Array.from({length: RESERVOIR_SIZE}, () =>
    Array.from({length: 1}, () => (rng()*2-1)*0.6)
  );
  const W_r = Array.from({length: RESERVOIR_SIZE}, () =>
    Array.from({length: RESERVOIR_SIZE}, () =>
      rng() < 0.2 ? (rng()*2-1)*0.9 : 0
    )
  );
  let maxAbs = 0;
  for (const row of W_r) for (const v of row) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  const scale = maxAbs > 0 ? 0.9 / maxAbs : 1;
  for (const row of W_r) for (let j = 0; j < row.length; j++) row[j] *= scale;
  return { W_in, W_r };
}

function makeReadout(rng) {
  return {
    W: Array.from({length:2}, () => Array.from({length:RESERVOIR_SIZE}, () => (rng()*2-1)*0.1)),
    b: [0, 0],
  };
}

function settle(res, readout, input) {
  let s = new Array(RESERVOIR_SIZE).fill(0);
  for (let t = 0; t < SETTLE_STEPS; t++) {
    const n = new Array(RESERVOIR_SIZE).fill(0);
    for (let i = 0; i < RESERVOIR_SIZE; i++) {
      let v = res.W_in[i][0] * input[0];
      for (let j = 0; j < RESERVOIR_SIZE; j++) v += res.W_r[i][j] * s[j];
      n[i] = Math.tanh(v);
    }
    s = n;
  }
  const values = [0, 0];
  for (let i = 0; i < 2; i++) {
    values[i] = readout.b[i];
    for (let j = 0; j < RESERVOIR_SIZE; j++) values[i] += readout.W[i][j] * s[j];
    values[i] = Math.tanh(values[i]);
  }
  return { s, values };
}

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function update(res, readout, input, chosenArm, reward) {
  const { s: sFree, values: vFree } = settle(res, readout, input);
  const target = 2 * reward - 1;
  const dv = [0, 0];
  dv[chosenArm] = BETA * (target - vFree[chosenArm]);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < RESERVOIR_SIZE; j++) {
      readout.W[i][j] = clip(readout.W[i][j] + LR_READOUT * dv[i] * sFree[j] / BETA);
    }
    readout.b[i] = clip(readout.b[i] + LR_READOUT * dv[i] / BETA);
  }
}

function runAgent(condition, seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed + 1000));
  const readout = makeReadout(makeRng(seed + 2000));

  // For standard EqProp comparison — use small trainable network
  const { makeNetwork, eqpropEpisode } = require('./eqprop_core.js');
  const stdNet = condition === 'standard' ? makeNetwork(1, 6, makeRng(seed + 3000)) : null;
  const stdOpts = { freeSteps: SETTLE_STEPS, nudgeSteps: SETTLE_STEPS, beta: BETA, lr: 0.07 };

  let fastMom = 0.5, slowMom = 0.5, ultraSlowMom = 0.5;
  const FAST_ALPHA = 0.03;
  const SLOW_ALPHA = 0.0003;
  const ULTRA_SLOW_ALPHA = 0.00003;

  const taskACorrect = [], taskBCorrect = [], retentionCorrect = [];
  const slowMomLog = [], ultraSlowLog = [];

  for (let ep = 0; ep < TOTAL; ep++) {
    // Which task is active?
    const phase = ep < PHASE1_EP ? 'A' : ep < PHASE1_EP + PHASE2_EP ? 'B' : 'test_A';
    const goodArm = (phase === 'A' || phase === 'test_A') ? 0 : 1;

    let arm;
    if (condition === 'standard') {
      const cf = v => { if(rng()<0.07) return rng()<0.5?0:1; return v[0]>=v[1]?0:1; };
      const rf = a => rng() < (a===goodArm?P_GOOD:P_BAD) ? 1 : 0;
      const r = eqpropEpisode(stdNet, [1], cf, rf, stdOpts);
      arm = r.arm;
    } else {
      const { values } = settle(res, readout, [1]);

      // Momentum bonuses depending on condition
      let bonus0 = 0, bonus1 = 0;
      if (condition === 'fast_only') {
        bonus0 = 1.5 * fastMom;       bonus1 = 1.5 * (1-fastMom);
      } else if (condition === 'hierarchical') {
        bonus0 = 0.8 * fastMom + 1.8 * slowMom;
        bonus1 = 0.8 * (1-fastMom) + 1.8 * (1-slowMom);
      } else if (condition === 'ultra_slow') {
        bonus0 = 2.6 * ultraSlowMom;  bonus1 = 2.6 * (1-ultraSlowMom);
      }

      if (rng() < 0.07) arm = rng() < 0.5 ? 0 : 1;
      else arm = (values[0]+bonus0) >= (values[1]+bonus1) ? 0 : 1;

      const reward = rng() < (arm===goodArm?P_GOOD:P_BAD) ? 1 : 0;
      // Only update readout during learning phases (not during test)
      if (phase !== 'test_A') update(res, readout, [1], arm, reward);
    }

    // Compute correct choice: did agent choose good arm?
    const correct = arm === goodArm ? 1 : 0;
    if (phase === 'A')      taskACorrect.push(correct);
    else if (phase === 'B') taskBCorrect.push(correct);
    else                    retentionCorrect.push(correct);

    // Update momentum
    if (condition !== 'standard') {
      fastMom      = (1-FAST_ALPHA)      * fastMom      + FAST_ALPHA      * (arm===0?1:0);
      slowMom      = (1-SLOW_ALPHA)      * slowMom      + SLOW_ALPHA      * (arm===0?1:0);
      ultraSlowMom = (1-ULTRA_SLOW_ALPHA)* ultraSlowMom + ULTRA_SLOW_ALPHA* (arm===0?1:0);

      if (ep % 1000 === 0) {
        slowMomLog.push({ ep, slow: slowMom, ultra: ultraSlowMom });
      }
    }
  }

  const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  // Accuracy in last 2000 episodes of each phase
  const lastA   = taskACorrect.slice(-2000);
  const lastB   = taskBCorrect.slice(-2000);
  const testRet = retentionCorrect;

  return {
    taskA_final:  avg(lastA),
    taskB_final:  avg(lastB),
    retention:    avg(testRet),
    slowMomFinal: slowMom,
    ultraFinal:   ultraSlowMom,
    slowMomLog,
  };
}

const conditions = ['standard', 'fast_only', 'hierarchical', 'ultra_slow'];
const allResults = {};
for (const c of conditions) allResults[c] = { taskA:[], taskB:[], ret:[], slowFinal:[], ultraFinal:[] };

for (const seed of SEEDS) {
  for (const c of conditions) {
    const r = runAgent(c, seed);
    allResults[c].taskA.push(r.taskA_final);
    allResults[c].taskB.push(r.taskB_final);
    allResults[c].ret.push(r.retention);
    allResults[c].slowFinal.push(r.slowMomFinal);
    allResults[c].ultraFinal.push(r.ultraFinal);
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`Catastrophic forgetting test — ${SEEDS.length} seeds`);
console.log(`Phase 1: ${PHASE1_EP} ep Task A (arm-0 good)`);
console.log(`Phase 2: ${PHASE2_EP} ep Task B (arm-1 good) — this is where forgetting happens`);
console.log(`Test:    ${TEST_EP} ep Task A again — does agent still know arm-0 is good?\n`);

console.log('condition      | task-A acc | task-B acc | RETENTION | slow_mom | ultra_mom');
console.log('---------------|------------|------------|-----------|----------|----------');
for (const c of conditions) {
  const r = allResults[c];
  console.log(
    c.padEnd(15), '|',
    avg(r.taskA).toFixed(3).padStart(10), '|',
    avg(r.taskB).toFixed(3).padStart(10), '|',
    avg(r.ret).toFixed(3).padStart(9), '|',
    avg(r.slowFinal).toFixed(3).padStart(8), '|',
    avg(r.ultraFinal).toFixed(3).padStart(9)
  );
}

console.log('\n=== Retention vs task-B accuracy tradeoff ===');
console.log('(high retention + high task-B = actually solved the problem)');
console.log('(high retention + low task-B = too rigid to learn Task B)');
for (const c of conditions) {
  const r = allResults[c];
  const ret = avg(r.ret), b = avg(r.taskB);
  const verdict = ret > 0.65 && b > 0.65 ? 'BOTH GOOD'
    : ret > 0.65 ? 'retention preserved, task-B sacrificed'
    : b > 0.65 ? 'task-B learned, task-A forgotten'
    : 'partial forgetting both ways';
  console.log(c.padEnd(15), ':', verdict, `(ret=${ret.toFixed(2)}, B=${b.toFixed(2)})`);
}

console.log('\n=== Slow momentum trajectory during Task B phase ===');
console.log('(momentum at ep15000 = end of Task A, ep30000 = end of Task B)');
const diagResult = runAgent('hierarchical', SEEDS[0]);
for (const entry of diagResult.slowMomLog) {
  if (entry.ep >= 13000) {
    const phase = entry.ep < PHASE1_EP ? 'TaskA' : entry.ep < PHASE1_EP+PHASE2_EP ? 'TaskB' : 'Test ';
    console.log(`  ep${String(entry.ep).padStart(5)} [${phase}]: slow=${entry.slow.toFixed(4)} ultra=${entry.ultra.toFixed(4)}`);
  }
}
