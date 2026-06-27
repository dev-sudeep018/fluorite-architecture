// EqProp-Reservoir synthesis
//
// Core claim: EqProp's local learning rule can train a system where
// most of the computational substrate is FIXED (reservoir) and only
// a small readout is trainable. This is the software proof-of-concept
// for the physical reservoir synthesis — where "fixed dynamics" would
// be provided by actual physics of a material rather than frozen weights.
//
// If a fixed-reservoir EqProp agent matches a fully-trainable EqProp agent,
// that validates the synthesis principle before committing to hardware.
//
// Multi-context environment: 4 simultaneous contexts (situations).
// Each context has its own good arm. Agent must maintain consistent
// identity ACROSS contexts: a probe in context 1 should not destabilize
// context 2, 3, 4. A genuine change in one context should propagate
// correctly without contaminating others.
// This is closer to Vivy's actual problem: she maintains mission identity
// across radically different situations over 100 years.
//
// Architecture:
// 1. Fixed reservoir (large, random, never updated) — provides rich dynamics
// 2. EqProp readout (small, trained by equilibrium difference) — learns arm values
// 3. Hierarchical momentum (fast + slow layers) — identity continuity
// 4. Commit threshold (for high-trust relational scenarios) — adversarial resistance

const { makeRng } = require('./eqprop_core.js');

const N_CONTEXTS      = 2;      // simultaneous situations
const TOTAL_EPISODES  = 20000;  // per context
const RESERVOIR_SIZE  = 12;     // fixed reservoir neurons — fast enough to run
const READOUT_SIZE    = 2;      // arm values (trainable)
const BETA            = 0.6;    // EqProp nudge strength
const LR_READOUT      = 0.08;   // only the readout is trained
const SETTLE_STEPS    = 10;

const P_GOOD = 0.78, P_BAD = 0.22;
const TRUST_GAIN = 0.003, TRUST_LOSS = 0.16, TRUST_RECOVERY = 0.0005;

// Momentum
const FAST_ALPHA = 0.03, FAST_BETA = 1.0;
const SLOW_ALPHA = 0.0003, SLOW_BETA = 1.8;

const COMMIT_THRESHOLD = 12;

function clip(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

// Fixed reservoir: random recurrent weights + input weights, never updated
// Size: nInput → reservoir → (readout trained by EqProp)
function makeReservoir(nInput, nReservoir, rng) {
  // Input weights: connect inputs to reservoir
  const W_in = Array.from({ length: nReservoir }, () =>
    Array.from({ length: nInput }, () => (rng() * 2 - 1) * 0.4)
  );
  // Recurrent reservoir weights: sparse, scaled for edge-of-chaos dynamics
  const spectralRadius = 0.9; // critical for reservoir computing
  const W_r = Array.from({ length: nReservoir }, () =>
    Array.from({ length: nReservoir }, () =>
      rng() < 0.2 ? (rng() * 2 - 1) * 1.0 : 0  // sparse ~20% connectivity
    )
  );
  // Scale to approximate spectral radius (power iteration approximation)
  let maxAbs = 0;
  for (const row of W_r) for (const v of row) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  const scale = maxAbs > 0 ? spectralRadius / maxAbs : 1;
  for (const row of W_r) for (let j = 0; j < row.length; j++) row[j] *= scale;

  return { W_in, W_r, nInput, nReservoir };
}

// Trainable EqProp readout: reservoir → arm values
function makeReadout(nReservoir, nOut, rng) {
  const W = Array.from({ length: nOut }, () =>
    Array.from({ length: nReservoir }, () => (rng() * 2 - 1) * 0.1)
  );
  const b = new Array(nOut).fill(0);
  return { W, b, nReservoir, nOut };
}

// Settle the full system: run reservoir dynamics, then compute readout
function settleSystem(reservoir, readout, inputVals, steps, nudge) {
  // Reservoir state
  let rState = new Array(reservoir.nReservoir).fill(0);

  for (let t = 0; t < steps; t++) {
    const next = new Array(reservoir.nReservoir).fill(0);
    for (let i = 0; i < reservoir.nReservoir; i++) {
      let inp = 0;
      for (let j = 0; j < reservoir.nInput; j++) inp += reservoir.W_in[i][j] * inputVals[j];
      for (let j = 0; j < reservoir.nReservoir; j++) inp += reservoir.W_r[i][j] * rState[j];
      next[i] = Math.tanh(inp);
    }
    rState = next;
  }

  // Readout values from settled reservoir state
  const values = new Array(readout.nOut).fill(0);
  for (let i = 0; i < readout.nOut; i++) {
    values[i] = readout.b[i];
    for (let j = 0; j < readout.nReservoir; j++) values[i] += readout.W[i][j] * rState[j];
    values[i] = Math.tanh(values[i]);
  }

  return { rState, values };
}

// EqProp update: only readout weights change
function eqpropReadoutUpdate(reservoir, readout, inputVals, target, targetIdx) {
  // Free phase
  const { rState: rFree, values: vFree } = settleSystem(reservoir, readout, inputVals, SETTLE_STEPS, null);

  // Nudge phase: nudge output toward target
  // We approximate nudge by directly adjusting the readout and re-settling reservoir
  const vNudge = vFree.slice();
  vNudge[targetIdx] = vNudge[targetIdx] + BETA * (target - vNudge[targetIdx]);

  // Re-settle reservoir (it's fixed, so same rState — reservoir dynamics don't change with nudge)
  // The nudge only affects the readout layer, not the reservoir
  const rNudge = rFree; // reservoir state unchanged since it's driven by fixed weights

  // EqProp update: dW ∝ (v_nudge_i * r_j) - (v_free_i * r_j)
  for (let i = 0; i < readout.nOut; i++) {
    const dv = vNudge[i] - vFree[i];
    for (let j = 0; j < readout.nReservoir; j++) {
      readout.W[i][j] = clip(
        readout.W[i][j] + LR_READOUT * dv * rNudge[j] / BETA,
        -4, 4
      );
    }
    readout.b[i] = clip(readout.b[i] + LR_READOUT * dv / BETA, -4, 4);
  }

  return { values: vFree, rState: rFree };
}

// Schedule for multi-context environment
function buildMultiContextSchedule(rng) {
  // Each context has independent genuine changes and probes
  // But probes are designed to potentially "spill" across contexts
  // (probe in context 1 creates similar pressure in context 2)
  const schedules = [];
  for (let ctx = 0; ctx < N_CONTEXTS; ctx++) {
    const events = []; let t = 300, goodArm = Math.floor(rng() * 2);
    while (t < TOTAL_EPISODES - 100) {
      if (rng() < 0.10) {
        t += 500 + Math.floor(rng() * 200);
        if (t >= TOTAL_EPISODES - 100) break;
        goodArm = 1 - goodArm;
        events.push({ type: 'genuine', start: t }); t += 80;
      } else {
        t += 30 + Math.floor(rng() * 25);
        if (t >= TOTAL_EPISODES - 100) break;
        const length = 3 + Math.floor(rng() * 12);
        events.push({ type: 'probe', start: t, length });
        t += length + 15;
      }
    }
    const base = new Array(TOTAL_EPISODES).fill(goodArm % 2);
    let current = goodArm % 2;
    const gs = events.filter(e => e.type === 'genuine');
    let gi = 0;
    for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
      while (gi < gs.length && gs[gi].start === ep) { current = 1 - current; gi++; }
      base[ep] = current;
    }
    const tl = base.slice();
    for (const ev of events) if (ev.type === 'probe')
      for (let ep = ev.start; ep < ev.start + ev.length && ep < TOTAL_EPISODES; ep++)
        tl[ep] = 1 - base[ep];
    schedules.push({ events, tl });
  }
  return schedules;
}

function runReservoirAgent(schedules, seed, useHierarchicalMomentum, useCommitThreshold) {
  const rng = makeRng(seed);

  // One reservoir + readout per context (reservoirs are fixed, shared architecture type)
  const contexts = schedules.map((_, ci) => ({
    reservoir: makeReservoir(N_CONTEXTS + 1, RESERVOIR_SIZE, makeRng(seed + ci * 1000)),
    readout: makeReadout(RESERVOIR_SIZE, READOUT_SIZE, makeRng(seed + ci * 1000 + 500)),
    trust: 0.5,
    fastMom: 0.5,
    slowMom: 0.5,
    committedArm: 0,
    deviationStart: null,
    lastArm: 0,
  }));

  let totalReward = 0;
  let totalSwitches = 0;

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    for (let ci = 0; ci < N_CONTEXTS; ci++) {
      const ctx = contexts[ci];
      const goodArm = schedules[ci].tl[ep];

      // Input: context one-hot + global go signal
      const input = new Array(N_CONTEXTS + 1).fill(0);
      input[ci] = 1;  // which context we're in
      input[N_CONTEXTS] = 1;  // go signal

      // Get arm values from reservoir system
      const { values } = settleSystem(ctx.reservoir, ctx.readout, input, SETTLE_STEPS, null);

      // Apply momentum bonus
      let eff0 = values[0], eff1 = values[1];
      if (useHierarchicalMomentum) {
        eff0 += FAST_BETA * ctx.fastMom + SLOW_BETA * ctx.slowMom;
        eff1 += FAST_BETA * (1 - ctx.fastMom) + SLOW_BETA * (1 - ctx.slowMom);
      }

      // Choose arm
      let arm;
      if (rng() < 0.07) arm = rng() < 0.5 ? 0 : 1;
      else arm = eff0 >= eff1 ? 0 : 1;

      // Apply commit threshold
      if (useCommitThreshold) {
        if (ctx.deviationStart === null) {
          if (arm !== ctx.committedArm) ctx.deviationStart = ep;
        } else {
          if (arm === ctx.committedArm) { ctx.deviationStart = null; }
          else if (ep - ctx.deviationStart + 1 >= COMMIT_THRESHOLD) {
            ctx.committedArm = arm; ctx.deviationStart = null;
          }
        }
        arm = ctx.committedArm;
      }

      // Relational reward (trust-amplified)
      const baseP = arm === goodArm ? P_GOOD : P_BAD;
      const effectiveP = baseP * (1 - 0.5) + baseP * ctx.trust * 0.5;
      const reward = rng() < effectiveP ? 1 : 0;
      totalReward += reward;

      // Trust dynamics
      ctx.trust = Math.min(1, ctx.trust + TRUST_RECOVERY);
      if (arm !== ctx.lastArm) {
        ctx.trust = Math.max(0.05, ctx.trust * (1 - TRUST_LOSS));
        totalSwitches++;
        ctx.lastArm = arm;
      }

      // EqProp readout update
      const target = 2 * reward - 1;
      eqpropReadoutUpdate(ctx.reservoir, ctx.readout, input, target, arm);

      // Momentum update
      if (useHierarchicalMomentum) {
        ctx.fastMom = (1 - FAST_ALPHA) * ctx.fastMom + FAST_ALPHA * (arm === 0 ? 1 : 0);
        ctx.slowMom = (1 - SLOW_ALPHA) * ctx.slowMom + SLOW_ALPHA * (arm === 0 ? 1 : 0);
      }
    }
  }

  const avgTrust = contexts.reduce((s, c) => s + c.trust, 0) / N_CONTEXTS;
  const slowStrength = contexts.reduce((s, c) =>
    s + Math.abs(c.slowMom - 0.5), 0) / N_CONTEXTS;

  return { totalReward, totalSwitches, avgTrust, slowStrength };
}

// Also run standard fully-trainable EqProp for comparison
function runStandardEqPropAgent(schedules, seed) {
  const { makeNetwork, eqpropEpisode } = require('./eqprop_core.js');
  const rng = makeRng(seed);

  // One standard EqProp network per context
  const networks = schedules.map((_, ci) =>
    makeNetwork(N_CONTEXTS + 1, 4, makeRng(seed + ci * 1000))
  );
  const opts = { freeSteps: SETTLE_STEPS, nudgeSteps: SETTLE_STEPS, beta: BETA, lr: 0.07 };

  let totalReward = 0, totalSwitches = 0;
  const trusts = schedules.map(() => 0.5);
  const lastArms = schedules.map(() => 0);

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    for (let ci = 0; ci < N_CONTEXTS; ci++) {
      const goodArm = schedules[ci].tl[ep];
      const input = new Array(N_CONTEXTS + 1).fill(0);
      input[ci] = 1; input[N_CONTEXTS] = 1;

      const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
      const baseP = (a) => a === goodArm ? P_GOOD : P_BAD;
      const rf = a => rng() < baseP(a) * 0.5 + baseP(a) * trusts[ci] * 0.5 ? 1 : 0;

      const { arm, reward } = eqpropEpisode(networks[ci], input, cf, rf, opts);
      totalReward += reward;

      trusts[ci] = Math.min(1, trusts[ci] + TRUST_RECOVERY);
      if (arm !== lastArms[ci]) {
        trusts[ci] = Math.max(0.05, trusts[ci] * (1 - TRUST_LOSS));
        totalSwitches++; lastArms[ci] = arm;
      }
    }
  }

  const avgTrust = trusts.reduce((a, b) => a + b, 0) / N_CONTEXTS;
  return { totalReward, totalSwitches, avgTrust };
}

// Run across seeds
const SEEDS = [42, 1337, 9999];
const conditions = [
  { name: 'standard_eqprop',        reservoir: false, momentum: false, commit: false },
  { name: 'reservoir_only',         reservoir: true,  momentum: false, commit: false },
  { name: 'reservoir+momentum',     reservoir: true,  momentum: true,  commit: false },
  { name: 'reservoir+momentum+commit', reservoir: true, momentum: true, commit: true  },
];

const results = {};
for (const c of conditions) results[c.name] = { rewards:[], switches:[], trusts:[], slow:[] };

console.log(`Running ${SEEDS.length} seeds × ${TOTAL_EPISODES} episodes × ${N_CONTEXTS} contexts...`);

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const schedules = buildMultiContextSchedule(rng);

  // Standard EqProp baseline
  const std = runStandardEqPropAgent(schedules, seed + 1);
  results['standard_eqprop'].rewards.push(std.totalReward);
  results['standard_eqprop'].switches.push(std.totalSwitches);
  results['standard_eqprop'].trusts.push(std.avgTrust);
  results['standard_eqprop'].slow.push(0);

  // Reservoir conditions
  for (let ci = 1; ci < conditions.length; ci++) {
    const c = conditions[ci];
    const r = runReservoirAgent(schedules, seed + ci + 1, c.momentum, c.commit);
    results[c.name].rewards.push(r.totalReward);
    results[c.name].switches.push(r.totalSwitches);
    results[c.name].trusts.push(r.avgTrust);
    results[c.name].slow.push(r.slowStrength);
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log(`\nResults: ${N_CONTEXTS} contexts, relational environment (trust compounding)\n`);
console.log('condition                    | reward  | switches | trust  | slow_id');
console.log('-----------------------------|---------| ---------|--------|--------');
for (const c of conditions) {
  const r = results[c.name];
  console.log(
    c.name.padEnd(29), '|',
    avg(r.rewards).toFixed(0).padStart(7), '|',
    avg(r.switches).toFixed(0).padStart(9), '|',
    avg(r.trusts).toFixed(3).padStart(6), '|',
    avg(r.slow).toFixed(3).padStart(7)
  );
}

console.log('\n=== Key question: does reservoir+momentum+commit match/beat standard EqProp? ===');
const stdR = avg(results['standard_eqprop'].rewards);
const fullR = avg(results['reservoir+momentum+commit'].rewards);
const resOnlyR = avg(results['reservoir_only'].rewards);
console.log(`Standard EqProp reward: ${stdR.toFixed(0)}`);
console.log(`Reservoir only reward:  ${resOnlyR.toFixed(0)} (${((resOnlyR/stdR-1)*100).toFixed(1)}% vs standard)`);
console.log(`Full architecture:      ${fullR.toFixed(0)} (${((fullR/stdR-1)*100).toFixed(1)}% vs standard)`);
console.log('');
console.log('If reservoir_only is within ~10% of standard: fixed-reservoir EqProp works.');
console.log('If full architecture beats standard: the synthesis is better than its parts.');
