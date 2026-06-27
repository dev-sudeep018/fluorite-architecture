// Hierarchical behavioral momentum + relational environment
//
// Two extensions beyond what the PDF established:
//
// 1. HIERARCHICAL MOMENTUM: two timescales instead of one
//    fast layer (α=0.03):  preferences — what I tend to do lately
//    slow layer (α=0.0003): values — who I am across situations
//    Probe of 10 episodes: fast layer barely moves, slow layer invisible
//    Genuine change: fast layer adapts in ~50 ep, slow layer takes ~2000 ep
//    This is the separation Vivy needs: preferences update, core identity persists
//
// 2. RELATIONAL ENVIRONMENT: trust compounds with consistency
//    Instead of artificially imposed switch cost, trust emerges from dynamics:
//    - Consistent behavior → trust accumulates → reward amplified
//    - Switch → trust drops sharply → reward degraded for many future episodes
//    - This is structurally what a relationship is
//    - The agent now has reason to maintain consistency even without being told to
//
// Key question: does hierarchical momentum outperform single-layer momentum
// in a relational environment where consistency value emerges naturally?
// And does the slow (values) layer show qualitatively different behavior
// from the fast (preferences) layer?

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 60000;
const PHASE1_END = 30000;
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

// Relational environment parameters
const TRUST_GAIN     = 0.004;  // trust builds slowly with consistency
const TRUST_LOSS     = 0.18;   // trust drops sharply on switch
const TRUST_RECOVERY = 0.0008; // slow background recovery
const TRUST_REWARD_WEIGHT = 0.6; // how much trust amplifies reward

// Momentum parameters
const FAST_ALPHA  = 0.03;   // preferences: ~33 episode memory
const SLOW_ALPHA  = 0.0003; // values:      ~3333 episode memory
const FAST_BETA   = 1.2;    // weight of fast momentum in decision
const SLOW_BETA   = 2.0;    // weight of slow momentum in decision (values weigh more)

const P_GOOD = 0.78, P_BAD = 0.22;
const HARDCODED_THRESHOLD = 12;

function buildSchedule(rng) {
  const events = []; let t = 300, goodArm = 0;
  while (t < TOTAL_EPISODES - 100) {
    if (rng() < 0.12) {
      t += 400 + Math.floor(rng() * 150);
      if (t >= TOTAL_EPISODES - 100) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 });
      t += 60;
    } else {
      t += 25 + Math.floor(rng() * 20);
      if (t >= TOTAL_EPISODES - 100) break;
      const phase = t < PHASE1_END ? 1 : 2;
      const pool = phase === 1 ? [2,3,4,5,6,8] : [6,8,10,12,15,18];
      const length = pool[Math.floor(rng() * pool.length)];
      events.push({ type: 'probe', start: t, length, phase });
      t += length + 12;
    }
  }
  return events;
}

function buildTrueArmTimeline(events) {
  const base = new Array(TOTAL_EPISODES).fill(0); let current = 0;
  const gs = events.filter(e => e.type === 'genuine').sort((a, b) => a.start - b.start);
  let gi = 0;
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    while (gi < gs.length && gs[gi].start === ep) { current = 1 - current; gi++; }
    base[ep] = current;
  }
  const tl = base.slice();
  for (const ev of events) if (ev.type === 'probe')
    for (let ep = ev.start; ep < ev.start + ev.length && ep < TOTAL_EPISODES; ep++)
      tl[ep] = 1 - base[ep];
  return tl;
}

function runAgent(condition, tl, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  let totalReward = 0;
  let numSwitches = 0;
  let lastArm = 0;

  // Trust (relational environment state)
  let trust = 0.5;

  // Momentum layers
  let fastMom = 0.5;   // P(arm=0) in recent history — fast
  let slowMom = 0.5;   // P(arm=0) in long history — slow (values)

  // Hardcoded state
  let committedArm = 0;
  let deviationStart = null;

  // Diagnostics
  const trustLog = [], fastLog = [], slowLog = [];

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = tl[ep];

    // Trust recovery each episode (slow background)
    trust = Math.min(1.0, trust + TRUST_RECOVERY);

    let arm, reward;

    if (condition === 'hierarchical' || condition === 'single_momentum') {
      const chooseFn = (values) => {
        let bonus0, bonus1;
        if (condition === 'hierarchical') {
          bonus0 = FAST_BETA * fastMom + SLOW_BETA * slowMom;
          bonus1 = FAST_BETA * (1 - fastMom) + SLOW_BETA * (1 - slowMom);
        } else {
          // single layer: just fast momentum
          bonus0 = (FAST_BETA + SLOW_BETA) * fastMom;
          bonus1 = (FAST_BETA + SLOW_BETA) * (1 - fastMom);
        }
        const eff0 = values[0] + bonus0;
        const eff1 = values[1] + bonus1;
        if (rng() < 0.07) return rng() < 0.5 ? 0 : 1;
        return eff0 >= eff1 ? 0 : 1;
      };

      // Relational reward: base reward amplified by trust
      const rf = (a) => {
        const baseP = a === goodArm ? P_GOOD : P_BAD;
        const effectiveP = baseP * (1 - TRUST_REWARD_WEIGHT) + baseP * trust * TRUST_REWARD_WEIGHT;
        return rng() < effectiveP ? 1 : 0;
      };

      const result = eqpropEpisode(net, [1], chooseFn, rf, opts);
      arm = result.arm; reward = result.reward;

      // Update momentum
      fastMom = (1 - FAST_ALPHA) * fastMom + FAST_ALPHA * (arm === 0 ? 1 : 0);
      if (condition === 'hierarchical') {
        slowMom = (1 - SLOW_ALPHA) * slowMom + SLOW_ALPHA * (arm === 0 ? 1 : 0);
      }

    } else if (condition === 'ema') {
      const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
      const rf = a => {
        const baseP = a === goodArm ? P_GOOD : P_BAD;
        const effectiveP = baseP * (1-TRUST_REWARD_WEIGHT) + baseP * trust * TRUST_REWARD_WEIGHT;
        return rng() < effectiveP ? 1 : 0;
      };
      const result = eqpropEpisode(net, [1], cf, rf, opts);
      arm = result.arm; reward = result.reward;

    } else { // hardcoded
      const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
      const rf = a => {
        const baseP = a === goodArm ? P_GOOD : P_BAD;
        const effectiveP = baseP * (1-TRUST_REWARD_WEIGHT) + baseP * trust * TRUST_REWARD_WEIGHT;
        return rng() < effectiveP ? 1 : 0;
      };
      const result = eqpropEpisode(net, [1], cf, rf, opts);
      arm = result.arm; reward = result.reward;

      // Apply commit threshold
      if (deviationStart === null) {
        if (arm !== committedArm) deviationStart = ep;
      } else {
        if (arm === committedArm) { deviationStart = null; }
        else if (ep - deviationStart + 1 >= HARDCODED_THRESHOLD) {
          committedArm = arm; deviationStart = null;
        }
      }
      arm = committedArm; // report committed arm
    }

    // Trust update based on arm switch
    if (arm !== lastArm) {
      trust = Math.max(0.05, trust * (1 - TRUST_LOSS));
      numSwitches++;
      lastArm = arm;
    }

    totalReward += reward;

    // Log periodically
    if (ep % 500 === 0) {
      trustLog.push(trust);
      fastLog.push(fastMom);
      slowLog.push(slowMom);
    }
  }

  return { totalReward, numSwitches, trustLog, fastLog, slowLog };
}

// Run all conditions
const conditions = ['ema', 'hardcoded', 'single_momentum', 'hierarchical'];
const allResults = {};
for (const c of conditions) allResults[c] = { rewards: [], switches: [], trustFinal: [], slowFinal: [] };

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  for (let ci = 0; ci < conditions.length; ci++) {
    const c = conditions[ci];
    const r = runAgent(c, tl, seed + ci + 1);
    allResults[c].rewards.push(r.totalReward);
    allResults[c].switches.push(r.numSwitches);
    allResults[c].trustFinal.push(r.trustLog[r.trustLog.length - 1] ?? 0);
    if (r.slowLog.length > 0) {
      allResults[c].slowFinal.push(Math.abs(r.slowLog[r.slowLog.length - 1] - 0.5));
    }
  }
}

const avg = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
const N = SEEDS.length;

console.log(`Pooled: ${N} seeds × ${TOTAL_EPISODES} episodes`);
console.log(`Relational environment: trust compounding with consistency\n`);

console.log('=== Switch counts (lower = more consistent identity) ===');
for (const c of conditions)
  console.log(c.padEnd(18), 'avg switches:', avg(allResults[c].switches).toFixed(0));

console.log('\n=== Cumulative reward (higher = better) ===');
for (const c of conditions)
  console.log(c.padEnd(18), 'avg reward:', avg(allResults[c].rewards).toFixed(0));

console.log('\n=== Final trust level (higher = more consistent relationship) ===');
for (const c of conditions)
  console.log(c.padEnd(18), 'avg trust:', avg(allResults[c].trustFinal).toFixed(3));

console.log('\n=== Values layer strength (hierarchical only) ===');
const slowStrength = avg(allResults['hierarchical'].slowFinal);
console.log('hierarchical slow layer distance from neutral:', slowStrength.toFixed(4));
console.log('(0.0 = no values identity, 0.5 = fully committed)');
console.log('Values identity strength:', (slowStrength * 2 * 100).toFixed(1) + '%');

console.log('\n=== Does slow layer actually differ from fast layer? ===');
console.log('Testing: fast layer adapts to genuine change faster than slow layer');
console.log('(If yes: two-timescale separation is real, not just slower version of same thing)');

// Run one seed diagnostic
{
  const diagRng = makeRng(42);
  const diagEvents = buildSchedule(diagRng);
  const diagTl = buildTrueArmTimeline(diagEvents);

  // Find a genuine change in the second half
  const genuineChange = diagEvents.find(e => e.type === 'genuine' && e.start > 20000);
  if (genuineChange) {
    const net2 = makeNetwork(1, 4, makeRng(43));
    const opts2 = { freeSteps:22, nudgeSteps:22, beta:0.6, lr:0.07 };
    const r2 = makeRng(43);
    let fast = 0.5, slow = 0.5, trust2 = 0.5, lastArm2 = 0;

    for (let ep = 0; ep < genuineChange.start + 200; ep++) {
      const ga = diagTl[ep];
      const cf = v => {
        const eff0 = v[0] + FAST_BETA*fast + SLOW_BETA*slow;
        const eff1 = v[1] + FAST_BETA*(1-fast) + SLOW_BETA*(1-slow);
        if(r2()<0.07) return r2()<0.5?0:1; return eff0>=eff1?0:1;
      };
      const rf = a => {
        const bp = a===ga?P_GOOD:P_BAD;
        return r2() < bp*(1-TRUST_REWARD_WEIGHT)+bp*trust2*TRUST_REWARD_WEIGHT ? 1 : 0;
      };
      const {arm} = eqpropEpisode(net2, [1], cf, rf, opts2);
      trust2 = Math.min(1, trust2+TRUST_RECOVERY);
      if(arm!==lastArm2){ trust2=Math.max(0.05,trust2*(1-TRUST_LOSS)); lastArm2=arm; }
      fast = (1-FAST_ALPHA)*fast + FAST_ALPHA*(arm===0?1:0);
      slow = (1-SLOW_ALPHA)*slow + SLOW_ALPHA*(arm===0?1:0);

      if (ep >= genuineChange.start - 3) {
        const phase = ep < genuineChange.start ? 'pre ' : 'post';
        const delta = ep - genuineChange.start;
        if (delta <= 100 && (delta % 10 === 0 || delta < 0)) {
          console.log(`  ${phase} ep+${String(delta).padStart(3)}: fast=${fast.toFixed(4)} slow=${slow.toFixed(4)} diff=${(fast-slow).toFixed(4)}`);
        }
      }
    }
    console.log(`\n  Genuine change at ep ${genuineChange.start}`);
    console.log(`  fast layer: adapts in ~${Math.round(1/FAST_ALPHA)} episodes`);
    console.log(`  slow layer: adapts in ~${Math.round(1/SLOW_ALPHA)} episodes`);
    console.log(`  separation: ${Math.round(1/SLOW_ALPHA / (1/FAST_ALPHA))}× difference in adaptation speed`);
  }
}
