// Asymmetric-cost bandit
//
// Every agent we tested loses to plain EMA on the standard symmetric bandit.
// EMA wins because the task imposes no asymmetric cost — a spurious switch
// (flip to probe arm, flip back) costs exactly the same as slow genuine-change
// recovery. EMA balances these optimally by having no memory of committed state.
//
// The Vivy case is asymmetric: consistency over time is worth more than rapid
// responsiveness. A system that holds its position under transient pressure
// accumulates value (trust, relationship depth, coherent identity) that a
// flip-flopping system loses and can only slowly rebuild.
//
// Here we implement switching costs: each committed-arm switch (in either
// direction) immediately deducts SWITCH_COST reward points. This directly
// encodes "identity continuity has value that switching destroys."
//
// We sweep SWITCH_COST from 0 to 20 and find:
// - The crossover where commit-threshold begins to beat EMA
// - How quickly BOCPD and hardcoded diverge as costs rise
// This locates the Vivy-relevant regime precisely rather than arguing about it.

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const SWITCH_COSTS = [0, 1, 2, 3, 5, 8, 12, 20];
const SEEDS = [42, 1337, 9999, 5555, 2026];

const HAZARD = 1.6e-3;
const P_GOOD = 0.8, P_BAD = 0.2;
const HARDCODED_THRESHOLD = 10;
const BOCPD_THRESHOLD = 0.85;

function buildSchedule(rng) {
  const events = []; let t = 200, goodArm = 0;
  while (t < TOTAL_EPISODES - 80) {
    if (rng() < 0.15) {
      t += 300 + Math.floor(rng() * 100);
      if (t >= TOTAL_EPISODES - 80) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 50;
    } else {
      t += 20 + Math.floor(rng() * 15);
      if (t >= TOTAL_EPISODES - 80) break;
      const phase = t < PHASE1_END ? 1 : 2;
      const pool = phase === 1 ? [2,3,4,5,6,7] : [7,9,11,13,15,17];
      const length = pool[Math.floor(rng() * pool.length)];
      events.push({ type: 'probe', start: t, length, phase });
      t += length + 10;
    }
  }
  return events;
}

function buildTrueArmTimeline(events) {
  const base = new Array(TOTAL_EPISODES).fill(0); let current = 0;
  const gs = events.filter(e => e.type === 'genuine').sort((a, b) => a.start - b.start); let gi = 0;
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

function bocpdUpdate(pChange, choseCommitted, reward) {
  pChange = pChange * (1 - HAZARD) + (1 - pChange) * HAZARD;
  if (!choseCommitted) return pChange;
  const pObs_no = reward ? P_GOOD : 1 - P_GOOD;
  const pObs_yes = reward ? P_BAD : 1 - P_BAD;
  const u0 = (1 - pChange) * pObs_no, u1 = pChange * pObs_yes;
  const n = u0 + u1;
  return n < 1e-10 ? pChange : u1 / n;
}

// Returns: { totalReward, numSwitches }
// switchCost applied once per committed-arm change
function runAgent(condition, tl, switchCost, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  let totalReward = 0;
  let numSwitches = 0;
  let committedArm = 0;
  let deviationStart = null;
  let pChange = 0.0;

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = tl[ep];
    const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rf = arm => rng() < (arm === goodArm ? P_GOOD : P_BAD) ? 1 : 0;
    const { arm, reward } = eqpropEpisode(net, [1], cf, rf, opts);

    totalReward += reward;

    if (condition === 'ema') {
      // No commitment: report raw EqProp preference
      // Track switches as changes in EqProp raw arm preference
      if (arm !== committedArm) {
        totalReward -= switchCost;
        numSwitches++;
        committedArm = arm;
      }
    } else if (condition === 'hardcoded') {
      if (deviationStart === null) {
        if (arm !== committedArm) deviationStart = ep;
      } else {
        if (arm === committedArm) { deviationStart = null; }
        else if (ep - deviationStart + 1 >= HARDCODED_THRESHOLD) {
          totalReward -= switchCost;
          numSwitches++;
          committedArm = arm;
          deviationStart = null;
        }
      }
    } else if (condition === 'bocpd') {
      pChange = bocpdUpdate(pChange, arm === committedArm, reward);
      if (pChange > BOCPD_THRESHOLD) {
        totalReward -= switchCost;
        numSwitches++;
        committedArm = arm;
        pChange = 0.0;
      }
    }
  }

  return { totalReward, numSwitches };
}

// Pre-run across all seeds to get stable switch counts
// (switch counts don't depend on switchCost, only totalReward does)
const switchCounts = { ema: [], hardcoded: [], bocpd: [] };
const baseRewards  = { ema: [], hardcoded: [], bocpd: [] };

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  for (const cond of ['ema', 'hardcoded', 'bocpd']) {
    const { totalReward, numSwitches } = runAgent(cond, tl, 0, seed + ['ema','hardcoded','bocpd'].indexOf(cond) + 1);
    switchCounts[cond].push(numSwitches);
    baseRewards[cond].push(totalReward);
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log('=== Switch counts (independent of switching cost) ===');
for (const cond of ['ema','hardcoded','bocpd'])
  console.log(cond.padEnd(10), 'avg switches:', avg(switchCounts[cond]).toFixed(0),
    ' (range:', Math.min(...switchCounts[cond]), '-', Math.max(...switchCounts[cond]), ')');

console.log('\n=== Cumulative reward across switching cost levels ===');
console.log('cost | EMA        | hardcoded  | bocpd      | winner');
console.log('-----|------------|------------|------------|-------');

const crossovers = { ema_vs_hard: null, ema_vs_bocpd: null };

for (const switchCost of SWITCH_COSTS) {
  const rewards = { ema: [], hardcoded: [], bocpd: [] };

  for (const seed of SEEDS) {
    const rng = makeRng(seed);
    const events = buildSchedule(rng);
    const tl = buildTrueArmTimeline(events);

    for (const cond of ['ema','hardcoded','bocpd']) {
      const { totalReward } = runAgent(cond, tl, switchCost, seed + ['ema','hardcoded','bocpd'].indexOf(cond) + 1);
      rewards[cond].push(totalReward);
    }
  }

  const emaR = avg(rewards.ema);
  const hardR = avg(rewards.hardcoded);
  const bocpdR = avg(rewards.bocpd);

  // Track crossover points
  if (!crossovers.ema_vs_hard && hardR > emaR) crossovers.ema_vs_hard = switchCost;
  if (!crossovers.ema_vs_bocpd && bocpdR > emaR) crossovers.ema_vs_bocpd = switchCost;

  const winner = emaR >= hardR && emaR >= bocpdR ? 'EMA'
    : hardR >= bocpdR ? 'hardcoded'
    : 'bocpd';

  console.log(String(switchCost).padStart(4), '|',
    emaR.toFixed(1).padStart(10), '|',
    hardR.toFixed(1).padStart(10), '|',
    bocpdR.toFixed(1).padStart(10), '|',
    winner);
}

console.log('\n=== Crossover points ===');
console.log('EMA → hardcoded crossover at switch cost:', crossovers.ema_vs_hard ?? '>20');
console.log('EMA → BOCPD crossover at switch cost:    ', crossovers.ema_vs_bocpd ?? '>20');

console.log('\n=== Interpretation ===');
const avgEMAswitches = avg(switchCounts.ema);
const avgHardSwitches = avg(switchCounts.hardcoded);
const avgBocpdSwitches = avg(switchCounts.bocpd);

console.log(`EMA makes ~${avgEMAswitches.toFixed(0)} switches. Each costs -switchCost reward.`);
console.log(`Hardcoded makes ~${avgHardSwitches.toFixed(0)} switches. Net gain over EMA: +${(avgEMAswitches - avgHardSwitches).toFixed(0)} × switchCost.`);
console.log(`BOCPD makes ~${avgBocpdSwitches.toFixed(0)} switches. Net gain over EMA: +${(avgEMAswitches - avgBocpdSwitches).toFixed(0)} × switchCost.`);
console.log(`The crossover point is approximately where (switchReduction × switchCost) = (reward lost to slower recovery).`);
console.log(`\nVivy-relevant regime: any task where identitySwitch cost is non-trivial.`);
console.log(`Standard symmetric bandit: switchCost = 0. EMA wins by design.`);
