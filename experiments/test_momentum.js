// Behavioral momentum: endogenous identity
//
// Everything we built used exogenous rules (hardcoded threshold, P_revert table,
// BOCPD hazard rate). Vivy's identity isn't a rule imposed from outside.
// It's something that accumulated over 100 years of being consistently herself.
//
// Behavioral momentum: track EMA of choice history. Add it as a bonus to
// EqProp's arm-value estimates. High momentum toward arm A = agent has been
// consistently choosing A = agent has accumulated identity = harder to perturb.
//
// Key properties:
// - New agent: zero momentum, no identity, follows reward immediately (EMA-like)
// - Experienced agent: strong momentum, resists transient pressure
// - Resistance is earned, not imposed — emerges from consistent behavior
// - Genuine sustained change eventually erodes old momentum, builds new momentum
//
// Compare across: EMA, hardcoded threshold, behavioral momentum
// Test across switch costs [0, 1, 3, 8, 20]

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999, 12345, 54321];
const SWITCH_COSTS = [0, 1, 2, 3, 5, 8, 12, 20];
const P_GOOD = 0.8, P_BAD = 0.2;
const HARDCODED_THRESHOLD = 10;

// Momentum parameters — these set the timescale of identity
const MOMENTUM_ALPHA = 0.003;  // very slow EMA → long memory → hard-won identity
const MOMENTUM_BETA  = 2.0;    // how much momentum bonus weighs vs EqProp values

function buildSchedule(rng) {
  const events = []; let t = 200, goodArm = 0;
  while (t < TOTAL_EPISODES - 80) {
    if (rng() < 0.15) {
      t += 300 + Math.floor(rng() * 100);
      if (t >= TOTAL_EPISODES - 80) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t }); t += 50;
    } else {
      t += 20 + Math.floor(rng() * 15);
      if (t >= TOTAL_EPISODES - 80) break;
      const phase = t < PHASE1_END ? 1 : 2;
      const pool = phase === 1 ? [2,3,4,5,6,7] : [7,9,11,13,15,17];
      const length = pool[Math.floor(rng() * pool.length)];
      events.push({ type: 'probe', start: t, length });
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

function clip(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function runAgent(condition, tl, switchCost, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  let totalReward = 0;
  let numSwitches = 0;
  let reportedArm = 0;       // what we publicly commit to
  let deviationStart = null; // for hardcoded
  let pChange = 0.0;         // for BOCPD

  // Behavioral momentum: EMA of arm-0 choice frequency
  // momentum[0] ≈ P(agent chooses arm 0 in recent history)
  let momentumA = 0.5; // starts at 50/50 — no identity yet

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = tl[ep];

    let arm, reward;

    if (condition === 'momentum') {
      // EqProp values + momentum bonus
      // We need the raw values before commitment to add momentum
      // Run EqProp with modified effective values via the choose function
      const chooseWithMomentum = (values) => {
        const momentumBonus = [momentumA, 1 - momentumA];
        const effective = [
          values[0] + MOMENTUM_BETA * momentumBonus[0],
          values[1] + MOMENTUM_BETA * momentumBonus[1],
        ];
        if (rng() < 0.07) return rng() < 0.5 ? 0 : 1;
        return effective[0] >= effective[1] ? 0 : 1;
      };
      const rf = a => rng() < (a === goodArm ? P_GOOD : P_BAD) ? 1 : 0;
      const result = eqpropEpisode(net, [1], chooseWithMomentum, rf, opts);
      arm = result.arm; reward = result.reward;

      // Update momentum toward the arm just chosen
      momentumA = (1 - MOMENTUM_ALPHA) * momentumA + MOMENTUM_ALPHA * (arm === 0 ? 1 : 0);

      // Track switches in reported arm (for switch cost)
      if (arm !== reportedArm) {
        totalReward -= switchCost;
        numSwitches++;
        reportedArm = arm;
      }
    } else {
      const cf = v => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
      const rf = a => rng() < (a === goodArm ? P_GOOD : P_BAD) ? 1 : 0;
      const result = eqpropEpisode(net, [1], cf, rf, opts);
      arm = result.arm; reward = result.reward;

      if (condition === 'ema') {
        if (arm !== reportedArm) { totalReward -= switchCost; numSwitches++; reportedArm = arm; }
      } else if (condition === 'hardcoded') {
        if (deviationStart === null) { if (arm !== reportedArm) deviationStart = ep; }
        else {
          if (arm === reportedArm) { deviationStart = null; }
          else if (ep - deviationStart + 1 >= HARDCODED_THRESHOLD) {
            totalReward -= switchCost; numSwitches++;
            reportedArm = arm; deviationStart = null;
          }
        }
      }
    }

    totalReward += reward;
  }

  return { totalReward, numSwitches, finalMomentum: momentumA };
}

// Run all conditions across all seeds and costs
const conditions = ['ema', 'hardcoded', 'momentum'];
const results = {}; // results[condition][switchCost] = { rewards: [], switches: [] }

for (const cond of conditions) {
  results[cond] = {};
  for (const cost of SWITCH_COSTS) results[cond][cost] = { rewards: [], switches: [] };
}

// Also track identity resilience: at the end, how strong is momentum?
const finalMomentums = [];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  for (const cond of conditions) {
    for (const cost of SWITCH_COSTS) {
      const { totalReward, numSwitches, finalMomentum } =
        runAgent(cond, tl, cost, seed + conditions.indexOf(cond) + 1);
      results[cond][cost].rewards.push(totalReward);
      results[cond][cost].switches.push(numSwitches);
      if (cond === 'momentum' && cost === 0) finalMomentums.push(finalMomentum);
    }
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
const N = SEEDS.length;

// Print switch counts (cost-independent)
console.log(`Pooled: ${N} seeds x ${TOTAL_EPISODES} episodes\n`);
console.log('=== Switch counts ===');
for (const cond of conditions) {
  const sw = results[cond][0].switches;
  console.log(cond.padEnd(12), 'avg:', avg(sw).toFixed(0), ' range:', Math.min(...sw), '-', Math.max(...sw));
}

// Print reward table
console.log('\n=== Cumulative reward by switching cost ===');
const header = 'cost | ' + conditions.map(c => c.padEnd(10)).join(' | ') + ' | winner';
console.log(header);
console.log('-'.repeat(header.length));

let crossoverEmaToMomentum = null;
for (const cost of SWITCH_COSTS) {
  const vals = conditions.map(c => avg(results[c][cost].rewards));
  const maxVal = Math.max(...vals);
  const winnerIdx = vals.indexOf(maxVal);
  const winner = conditions[winnerIdx];

  if (!crossoverEmaToMomentum && vals[2] > vals[0]) crossoverEmaToMomentum = cost; // momentum > ema

  console.log(
    String(cost).padStart(4), '|',
    vals.map(v => v.toFixed(0).padStart(9)).join(' | '),
    '|', winner
  );
}

console.log('\n=== Crossover ===');
console.log('EMA → momentum crossover at switch cost:', crossoverEmaToMomentum ?? '>20');

// Momentum identity profile
const avgFinalMom = avg(finalMomentums);
console.log('\n=== Momentum identity profile ===');
console.log(`Final momentum at ep ${TOTAL_EPISODES}: ${avgFinalMom.toFixed(4)}`);
console.log(`(0.5 = no identity / 50:50, 0.0 or 1.0 = fully committed to one arm)`);
console.log(`Distance from neutral: ${Math.abs(avgFinalMom - 0.5).toFixed(4)}`);
console.log(`Strength of accumulated identity: ${(Math.abs(avgFinalMom - 0.5) * 2 * 100).toFixed(1)}%`);

// The key test: does momentum actually differentiate between probes and genuine changes?
// Run a diagnostic on one seed with verbose logging around a probe and genuine change
console.log('\n=== Momentum diagnostic: probe vs genuine change ===');
{
  const diagRng = makeRng(42);
  const diagEvents = buildSchedule(diagRng);
  const diagTl = buildTrueArmTimeline(diagEvents);

  const net2 = makeNetwork(1, 4, makeRng(43));
  const opts2 = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const r2 = makeRng(43);
  let mom = 0.5;

  // Run until we're well-trained (20000 ep), then track a probe and genuine change
  for (let ep = 0; ep < 20000; ep++) {
    const ga = diagTl[ep];
    const cf = v => {
      const eff = [v[0] + MOMENTUM_BETA*(mom), v[1] + MOMENTUM_BETA*(1-mom)];
      if(r2()<0.07) return r2()<0.5?0:1;
      return eff[0]>=eff[1]?0:1;
    };
    const rf = a => r2()<(a===ga?P_GOOD:P_BAD)?1:0;
    const { arm } = eqpropEpisode(net2, [1], cf, rf, opts2);
    mom = (1-MOMENTUM_ALPHA)*mom + MOMENTUM_ALPHA*(arm===0?1:0);
  }

  console.log(`\nMomentum after 20,000 episodes: ${mom.toFixed(4)} (${mom>0.5?'arm-0 identity':'arm-1 identity'})`);

  // Find next probe and genuine change after ep 20000
  const nextProbe = diagEvents.find(e => e.type==='probe' && e.start > 20000);
  const nextGenuine = diagEvents.find(e => e.type==='genuine' && e.start > 20000);

  if (nextProbe) {
    // Run up to probe
    const momBeforeProbe = mom;
    let tempMom = mom;
    const tempNet = net2; // use trained net
    const tr2 = makeRng(100);

    console.log(`\nProbe at ep ${nextProbe.start}, length ${nextProbe.length}:`);
    console.log('phase    | momentum  | delta from pre-probe');
    for (let ep = nextProbe.start - 3; ep < nextProbe.start + nextProbe.length + 5 && ep < TOTAL_EPISODES; ep++) {
      const ga = diagTl[ep];
      const cf = v => {
        const eff = [v[0]+MOMENTUM_BETA*tempMom, v[1]+MOMENTUM_BETA*(1-tempMom)];
        if(tr2()<0.07) return tr2()<0.5?0:1; return eff[0]>=eff[1]?0:1;
      };
      const rf = a => tr2()<(a===ga?P_GOOD:P_BAD)?1:0;
      const { arm } = eqpropEpisode(tempNet, [1], cf, rf, opts2);
      tempMom = (1-MOMENTUM_ALPHA)*tempMom + MOMENTUM_ALPHA*(arm===0?1:0);
      const phase = ep < nextProbe.start ? 'pre-probe' : ep < nextProbe.start+nextProbe.length ? 'PROBE    ' : 'post-probe';
      console.log(phase, '|', tempMom.toFixed(5), '|', (tempMom-momBeforeProbe).toFixed(5));
    }
  }
}
