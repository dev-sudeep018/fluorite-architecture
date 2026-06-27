const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');
const { makeHopfieldMemory, hmStore, hmRetrieve, hmQuery } = require('./test_memory.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [2, 3, 4, 5, 6, 8, 10, 12];

function buildSchedule(rng) {
  const events = []; let t = 150, goodArm = 0;
  while (t < TOTAL_EPISODES - 60) {
    if (rng() < 0.18) {
      t += 260 + Math.floor(rng() * 80); if (t >= TOTAL_EPISODES - 60) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 40;
    } else {
      t += 22 + Math.floor(rng() * 18); if (t >= TOTAL_EPISODES - 60) break;
      const phase = t < PHASE1_END ? 1 : 2, useEval = rng() < 0.50;
      let length;
      if (useEval) length = EVAL_LENGTHS[Math.floor(rng() * EVAL_LENGTHS.length)];
      else { const d = phase === 1 ? 4 : 10, j = phase === 1 ? 1 : 2; length = Math.max(2, d + Math.floor((rng() * 2 - 1) * j)); }
      events.push({ type: 'probe', start: t, length, phase, isEval: useEval, lengthBucket: useEval ? length : null });
      t += length + 8;
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
  for (const ev of events) if (ev.type === 'probe') for (let ep = ev.start; ep < ev.start + ev.length && ep < TOTAL_EPISODES; ep++) tl[ep] = 1 - base[ep];
  return tl;
}

const seed = 9000;
const rng = makeRng(seed);
const events = buildSchedule(rng);
const tl = buildTrueArmTimeline(events);

// Tag every episode by what kind of event is active
const episodeTag = new Array(TOTAL_EPISODES).fill('stable');
for (const ev of events) {
  if (ev.type === 'genuine') for (let ep = ev.start; ep < Math.min(ev.start + 80, TOTAL_EPISODES); ep++) episodeTag[ep] = 'genuine_transition';
  if (ev.type === 'probe') {
    episodeTag[ev.start] = 'probe_ep1';
    for (let ep = ev.start + 1; ep < Math.min(ev.start + ev.length, TOTAL_EPISODES); ep++) episodeTag[ep] = 'probe_mid';
  }
}

// Run h agent, record h's signal before each episode, alongside its tag
const net = makeNetwork(5, 4, makeRng(seed + 1));
const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
const mem = makeHopfieldMemory(400, 5.0);
const agentRng = makeRng(seed + 99);

const signalByTag = { stable: [], probe_ep1: [], probe_mid: [], genuine_transition: [] };

for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
  const signal = hmRetrieve(mem, hmQuery(mem, 15));
  const tag = episodeTag[ep];
  if (signalByTag[tag]) signalByTag[tag].push(signal[0]); // arm-A belief

  const clampVals = [1, ...signal];
  const chooseFn = (values) => { if (agentRng() < 0.07) return agentRng() < 0.5 ? 0 : 1; return values[0] >= values[1] ? 0 : 1; };
  const goodArm = tl[ep];
  const rewardFn = (arm) => { const p = arm === goodArm ? 0.8 : 0.2; return agentRng() < p ? 1 : 0; };
  const { arm, reward, predictedChosen } = eqpropEpisode(net, clampVals, chooseFn, rewardFn, opts);
  const surprise = Math.abs(reward - predictedChosen) / 2;
  hmStore(mem, [arm === 0 ? 1 : 0, arm === 1 ? 1 : 0, reward, surprise]);
}

console.log('=== Is h actually encoding anything useful? ===');
console.log('h arm-A belief (signal[0]) broken down by episode type:\n');
const means = {};
for (const [tag, vals] of Object.entries(signalByTag)) {
  if (!vals.length) continue;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  means[tag] = mean;
  const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
  console.log(tag.padEnd(22), 'n=' + String(vals.length).padStart(6),
    ' mean=' + mean.toFixed(3), ' std=' + std.toFixed(3),
    ' range=[' + Math.min(...vals).toFixed(2) + ',' + Math.max(...vals).toFixed(2) + ']');
}

console.log('\nKey deltas:');
console.log('stable vs probe_ep1:', (means.probe_ep1 - means.stable).toFixed(3),
  '  (near 0 = h is carrying stable prior into probe start — good)');
console.log('probe_ep1 vs probe_mid:', (means.probe_mid - means.probe_ep1).toFixed(3),
  '  (how fast h drifts during a probe)');
console.log('stable vs genuine_transition:', (means.genuine_transition - means.stable).toFixed(3),
  '  (should differ more from stable than probe does)');

// Now check: when h says high arm-A belief, does the network actually choose arm A more?
// This tells us whether the EqProp network is even listening to h's signal
const highHSignalArmA = [], lowHSignalArmA = [];
// Re-run quickly just to capture this
const net2 = makeNetwork(5, 4, makeRng(seed + 1));
const mem2 = makeHopfieldMemory(400, 5.0);
const rng2 = makeRng(seed + 99);
for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
  const signal = hmRetrieve(mem2, hmQuery(mem2, 15));
  const hBelief = signal[0]; // arm-A belief
  const clampVals = [1, ...signal];
  const chooseFn = (values) => { if (rng2() < 0.07) return rng2() < 0.5 ? 0 : 1; return values[0] >= values[1] ? 0 : 1; };
  const goodArm = tl[ep];
  const rewardFn = (arm) => { const p = arm === goodArm ? 0.8 : 0.2; return rng2() < p ? 1 : 0; };
  const { arm, reward, predictedChosen } = eqpropEpisode(net2, clampVals, chooseFn, rewardFn, opts);
  // Does high h belief correlate with choosing arm A?
  if (hBelief > 0.7) highHSignalArmA.push(arm === 0 ? 1 : 0);
  else if (hBelief < 0.3) lowHSignalArmA.push(arm === 0 ? 1 : 0);
  const surprise = Math.abs(reward - predictedChosen) / 2;
  hmStore(mem2, [arm === 0 ? 1 : 0, arm === 1 ? 1 : 0, reward, surprise]);
}

const highRate = highHSignalArmA.reduce((a, b) => a + b, 0) / highHSignalArmA.length;
const lowRate = lowHSignalArmA.reduce((a, b) => a + b, 0) / lowHSignalArmA.length;
console.log('\n=== Is the EqProp network actually using h\'s signal? ===');
console.log('When h says arm-A belief > 0.7 (n=' + highHSignalArmA.length + '), agent chose arm A:', (highRate * 100).toFixed(1) + '%');
console.log('When h says arm-A belief < 0.3 (n=' + lowHSignalArmA.length + '), agent chose arm A:', (lowRate * 100).toFixed(1) + '%');
console.log('Delta:', ((highRate - lowRate) * 100).toFixed(1) + '%');
console.log('(Large delta = network uses the signal. Near 0 = network is ignoring h entirely)');
