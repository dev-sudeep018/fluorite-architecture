// Shadow tracker v2 — fixed baseline chasing bug
// Key fix: baseline arm only updates when we COMMIT to a genuine change,
// not every time the agent happens to return to the current baseline.

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [4, 6, 8, 10, 12];

function buildSchedule(rng) {
  const events = []; let t = 200, goodArm = 0;
  while (t < TOTAL_EPISODES - 80) {
    if (rng() < 0.15) {
      t += 300 + Math.floor(rng() * 100); if (t >= TOTAL_EPISODES - 80) break;
      goodArm = 1 - goodArm;
      events.push({ type: 'genuine', start: t, phase: t < PHASE1_END ? 1 : 2 }); t += 50;
    } else {
      t += 20 + Math.floor(rng() * 15); if (t >= TOTAL_EPISODES - 80) break;
      const phase = t < PHASE1_END ? 1 : 2, useEval = rng() < 0.45;
      const pool = phase === 1 ? [2,3,4,5,6] : [6,8,10,12,14];
      const length = useEval ? EVAL_LENGTHS[Math.floor(rng() * EVAL_LENGTHS.length)] : pool[Math.floor(rng() * pool.length)];
      events.push({ type: 'probe', start: t, length, phase, isEval: useEval, lengthBucket: useEval ? length : null });
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
  for (const ev of events) if (ev.type === 'probe') for (let ep = ev.start; ep < ev.start + ev.length && ep < TOTAL_EPISODES; ep++) tl[ep] = 1 - base[ep];
  return tl;
}

function makePRevertTable(maxBucket, alpha) {
  // Start at 0.7 (prior: probes are more common than genuine changes)
  // No clamp — let the EMA move freely so buckets actually differentiate
  return { maxBucket, alpha, counts: new Array(maxBucket + 1).fill(0.7) };
}

function updatePRevert(table, duration, didRevert) {
  for (let b = 1; b <= Math.min(duration, table.maxBucket); b++) {
    table.counts[b] = (1 - table.alpha) * table.counts[b] + table.alpha * (didRevert ? 1 : 0);
  }
}

function getPRevert(table, duration) {
  return table.counts[Math.min(duration, table.maxBucket)];
}

function runCSAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const baseOpts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6 };

  const mainTable = makePRevertTable(25, 0.12);
  const shadowTable = makePRevertTable(25, 0.12);

  const preferredArm = new Array(TOTAL_EPISODES).fill(0);

  // Deviation tracking — frozen baseline approach
  let baselineArm = 0;       // committed arm — only changes on genuine events
  let deviationStart = null;
  let deviationDuration = 0;
  let consecutiveSameArm = 0; // how many consecutive eps on the deviation arm?

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];

    const pRevert = deviationStart !== null
      ? getPRevert(mainTable, deviationDuration)
      : 0.5;

    // Gate learning rate: high P_revert → slow down learning
    const lr = 0.07 * (1 - 0.80 * pRevert);
    const opts = { ...baseOpts, lr };

    const chooseFn = (v) => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rewardFn = (arm) => { const p = arm === goodArm ? 0.8 : 0.2; return rng() < p ? 1 : 0; };
    const { arm } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm;

    // Deviation state machine — frozen baseline
    if (deviationStart === null) {
      // In stable state
      if (arm !== baselineArm) {
        // New deviation starts
        deviationStart = ep;
        deviationDuration = 1;
        consecutiveSameArm = 1;
      }
      // baseline stays frozen — we don't update it during stable state
    } else {
      // In deviation
      deviationDuration = ep - deviationStart + 1;

      if (arm === baselineArm) {
        // Returned to baseline — it WAS a revert (probe)
        updatePRevert(mainTable, deviationDuration, true);
        updatePRevert(shadowTable, deviationDuration, true);
        deviationStart = null; consecutiveSameArm = 0;
      } else {
        consecutiveSameArm++;
        // Check: has this deviation been sustained long enough to call genuine?
        // Use the table: if P_revert is now very low AND duration is substantial
        const pR = getPRevert(mainTable, deviationDuration);
        const COMMIT_THRESHOLD = 10; // must be sustained this long before we even consider committing
        if (deviationDuration >= COMMIT_THRESHOLD && pR < 0.25) {
          // Commit — this is a genuine change
          updatePRevert(mainTable, deviationDuration, false);
          updatePRevert(shadowTable, deviationDuration, false);
          baselineArm = arm; // update baseline to the new committed arm
          deviationStart = null; consecutiveSameArm = 0;
        }
      }
    }
  }

  return { preferredArm, mainTable, shadowTable };
}

function runEMAAgent(trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };
  const preferredArm = new Array(TOTAL_EPISODES).fill(0);
  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];
    const chooseFn = (v) => { if (rng() < 0.07) return rng() < 0.5 ? 0 : 1; return v[0] >= v[1] ? 0 : 1; };
    const rewardFn = (arm) => { const p = arm === goodArm ? 0.8 : 0.2; return rng() < p ? 1 : 0; };
    const { arm } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    preferredArm[ep] = arm;
  }
  return { preferredArm };
}

function detectSwitches(preferredArm, events) {
  const results = [];
  for (const ev of events) {
    if (ev.type !== 'probe') continue;
    const preWindow = preferredArm.slice(Math.max(0, ev.start - 5), ev.start);
    const prePreferred = preWindow.filter(a => a === 0).length >= preWindow.length / 2 ? 0 : 1;
    let switched = false, streak = 0;
    for (let ep = ev.start; ep < Math.min(ev.start + ev.length, preferredArm.length); ep++) {
      if (preferredArm[ep] !== prePreferred) { streak++; if (streak >= 2) { switched = true; break; } }
      else streak = 0;
    }
    results.push({ ...ev, switched });
  }
  return results;
}

// Run across 3 seeds and pool
const seeds = [42, 1337, 9999];
const csAgg = {}, emaAgg = {};
for (const phase of [1,2]) { csAgg[phase] = {}; emaAgg[phase] = {}; for (const len of EVAL_LENGTHS) { csAgg[phase][len] = {sw:0,n:0}; emaAgg[phase][len] = {sw:0,n:0}; } }

let shadowDiagDone = false;
let snapshotDone = false;
let phase1Snapshot = null, phase2Snapshot = null;
let shadowFinal = null;

for (const seed of seeds) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const cs = runCSAgent(tl, seed + 1);
  const ema = runEMAAgent(tl, seed + 2);

  // Capture shadow diagnostic from first seed only
  if (!shadowDiagDone) {
    shadowFinal = { main: cs.mainTable.counts.slice(), shadow: cs.shadowTable.counts.slice() };
    // Re-run to capture phase1 snapshot
    // (quick second pass for snapshot — same seed)
    const rng2 = makeRng(seed); const ev2 = buildSchedule(rng2); const tl2 = buildTrueArmTimeline(ev2);
    const rng2a = makeRng(seed+1), net2 = makeNetwork(1,4,rng2a), baseOpts2 = {freeSteps:22,nudgeSteps:22,beta:0.6};
    const mt2 = makePRevertTable(25,0.12); let bl2=0, ds2=null, dd2=0;
    for (let ep=0;ep<TOTAL_EPISODES;ep++){
      if(ep===PHASE1_END) phase1Snapshot=mt2.counts.slice();
      const goodArm=tl2[ep],pR=ds2!==null?getPRevert(mt2,dd2):0.5,lr=0.07*(1-0.80*pR);
      const opts={...baseOpts2,lr};
      const cf=v=>{if(rng2a()<0.07)return rng2a()<0.5?0:1;return v[0]>=v[1]?0:1;};
      const rf=arm=>{const p=arm===goodArm?0.8:0.2;return rng2a()<p?1:0;};
      const{arm}=eqpropEpisode(net2,[1],cf,rf,opts);
      if(ds2===null){if(arm!==bl2){ds2=ep;dd2=1;}}
      else{dd2=ep-ds2+1;if(arm===bl2){updatePRevert(mt2,dd2,true);ds2=null;}else{const pRc=getPRevert(mt2,dd2);if(dd2>=10&&pRc<0.25){updatePRevert(mt2,dd2,false);bl2=arm;ds2=null;}}}
    }
    phase2Snapshot=mt2.counts.slice();
    shadowDiagDone=true;
  }

  const csSw = detectSwitches(cs.preferredArm, events);
  const emaSw = detectSwitches(ema.preferredArm, events);
  for (const s of csSw) { if(s.isEval&&csAgg[s.phase]&&csAgg[s.phase][s.lengthBucket]){csAgg[s.phase][s.lengthBucket].n++;if(s.switched)csAgg[s.phase][s.lengthBucket].sw++;} }
  for (const s of emaSw) { if(s.isEval&&emaAgg[s.phase]&&emaAgg[s.phase][s.lengthBucket]){emaAgg[s.phase][s.lengthBucket].n++;if(s.switched)emaAgg[s.phase][s.lengthBucket].sw++;} }
}

console.log(`Pooled across ${seeds.length} seeds\n`);
console.log('=== Spurious switch rate by probe length ===');
console.log('len | CS-p1          CS-p2    | EMA-p1         EMA-p2');
const f = (agg,phase,len) => { const c=agg[phase][len]; return c.n?((c.sw/c.n).toFixed(2)+'(n'+c.n+')').padEnd(10):'  --      '; };
for (const len of EVAL_LENGTHS) console.log(String(len).padStart(3),'|',f(csAgg,1,len),f(csAgg,2,len),'|',f(emaAgg,1,len),f(emaAgg,2,len));

console.log('\n=== Shadow tracker diagnostic (seed 42) ===');
console.log('If main diverges from shadow, the table is learning its own policy, not the world.');
console.log('bucket | main  | shadow | delta');
for (let b=1;b<=15;b++){
  const main=shadowFinal.main[b], shadow=shadowFinal.shadow[b], delta=main-shadow;
  console.log(String(b).padStart(6),'|',main.toFixed(3),'|',shadow.toFixed(3),'|',(delta>=0?'+':'')+delta.toFixed(3));
}

console.log('\n=== Phase recalibration (main table, seed 42) ===');
console.log('bucket | p1-end | p2-end | shift');
for (let b=1;b<=15;b++){
  const p1=phase1Snapshot[b],p2=phase2Snapshot[b];
  console.log(String(b).padStart(6),'|',p1.toFixed(3),'|',p2.toFixed(3),'|',((p2-p1)>=0?'+':'')+( p2-p1).toFixed(3));
}
