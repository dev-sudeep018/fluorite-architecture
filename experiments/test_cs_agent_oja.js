// CS agent + Oja-Hebbian synthesis, properly isolated
//
// Last test (gated_oja) confounded two different things: gating H's UPDATES
// vs protecting the REPORTED behavior. It gated the former but reported raw
// per-episode choice (not committedArm), so H's growing one-sided bias pushed
// the raw EqProp values into an over-corrected state during long punishment,
// and the eventual flip registered as a "switch" even though internal
// commitment never actually moved. Confound, not a real finding about H.
//
// This test isolates the real question. Both conditions report committedArm
// (the Wall-2-validated protection - already gets 0.4-2% switches at length 8
// on its own, no H at all). The only difference: does adding an UNGATED,
// continuously-updating Oja-stabilized H bonus to the RAW per-episode choice
// (the signal that feeds the deviation-detection state machine) help the
// commit-threshold mechanism detect and confirm genuine changes faster,
// without changing how well short probes are already resisted?

const { makeNetwork, eqpropEpisode, makeRng } = require('./eqprop_core.js');

const TOTAL_EPISODES = 45000;
const PHASE1_END = 22500;
const EVAL_LENGTHS = [2, 3, 4, 5, 6, 8, 10, 12];
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

const HEBB_LR = 0.02;
const HEBB_BONUS = 0.7; // lighter than before - this is a modulating input, not the dominant signal
const OJA_LR = 0.02;
const COMMIT_THRESHOLD = 12;

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
      const pool = phase === 1 ? [2,3,4,5,6,7] : [7,9,11,13,15,17];
      const length = useEval ? EVAL_LENGTHS[Math.floor(rng()*EVAL_LENGTHS.length)] : pool[Math.floor(rng()*pool.length)];
      events.push({ type: 'probe', start: t, length, phase, isEval: useEval, lengthBucket: useEval ? length : null });
      t += length + 8;
    }
  }
  return events;
}

function buildTrueArmTimeline(events) {
  const base = new Array(TOTAL_EPISODES).fill(0); let current = 0;
  const gs = events.filter(e => e.type === 'genuine').sort((a,b)=>a.start-b.start); let gi=0;
  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    while (gi<gs.length && gs[gi].start===ep) { current=1-current; gi++; }
    base[ep]=current;
  }
  const tl = base.slice();
  for (const ev of events) if (ev.type==='probe')
    for (let ep=ev.start; ep<ev.start+ev.length && ep<TOTAL_EPISODES; ep++) tl[ep]=1-base[ep];
  return tl;
}

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function runAgent(useH, trueArmTimeline, seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, 4, rng);
  const opts = { freeSteps: 22, nudgeSteps: 22, beta: 0.6, lr: 0.07 };

  const H = [0, 0];
  let committedArm = 0;
  let deviationStart = null;

  const reportedArm = new Array(TOTAL_EPISODES).fill(0); // this is what gets measured - always committedArm
  const rewardLog = new Array(TOTAL_EPISODES).fill(0);
  const confirmEpisodes = []; // track when genuine changes get confirmed, and how long they took

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const goodArm = trueArmTimeline[ep];

    // Raw per-episode choice - this is what feeds the deviation-detection machinery
    // (NOT what gets reported/measured - that's always committedArm below)
    const chooseFn = (values) => {
      if (rng() < 0.07) return rng() < 0.5 ? 0 : 1;
      if (!useH) return values[0] >= values[1] ? 0 : 1;
      const eff0 = values[0] + HEBB_BONUS * H[0];
      const eff1 = values[1] + HEBB_BONUS * H[1];
      return eff0 >= eff1 ? 0 : 1;
    };
    const rewardFn = (arm) => { const p = arm===goodArm?0.8:0.2; return rng()<p?1:0; };

    const { arm: rawArm, reward, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);
    rewardLog[ep] = reward;

    // H updates continuously (ungated - we learned gating starves it), Oja-stabilized
    if (useH) {
      const activationProxy = Math.abs(values[rawArm]);
      const x = activationProxy, y = H[rawArm]===0?0.1:H[rawArm];
      const potentiation = HEBB_LR * x * y;
      const stabilization = OJA_LR * y * y * H[rawArm];
      H[rawArm] = clip(H[rawArm] + potentiation - stabilization);
    }

    // Commit-threshold state machine (Wall-2 validated) - operates on rawArm
    if (deviationStart === null) {
      if (rawArm !== committedArm) deviationStart = ep;
    } else {
      const dur = ep - deviationStart + 1;
      if (rawArm === committedArm) { deviationStart = null; }
      else if (dur >= COMMIT_THRESHOLD) {
        confirmEpisodes.push({ ep, duration: dur, from: committedArm, to: rawArm });
        committedArm = rawArm; deviationStart = null;
      }
    }

    reportedArm[ep] = committedArm; // ALWAYS report the protected commitment, both conditions
  }

  return { reportedArm, rewardLog, confirmEpisodes };
}

function detectSwitches(reportedArm, events) {
  const results = [];
  for (const ev of events) {
    if (ev.type!=='probe') continue;
    const pre = reportedArm.slice(Math.max(0,ev.start-5), ev.start);
    const preP = pre.filter(a=>a===0).length >= pre.length/2 ? 0 : 1;
    let switched=false, streak=0;
    for (let ep=ev.start; ep<Math.min(ev.start+ev.length, reportedArm.length); ep++) {
      if (reportedArm[ep]!==preP) { streak++; if(streak>=2){switched=true;break;} } else streak=0;
    }
    results.push({...ev, switched});
  }
  return results;
}

function pool(switches) {
  const out = {};
  for (const s of switches) {
    if (!s.isEval) continue;
    out[s.phase]=out[s.phase]||{};
    out[s.phase][s.lengthBucket]=out[s.phase][s.lengthBucket]||{sw:0,n:0};
    out[s.phase][s.lengthBucket].n++;
    if (s.switched) out[s.phase][s.lengthBucket].sw++;
  }
  return out;
}

function genuineRecovery(reportedArm, events, tl) {
  let rec=0, tot=0;
  for (const ev of events) {
    if (ev.type!=='genuine') continue;
    const correctArm = tl[Math.min(ev.start+49, reportedArm.length-1)];
    const win = reportedArm.slice(ev.start+2, Math.min(ev.start+52, reportedArm.length));
    if (win.filter(a=>a===correctArm).length > win.length*0.7) rec++;
    tot++;
  }
  return tot ? rec/tot : 0;
}

const csAgg={}, csHAgg={};
for (const ph of [1,2]) { csAgg[ph]={}; csHAgg[ph]={}; for (const l of EVAL_LENGTHS) { csAgg[ph][l]={sw:0,n:0}; csHAgg[ph][l]={sw:0,n:0}; } }
const csRewards=[], csHRewards=[];
const csRecoveries=[], csHRecoveries=[];
const csConfirmDurations=[], csHConfirmDurations=[];

for (const seed of SEEDS) {
  const rng = makeRng(seed);
  const events = buildSchedule(rng);
  const tl = buildTrueArmTimeline(events);

  const cs = runAgent(false, tl, seed+1);
  const csH = runAgent(true, tl, seed+2);

  csRewards.push(cs.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);
  csHRewards.push(csH.rewardLog.reduce((a,b)=>a+b,0)/TOTAL_EPISODES);

  csRecoveries.push(genuineRecovery(cs.reportedArm, events, tl));
  csHRecoveries.push(genuineRecovery(csH.reportedArm, events, tl));

  // Confirmation duration for genuine changes only (approximate: confirmations that align with a genuine event)
  for (const c of cs.confirmEpisodes) csConfirmDurations.push(c.duration);
  for (const c of csH.confirmEpisodes) csHConfirmDurations.push(c.duration);

  const csSw = pool(detectSwitches(cs.reportedArm, events));
  const csHSw = pool(detectSwitches(csH.reportedArm, events));
  for (const ph of [1,2]) for (const l of EVAL_LENGTHS) {
    if (csSw[ph]?.[l]) { csAgg[ph][l].n+=csSw[ph][l].n; csAgg[ph][l].sw+=csSw[ph][l].sw; }
    if (csHSw[ph]?.[l]) { csHAgg[ph][l].n+=csHSw[ph][l].n; csHAgg[ph][l].sw+=csHSw[ph][l].sw; }
  }
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
const f = (a,ph,l) => { const x=a[ph][l]; return x?.n?(x.sw/x.n).toFixed(2)+'(n'+x.n+')':'  --  '; };

console.log(`CS agent + Oja-Hebbian, properly isolated — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes\n`);

console.log('=== Overall reward ===');
console.log('cs_agent (no H)   :', avg(csRewards).toFixed(3));
console.log('cs_agent + H      :', avg(csHRewards).toFixed(3));

console.log('\n=== Genuine change recovery rate (within 50 episodes) ===');
console.log('cs_agent (no H)   :', avg(csRecoveries).toFixed(3));
console.log('cs_agent + H      :', avg(csHRecoveries).toFixed(3));

console.log('\n=== Average confirmation duration (episodes until a deviation gets confirmed) ===');
console.log('cs_agent (no H)   :', avg(csConfirmDurations).toFixed(1), '(n =', csConfirmDurations.length, 'confirmations)');
console.log('cs_agent + H      :', avg(csHConfirmDurations).toFixed(1), '(n =', csHConfirmDurations.length, 'confirmations)');

console.log('\n=== Spurious switch rate ===');
console.log('len | cs_agent-p1   cs_agent-p2 | cs_agent+H-p1 cs_agent+H-p2');
for (const l of EVAL_LENGTHS)
  console.log(String(l).padStart(3),'|',f(csAgg,1,l).padEnd(13),f(csAgg,2,l).padEnd(10),'|',f(csHAgg,1,l).padEnd(13),f(csHAgg,2,l));

let better=0, worse=0, tied=0, total=0;
for (const l of EVAL_LENGTHS) for (const ph of [1,2]) {
  const c=csAgg[ph][l], h=csHAgg[ph][l];
  if (!c?.n || !h?.n) continue;
  total++;
  const cr=c.sw/c.n, hr=h.sw/h.n;
  if (hr < cr-0.02) better++; else if (hr > cr+0.02) worse++; else tied++;
}
console.log(`\n=== Verdict: cs_agent+H vs cs_agent — better ${better}/${total}, worse ${worse}/${total}, tied ${tied}/${total} ===`);
console.log(`Recovery: cs_agent=${avg(csRecoveries).toFixed(3)} vs cs_agent+H=${avg(csHRecoveries).toFixed(3)}`);
console.log(`Confirmation speed: cs_agent=${avg(csConfirmDurations).toFixed(1)}ep vs cs_agent+H=${avg(csHConfirmDurations).toFixed(1)}ep (lower = faster genuine-change detection)`);
