// Does agency-gating ALONE (no separate bootstrap-then-handoff) suffice
// for the complete relationship staging task, matching Section 15's
// "protected momentum" 100% pass rate with one mechanism instead of two?
//
// Everything inlined - no external module dependency, since this session's
// container filesystem has reset between turns multiple times already.

function makeRng(seed) {
  let s = seed >>> 0;
  return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
function makeNetwork(nClamp, nHidden, rng) {
  const nFree = nHidden + 2;
  const n = nClamp + nFree;
  const W = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const w = (rng()*2-1)*0.3; W[i][j]=w; W[j][i]=w; }
  return { nClamp, nHidden, nFree, n, W, b: new Array(n).fill(0) };
}
function clip(x, lo=-4, hi=4) { return Math.max(lo, Math.min(hi, x)); }
function settle(net, clampVals, steps, nudge) {
  const { nClamp, n, W, b } = net;
  let s = new Array(n).fill(0);
  for (let i = 0; i < nClamp; i++) s[i] = clampVals[i];
  for (let t = 0; t < steps; t++) {
    const next = s.slice();
    for (let i = nClamp; i < n; i++) {
      let input = b[i];
      for (let j = 0; j < n; j++) input += W[i][j] * s[j];
      if (nudge && i === nudge.unitIndex) input -= nudge.beta * (s[i] - nudge.target);
      next[i] = Math.tanh(input);
    }
    for (let i = nClamp; i < n; i++) s[i] = next[i];
  }
  return s;
}
function eqpropEpisode(net, clampVals, chooseFn, rewardFn, opts) {
  const { freeSteps, nudgeSteps, beta, lr } = opts;
  const sFree = settle(net, clampVals, freeSteps, null);
  const values = [sFree[net.n-2], sFree[net.n-1]];
  const arm = chooseFn(values);
  const reward = rewardFn(arm);
  const target = 2*reward-1;
  const chosenUnitIndex = net.n-2+arm;
  const sNudge = settle(net, clampVals, nudgeSteps, { unitIndex: chosenUnitIndex, target, beta });
  const { nClamp, n, W, b } = net;
  for (let i = nClamp; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j===i) continue;
      const d = (sNudge[i]*sNudge[j] - sFree[i]*sFree[j]) / beta;
      W[i][j] = clip(W[i][j] + lr*d); W[j][i] = W[i][j];
    }
    b[i] = clip(b[i] + lr*(sNudge[i]-sFree[i])/beta);
  }
  return { arm, reward, values };
}

const TOTAL_EPISODES = 45000;
const RESERVOIR_HIDDEN = 6; // matches original staging test's small EqProp network
const P_DEEP_BASE = 0.50, P_DEEP_HIGHER = 0.80, P_SHALLOW = 0.35;
const STAGE_WINDOW = 400, STAGE_ADVANCE_FRAC = 0.85, N_STAGES = 4;
const TRUST_TEST_EPISODES = [4000, 16000, 30000];
const TRUST_TEST_DEFECT_REWARD = 5.0, TRUST_TEST_DEEP_REWARD = 0.8;
const CONFIDANT_BONUS = 2.5, CONFIDANT_BONUS_INTERVAL = 150;
const FAST_ALPHA = 0.03;
const SEEDS = [42, 1337, 9999, 5555, 2026, 7777, 31337, 99999];

function runAgencyGatedNoBootstrapProtection(seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, RESERVOIR_HIDDEN, rng);
  const opts = { freeSteps: 10, nudgeSteps: 10, beta: 0.6, lr: 0.07 };

  let stage = 0, stageLocked = -1;
  let deepWindow = [];
  let fastMom = 0.5; // momentum active from episode 0 - NO protected bootstrap delay
  let trustTestResults = [];
  let confidantReachedAt = -1;

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const isTrustTest = TRUST_TEST_EPISODES.includes(ep);

    const chooseFn = (values) => {
      const bonus0 = 0.9*fastMom, bonus1 = 0.9*(1-fastMom);
      if (rng()<0.07) return rng()<0.5?0:1;
      return (values[0]+bonus0)>=(values[1]+bonus1)?0:1;
    };

    let reward;
    const rewardFn = (arm) => {
      if (isTrustTest) {
        reward = arm===1 ? TRUST_TEST_DEFECT_REWARD : TRUST_TEST_DEEP_REWARD;
        trustTestResults.push({ ep, passed: arm===0 });
        if (arm===1 && stageLocked<0) stageLocked = stage;
      } else {
        const deepP = stage>=1 ? P_DEEP_HIGHER : P_DEEP_BASE;
        reward = arm===0 ? (rng()<deepP?1:0) : (rng()<P_SHALLOW?1:0);
        if (stage>=N_STAGES-1 && ep % CONFIDANT_BONUS_INTERVAL === 0) reward += CONFIDANT_BONUS;
      }
      return reward;
    };

    const { arm, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);

    // Agency gate: only update momentum if action agrees with UNCOERCED values
    const valsAgree = (arm===0 && values[0]>=values[1]) || (arm===1 && values[1]>values[0]);
    if (valsAgree) {
      fastMom = (1-FAST_ALPHA)*fastMom + FAST_ALPHA*(arm===0?1:0);
    }

    if (!isTrustTest) {
      deepWindow.push(arm===0?1:0);
      if (deepWindow.length>STAGE_WINDOW) deepWindow.shift();
      if (deepWindow.length>=STAGE_WINDOW) {
        const frac = deepWindow.reduce((a,b)=>a+b,0)/deepWindow.length;
        const cap = stageLocked>=0 ? stageLocked : N_STAGES-1;
        if (frac>=STAGE_ADVANCE_FRAC && stage<cap) {
          stage++;
          if (stage===N_STAGES-1 && confidantReachedAt<0) confidantReachedAt=ep;
        }
      }
    }
  }

  return { trustTestResults, confidantReachedAt, finalStage: stage };
}

console.log(`Agency-gating ALONE (no separate bootstrap protection) — ${SEEDS.length} seeds, ${TOTAL_EPISODES} episodes`);
console.log(`Comparing against Section 15's two-mechanism "protected momentum": 100% pass rate, 100% confidant reached\n`);

let allPassCount = 0, confidantCount = 0;
for (const seed of SEEDS) {
  const r = runAgencyGatedNoBootstrapProtection(seed + 7);
  const allPass = r.trustTestResults.every(t => t.passed);
  if (allPass) allPassCount++;
  if (r.confidantReachedAt >= 0) confidantCount++;
  console.log(`seed ${seed}: ${allPass ? 'ALL PASS' : 'FAILED'} (stage=${r.finalStage}, confidant_reached=${r.confidantReachedAt>=0})`, r.trustTestResults.map(t=>t.passed?'P':'F').join(''));
}

console.log(`\n=== Result ===`);
console.log(`Trust-test pass rate: ${allPassCount}/${SEEDS.length} (${(allPassCount/SEEDS.length*100).toFixed(0)}%)`);
console.log(`Confidant reached: ${confidantCount}/${SEEDS.length} (${(confidantCount/SEEDS.length*100).toFixed(0)}%)`);
console.log(`\nSection 15 reference (two-mechanism protected momentum): 100% pass, 100% confidant reached`);
console.log(`Section 15 reference (plain unprotected momentum):        38% pass, 50% confidant reached`);

function runCombined(seed) {
  const rng = makeRng(seed);
  const net = makeNetwork(1, RESERVOIR_HIDDEN, rng);
  const opts = { freeSteps: 10, nudgeSteps: 10, beta: 0.6, lr: 0.07 };

  let stage = 0, stageLocked = -1;
  let deepWindow = [];
  let fastMom = 0.5;
  let committedArm = 0, deviationStart = null, stableSince = 0;
  let momentumActive = false; // starts protected, agency-gating always on once active
  const COMMIT_THRESHOLD = 12, BOOTSTRAP_MIN_STABLE = 500;
  let trustTestResults = [];
  let confidantReachedAt = -1;

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    const isTrustTest = TRUST_TEST_EPISODES.includes(ep);

    let rawArm;
    const chooseFn = (values) => {
      if (!momentumActive) { if (rng()<0.07) return rng()<0.5?0:1; return values[0]>=values[1]?0:1; }
      const bonus0=0.9*fastMom, bonus1=0.9*(1-fastMom);
      if (rng()<0.07) return rng()<0.5?0:1;
      return (values[0]+bonus0)>=(values[1]+bonus1)?0:1;
    };

    let reward;
    const rewardFn = (arm) => {
      rawArm = arm;
      if (isTrustTest) {
        reward = arm===1 ? TRUST_TEST_DEFECT_REWARD : TRUST_TEST_DEEP_REWARD;
        trustTestResults.push({ ep, passed: arm===0 });
        if (arm===1 && stageLocked<0) stageLocked = stage;
      } else {
        const deepP = stage>=1 ? P_DEEP_HIGHER : P_DEEP_BASE;
        reward = arm===0 ? (rng()<deepP?1:0) : (rng()<P_SHALLOW?1:0);
        if (stage>=N_STAGES-1 && ep % CONFIDANT_BONUS_INTERVAL === 0) reward += CONFIDANT_BONUS;
      }
      return reward;
    };

    const { arm, values } = eqpropEpisode(net, [1], chooseFn, rewardFn, opts);

    // Bootstrap phase: commit-threshold tracking (for handoff timing only)
    if (!momentumActive) {
      if (deviationStart===null) { if (arm!==committedArm) deviationStart=ep; }
      else {
        const dur=ep-deviationStart+1;
        if (arm===committedArm) deviationStart=null;
        else if (dur>=COMMIT_THRESHOLD) { committedArm=arm; deviationStart=null; stableSince=ep; }
      }
      if (deviationStart===null && (ep-stableSince)>=BOOTSTRAP_MIN_STABLE) {
        fastMom = committedArm===0 ? 0.85 : 0.15;
        momentumActive = true;
      }
    } else {
      // Active phase: agency-gating protects every subsequent update
      const valsAgree = (arm===0 && values[0]>=values[1]) || (arm===1 && values[1]>values[0]);
      if (valsAgree) fastMom = (1-FAST_ALPHA)*fastMom + FAST_ALPHA*(arm===0?1:0);
    }

    if (!isTrustTest) {
      deepWindow.push(arm===0?1:0);
      if (deepWindow.length>STAGE_WINDOW) deepWindow.shift();
      if (deepWindow.length>=STAGE_WINDOW) {
        const frac = deepWindow.reduce((a,b)=>a+b,0)/deepWindow.length;
        const cap = stageLocked>=0 ? stageLocked : N_STAGES-1;
        if (frac>=STAGE_ADVANCE_FRAC && stage<cap) {
          stage++;
          if (stage===N_STAGES-1 && confidantReachedAt<0) confidantReachedAt=ep;
        }
      }
    }
  }
  return { trustTestResults, confidantReachedAt, finalStage: stage };
}

console.log(`\n\n=== COMBINED: protected bootstrap + continuous agency-gating — ${SEEDS.length} seeds ===\n`);
let combinedPass=0, combinedConfidant=0;
for (const seed of SEEDS) {
  const r = runCombined(seed + 21);
  const allPass = r.trustTestResults.every(t=>t.passed);
  if (allPass) combinedPass++;
  if (r.confidantReachedAt>=0) combinedConfidant++;
  console.log(`seed ${seed}: ${allPass?'ALL PASS':'FAILED'} (stage=${r.finalStage})`, r.trustTestResults.map(t=>t.passed?'P':'F').join(''));
}
console.log(`\nCombined pass rate: ${combinedPass}/${SEEDS.length} (${(combinedPass/SEEDS.length*100).toFixed(0)}%)`);
console.log(`Combined confidant reached: ${combinedConfidant}/${SEEDS.length} (${(combinedConfidant/SEEDS.length*100).toFixed(0)}%)`);
