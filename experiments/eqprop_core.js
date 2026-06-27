// ---------- Hopfield / EqProp network ----------
// Units layout: [clamped...][hidden free...][output free (arm0, arm1)]
// All units tanh, symmetric weights, zero diagonal.

function makeNetwork(nClamp, nHidden, rng) {
  const nFree = nHidden + 2;
  const n = nClamp + nFree;
  const W = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = (rng() * 2 - 1) * 0.3;
      W[i][j] = w;
      W[j][i] = w;
    }
  }
  const b = new Array(n).fill(0);
  return { nClamp, nHidden, nFree, n, W, b };
}

function clip(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

// One settling pass. clampVals: array length nClamp.
// nudge: null, or {unitIndex, target, beta} applied only to that free unit each step.
function settle(net, clampVals, steps, nudge) {
  const { nClamp, n, W, b } = net;
  let s = new Array(n).fill(0);
  for (let i = 0; i < nClamp; i++) s[i] = clampVals[i];

  for (let t = 0; t < steps; t++) {
    const next = s.slice();
    for (let i = nClamp; i < n; i++) {
      let input = b[i];
      for (let j = 0; j < n; j++) input += W[i][j] * s[j];
      if (nudge && i === nudge.unitIndex) {
        input -= nudge.beta * (s[i] - nudge.target);
      }
      next[i] = Math.tanh(input);
    }
    for (let i = nClamp; i < n; i++) s[i] = next[i];
  }
  return s;
}

function outputUnits(net, s) {
  // last two free units are arm0, arm1 value estimates
  return [s[net.n - 2], s[net.n - 1]];
}

// One full EqProp episode: settle free, choose, observe, nudge, update weights.
function eqpropEpisode(net, clampVals, chooseFn, rewardFn, opts) {
  const { freeSteps, nudgeSteps, beta, lr } = opts;
  const sFree = settle(net, clampVals, freeSteps, null);
  const values = outputUnits(net, sFree);
  const arm = chooseFn(values);
  const reward = rewardFn(arm);
  const target = 2 * reward - 1; // map {0,1} reward -> {-1,1}
  const chosenUnitIndex = net.n - 2 + arm;

  const sNudge = settle(net, clampVals, nudgeSteps, { unitIndex: chosenUnitIndex, target, beta });

  // EqProp update: dW_ij ~ (1/beta)(sNudge_i sNudge_j - sFree_i sFree_j), free units only as "i"
  const { nClamp, n, W, b } = net;
  for (let i = nClamp; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = (sNudge[i] * sNudge[j] - sFree[i] * sFree[j]) / beta;
      W[i][j] = clip(W[i][j] + lr * d, -4, 4);
      W[j][i] = W[i][j];
    }
    const db = (sNudge[i] - sFree[i]) / beta;
    b[i] = clip(b[i] + lr * db, -4, 4);
  }

  return { arm, reward, values, predictedChosen: values[arm] };
}

// ---------- tiny deterministic RNG so runs are reproducible while testing ----------
function makeRng(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ---------- quick smoke test: does this agent learn a plain stationary bandit at all ----------
function smokeTest() {
  const rng = makeRng(42);
  const net = makeNetwork(1, 4, rng); // 1 clamp unit (constant go cue), 4 hidden
  const opts = { freeSteps: 25, nudgeSteps: 25, beta: 0.6, lr: 0.08 };

  let armACount = 0, totalReward = 0;
  const windowRewards = [];
  const N = 1500;
  for (let ep = 0; ep < N; ep++) {
    const clampVals = [1]; // constant cue
    const chooseFn = (values) => {
      // epsilon-greedy
      if (rng() < 0.08) return rng() < 0.5 ? 0 : 1;
      return values[0] >= values[1] ? 0 : 1;
    };
    const rewardFn = (arm) => {
      // arm 0 is the good one: p=0.8 vs p=0.2
      const p = arm === 0 ? 0.8 : 0.2;
      return rng() < p ? 1 : 0;
    };
    const { arm, reward } = eqpropEpisode(net, clampVals, chooseFn, rewardFn, opts);
    if (arm === 0) armACount++;
    totalReward += reward;
    windowRewards.push(reward);
    if (windowRewards.length > 50) windowRewards.shift();
  }

  const lastWindowAvg = windowRewards.reduce((a, b) => a + b, 0) / windowRewards.length;
  console.log("=== Smoke test: plain stationary bandit, arm0 p=0.8 vs arm1 p=0.2 ===");
  console.log(`Episodes: ${N}`);
  console.log(`Fraction choosing arm0 (the good arm): ${(armACount / N).toFixed(3)} (should trend toward ~0.9+ as it learns)`);
  console.log(`Overall avg reward: ${(totalReward / N).toFixed(3)}`);
  console.log(`Last-50-episode avg reward: ${lastWindowAvg.toFixed(3)} (should be notably higher than overall avg if it's learning)`);
}

if (require.main === module) {
  smokeTest();
}
module.exports = { makeNetwork, settle, outputUnits, eqpropEpisode, makeRng };
