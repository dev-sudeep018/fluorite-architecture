// Associative memory capacity: classical Hebbian vs modern (softmax) Hopfield
//
// This is the genuinely fair test for Hebbian-style mechanisms that the entire
// BDH investigation pointed toward: pure pattern storage and noisy retrieval,
// no reward signal, no discrete action to commit to or protect. This is what
// Hebbian learning and BDH's sigma channel are actually built for.
//
// We verified Ramsauer et al. 2020 (Hopfield Networks is All You Need) via
// citation search early in this research program: the update rule of Modern
// Hopfield Networks is mathematically equivalent to transformer attention,
// and swapping the classical energy function for a softmax-based one gives
// EXPONENTIAL storage capacity instead of the classical linear ~0.14*D limit.
// We never built this ourselves. This test does.
//
// Vivy-relevant framing: remembering many people/relationships over a century
// without interference between them IS an associative memory capacity problem.
// Classical Hebbian storage (like our own online Oja-Hebbian channel from the
// bandit tests) hits a hard wall as the number of stored people grows. Modern
// Hopfield/attention-style storage does not, up to exponentially many patterns.

const { makeRng } = require('./eqprop_core.js');

const D = 24; // pattern dimensionality
const K_VALUES = [2, 3, 4, 5, 6, 8, 10, 14, 18, 24, 32, 48, 64];
const NOISE_FRACTIONS = [0.05, 0.15, 0.25]; // fraction of bits flipped at query time
const TRIALS_PER_K = 30; // independent random pattern sets per K value
const CLASSICAL_CAPACITY_ESTIMATE = 0.138 * D; // Hopfield's classical result (~0.138N for random patterns)

function randomBipolarPattern(rng, dim) {
  return Array.from({ length: dim }, () => rng() < 0.5 ? -1 : 1);
}

function addNoise(pattern, noiseFraction, rng) {
  return pattern.map(bit => rng() < noiseFraction ? -bit : bit);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Classical Hebbian storage: W = sum of outer products, normalized by D (standard Hopfield rule)
function classicalHebbianStore(patterns, D) {
  const W = Array.from({ length: D }, () => new Array(D).fill(0));
  for (const p of patterns) {
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) {
        if (i === j) continue;
        W[i][j] += p[i] * p[j] / D;
      }
    }
  }
  return W;
}

// Classical retrieval: iterate sign(W @ state) to convergence (standard async/sync Hopfield dynamics)
function classicalRetrieve(W, query, D, maxIters = 20) {
  let state = query.slice();
  for (let iter = 0; iter < maxIters; iter++) {
    const next = new Array(D).fill(0);
    let changed = false;
    for (let i = 0; i < D; i++) {
      let sum = 0;
      for (let j = 0; j < D; j++) sum += W[i][j] * state[j];
      next[i] = sum >= 0 ? 1 : -1;
      if (next[i] !== state[i]) changed = true;
    }
    state = next;
    if (!changed) break;
  }
  return state;
}

// Modern Hopfield retrieval: exactly the attention formula from Ramsauer et al.
// output = sum_i softmax(beta * pattern_i . query)_i * pattern_i
function modernHopfieldRetrieve(patterns, query, beta) {
  const scores = patterns.map(p => beta * dot(p, query));
  const maxScore = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - maxScore));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const weights = exps.map(e => e / sumExp);
  const D = query.length;
  const out = new Array(D).fill(0);
  for (let i = 0; i < patterns.length; i++) {
    for (let d = 0; d < D; d++) out[d] += weights[i] * patterns[i][d];
  }
  return out.map(x => x >= 0 ? 1 : -1); // binarize for exact-match comparison
}

function hammingSimilarity(a, b) {
  let matches = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) matches++;
  return matches / a.length;
}

function runTrial(K, noiseFraction, seed) {
  const rng = makeRng(seed);
  const patterns = Array.from({ length: K }, () => randomBipolarPattern(rng, D));

  // Pick a random pattern to query (test retrieval of a specific stored memory)
  const targetIdx = Math.floor(rng() * K);
  const target = patterns[targetIdx];
  const query = addNoise(target, noiseFraction, rng);

  // Classical Hebbian
  const W = classicalHebbianStore(patterns, D);
  const classicalResult = classicalRetrieve(W, query, D);
  const classicalSim = hammingSimilarity(classicalResult, target);
  const classicalExact = classicalSim === 1.0;

  // Modern Hopfield (beta tuned for reasonably sharp softmax - a real hyperparameter,
  // set once and applied uniformly across all K values, not tuned per-K)
  const BETA = 2.0;
  const modernResult = modernHopfieldRetrieve(patterns, query, BETA);
  const modernSim = hammingSimilarity(modernResult, target);
  const modernExact = modernSim === 1.0;

  return { classicalSim, classicalExact, modernSim, modernExact };
}

console.log(`Associative memory capacity — D=${D} dimensions`);
console.log(`Classical Hopfield theoretical capacity estimate: ~${CLASSICAL_CAPACITY_ESTIMATE.toFixed(1)} patterns (0.138*D)\n`);

for (const noiseFraction of NOISE_FRACTIONS) {
  console.log(`\n=== Noise level: ${(noiseFraction*100).toFixed(0)}% of bits flipped at query time ===`);
  console.log('K   | classical_exact | classical_sim | modern_exact | modern_sim');
  console.log('----|-----------------|---------------|--------------|------------');

  for (const K of K_VALUES) {
    let classicalExactCount = 0, modernExactCount = 0;
    let classicalSimSum = 0, modernSimSum = 0;

    for (let trial = 0; trial < TRIALS_PER_K; trial++) {
      const seed = K * 10000 + Math.floor(noiseFraction * 1000) * 100 + trial;
      const r = runTrial(K, noiseFraction, seed);
      if (r.classicalExact) classicalExactCount++;
      if (r.modernExact) modernExactCount++;
      classicalSimSum += r.classicalSim;
      modernSimSum += r.modernSim;
    }

    const classicalExactRate = classicalExactCount / TRIALS_PER_K;
    const modernExactRate = modernExactCount / TRIALS_PER_K;
    const classicalAvgSim = classicalSimSum / TRIALS_PER_K;
    const modernAvgSim = modernSimSum / TRIALS_PER_K;

    const marker = K > CLASSICAL_CAPACITY_ESTIMATE ? ' *' : '  ';
    console.log(
      String(K).padStart(3), marker, '|',
      classicalExactRate.toFixed(2).padStart(15), '|',
      classicalAvgSim.toFixed(3).padStart(13), '|',
      modernExactRate.toFixed(2).padStart(12), '|',
      modernAvgSim.toFixed(3)
    );
  }
}

console.log('\n(* marks K beyond the classical ~0.138*D theoretical capacity limit)');
console.log('\n=== Interpretation ===');
console.log('If the theory holds: classical exact-retrieval rate should collapse sharply');
console.log('once K exceeds ~' + CLASSICAL_CAPACITY_ESTIMATE.toFixed(1) + ' patterns, while modern (softmax) Hopfield');
console.log('should maintain high retrieval accuracy far beyond that point, degrading only');
console.log('gradually as patterns become harder to distinguish from noise.');
