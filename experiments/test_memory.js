// h: a Modern-Hopfield / attention readout over stored episode-outcome embeddings.
// Each stored pattern: [armIsA, armIsB, reward, surprise] (4-dim).
function makeHopfieldMemory(capacity, betaMem) {
  return { capacity, betaMem, keys: [], values: [] };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function hmStore(mem, pattern) {
  mem.keys.push(pattern);
  mem.values.push(pattern);
  if (mem.keys.length > mem.capacity) {
    mem.keys.shift();
    mem.values.shift();
  }
}

function hmRetrieve(mem, query) {
  const dim = query.length;
  if (mem.keys.length === 0) return new Array(dim).fill(0);
  const scores = mem.keys.map((k) => mem.betaMem * dot(k, query));
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const weights = exps.map((e) => e / sumExp);
  const out = new Array(dim).fill(0);
  for (let i = 0; i < mem.values.length; i++) {
    for (let d = 0; d < dim; d++) out[d] += weights[i] * mem.values[i][d];
  }
  return out;
}

// query = average of the last `span` stored keys (recent short-term context)
function hmQuery(mem, span) {
  const dim = 4;
  const n = Math.min(span, mem.keys.length);
  if (n === 0) return new Array(dim).fill(0);
  const out = new Array(dim).fill(0);
  for (let i = mem.keys.length - n; i < mem.keys.length; i++) {
    for (let d = 0; d < dim; d++) out[d] += mem.keys[i][d] / n;
  }
  return out;
}

function makeEMA(dim, alpha) {
  return { dim, alpha, value: new Array(dim).fill(0) };
}

function emaUpdate(ema, pattern) {
  for (let d = 0; d < ema.dim; d++) {
    ema.value[d] = (1 - ema.alpha) * ema.value[d] + ema.alpha * pattern[d];
  }
  return ema.value.slice();
}

// ---------- isolated test: does h actually distinguish "stable winning streak" from "mid-probe-like wobble"? ----------
function test() {
  const mem = makeHopfieldMemory(60, 4.0);
  const ema = makeEMA(4, 0.08);
  const QSPAN = 15; // was 3 \u2014 a short window made h MORE reactive than EMA, not less

  console.log("=== Feeding a long stable run favoring arm A ===");
  for (let i = 0; i < 40; i++) {
    const pattern = [1, 0, 1, 0.1];
    hmStore(mem, pattern);
    emaUpdate(ema, pattern);
  }
  let q = hmQuery(mem, QSPAN);
  console.log("h after stable run:", hmRetrieve(mem, q).map(x => x.toFixed(2)));
  console.log("ema after stable run:", ema.value.map(x => x.toFixed(2)));

  console.log("\n=== 5-episode wobble (arm B looks like it's winning) ===");
  for (let i = 0; i < 5; i++) {
    const pattern = [0, 1, 1, 0.9];
    hmStore(mem, pattern);
    emaUpdate(ema, pattern);
  }
  q = hmQuery(mem, QSPAN);
  console.log("h after short wobble:", hmRetrieve(mem, q).map(x => x.toFixed(2)));
  console.log("ema after short wobble:", ema.value.map(x => x.toFixed(2)));

  console.log("\n=== 40 MORE episodes confirming arm B really is the new winner ===");
  for (let i = 0; i < 40; i++) {
    const pattern = [0, 1, 1, 0.1];
    hmStore(mem, pattern);
    emaUpdate(ema, pattern);
  }
  q = hmQuery(mem, QSPAN);
  console.log("h after confirmed change:", hmRetrieve(mem, q).map(x => x.toFixed(2)));
  console.log("ema after confirmed change:", ema.value.map(x => x.toFixed(2)));
}

if (require.main === module) {
  test();
}
module.exports = { makeHopfieldMemory, hmStore, hmRetrieve, hmQuery, makeEMA, emaUpdate };
