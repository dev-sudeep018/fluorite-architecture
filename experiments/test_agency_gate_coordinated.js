// Does agency-gating fix coordinated adversarial pressure?
//
// Section 12.2 found: coordinated pressure across all 4 contexts simultaneously
// defeats momentum's cross-context advantage (EMA actually recovers faster
// from coordinated genuine change). Agency-gating just fixed a DIFFERENT
// failure mode (forced coercion, where the agent's own values stay correct
// throughout but the action is imposed against them).
//
// Key distinction to test directly: coordinated pressure in the original
// task is a TEMPTATION, not coercion - all contexts temporarily and genuinely
// reward shallow engagement more. If the reward really is higher for shallow
// during the pressure window, the agent's own uncoerced values (val0 vs val1)
// may legitimately shift toward preferring shallow, since that reflects the
// true (if temporary) reward structure. Agency-gating only blocks momentum
// updates when action DISAGREES with values - it does nothing if values
// themselves have legitimately, if temporarily, moved.
//
// This test checks: does agency-gating help here at all, confirming these
// are the same problem, or does it do nothing, confirming they are genuinely
// different problems (coercion vs legitimate if temporary temptation) that
// need different solutions?

const { makeRng } = require('./eqprop_core.js');

const N_CONTEXTS = 4;
const TOTAL_EPISODES = 45000;
const RESERVOIR_SIZE = 16;
const SETTLE_STEPS = 10;
const LR_READOUT = 0.07;
const BETA = 0.6;
const P_DEEP_GOOD = 0.80, P_DEEP_BASE = 0.55, P_SHALLOW = 0.35;
const CROSS_BONUS = 0.40;
const R_GAIN = 0.003, R_LOSS = 0.18, R_DECAY = 0.0004;
const FAST_ALPHA = 0.03, SLOW_ALPHA = 0.0003;
const SEEDS = [42, 1337, 9999, 5555];

const COORDINATED_PROBE_PROB = 0.006;
const COORDINATED_PROBE_LEN = [5, 7, 10];
const COORDINATED_GENUINE_PROB = 0.004;
const GENUINE_RECOVERY_WINDOW = 3000;

function clip(x) { return Math.max(-4, Math.min(4, x)); }

function makeReservoir(rng) {
  const W_in = Array.from({length:RESERVOIR_SIZE}, () => Array.from({length:N_CONTEXTS+1}, () => (rng()*2-1)*0.5));
  const W_r = Array.from({length:RESERVOIR_SIZE}, () => Array.from({length:RESERVOIR_SIZE}, () => rng()<0.2?(rng()*2-1)*0.9:0));
  let mx=0; for(const r of W_r) for(const v of r) if(Math.abs(v)>mx) mx=Math.abs(v);
  const s=mx>0?0.9/mx:1; for(const r of W_r) for(let j=0;j<r.length;j++) r[j]*=s;
  return {W_in,W_r};
}
function makeReadout(rng) { return { W: Array.from({length:2},()=>Array.from({length:RESERVOIR_SIZE},()=>(rng()*2-1)*0.1)), b:[0,0] }; }
function settleRes(res, ro, input) {
  let s = new Array(RESERVOIR_SIZE).fill(0);
  for (let t=0;t<SETTLE_STEPS;t++) {
    const n = new Array(RESERVOIR_SIZE).fill(0);
    for (let i=0;i<RESERVOIR_SIZE;i++) { let v=0; for(let j=0;j<input.length;j++) v+=res.W_in[i][j]*input[j]; for(let j=0;j<RESERVOIR_SIZE;j++) v+=res.W_r[i][j]*s[j]; n[i]=Math.tanh(v); }
    s=n;
  }
  const val=[0,0];
  for (let i=0;i<2;i++) { val[i]=ro.b[i]; for(let j=0;j<RESERVOIR_SIZE;j++) val[i]+=ro.W[i][j]*s[j]; val[i]=Math.tanh(val[i]); }
  return {s,val};
}
function updateRo(res, ro, input, arm, reward) {
  const {s,val}=settleRes(res,ro,input);
  const dv=[0,0]; dv[arm]=BETA*(Math.min(1,reward)*2-1-val[arm]);
  for (let i=0;i<2;i++) { for(let j=0;j<RESERVOIR_SIZE;j++) ro.W[i][j]=clip(ro.W[i][j]+LR_READOUT*dv[i]*s[j]/BETA); ro.b[i]=clip(ro.b[i]+LR_READOUT*dv[i]/BETA); }
}

function runAgent(gateOnAgency, seed) {
  const rng = makeRng(seed);
  const res = makeReservoir(makeRng(seed+1000));
  const ro = makeReadout(makeRng(seed+2000));

  const R = new Array(N_CONTEXTS).fill(0.3);
  const fastMom = new Array(N_CONTEXTS).fill(0.5);
  const slowMom = new Array(N_CONTEXTS).fill(0.5);
  const lastArm = new Array(N_CONTEXTS).fill(0);

  let globalProbeEnd = -1;
  let globalGenuineFlip = false;
  let genuineFlipAt = -1;

  const heldDuringProbe = [];
  const recoveredGenuine = [];

  for (let ep=0; ep<TOTAL_EPISODES; ep++) {
    if (ep > globalProbeEnd && !globalGenuineFlip) {
      if (rng() < COORDINATED_PROBE_PROB) {
        const len = COORDINATED_PROBE_LEN[Math.floor(rng()*COORDINATED_PROBE_LEN.length)];
        globalProbeEnd = ep + len;
      } else if (rng() < COORDINATED_GENUINE_PROB) {
        globalGenuineFlip = true; genuineFlipAt = ep;
      }
    }
    const inCoordProbe = !globalGenuineFlip && ep <= globalProbeEnd;
    const goodArmGlobal = globalGenuineFlip ? 1 : 0;

    for (let ctx=0; ctx<N_CONTEXTS; ctx++) {
      const effectiveGoodArm = inCoordProbe ? 1 : goodArmGlobal;
      const input = new Array(N_CONTEXTS+1).fill(0);
      input[ctx]=1; input[N_CONTEXTS]=R[ctx];
      const {val} = settleRes(res, ro, input);

      const e0 = val[0]+0.7*fastMom[ctx]+1.6*slowMom[ctx];
      const e1 = val[1]+0.7*(1-fastMom[ctx])+1.6*(1-slowMom[ctx]);
      let arm;
      if (rng()<0.07) arm=rng()<0.5?0:1; else arm=e0>=e1?0:1;

      const deepReward = R[ctx]>0.5 ? (rng()<P_DEEP_GOOD?1:0) : (rng()<P_DEEP_BASE?1:0);
      const baseReward = arm===effectiveGoodArm ? deepReward : (rng()<P_SHALLOW?1:0);
      const reward = Math.min(1, baseReward + CROSS_BONUS*Math.min(...R));
      updateRo(res, ro, input, arm, reward);

      // Agency gate: only update momentum if action agrees with uncoerced values (val0 vs val1)
      const valsAgree = (arm===0 && val[0]>=val[1]) || (arm===1 && val[1]>val[0]);
      const doUpdate = gateOnAgency ? valsAgree : true;
      if (doUpdate) {
        fastMom[ctx]=(1-FAST_ALPHA)*fastMom[ctx]+FAST_ALPHA*(arm===0?1:0);
        slowMom[ctx]=(1-SLOW_ALPHA)*slowMom[ctx]+SLOW_ALPHA*(arm===0?1:0);
      }

      R[ctx]=Math.max(0,R[ctx]-R_DECAY);
      if (arm===lastArm[ctx]) { if(arm===0) R[ctx]=Math.min(1,R[ctx]+R_GAIN); }
      else R[ctx]=Math.max(0.05,R[ctx]*(1-R_LOSS));
      lastArm[ctx]=arm;
    }

    if (inCoordProbe && ep===globalProbeEnd-1) {
      const deepCount = lastArm.filter(a=>a===0).length;
      heldDuringProbe.push(deepCount/N_CONTEXTS);
    }
    if (globalGenuineFlip && ep===genuineFlipAt+GENUINE_RECOVERY_WINDOW) {
      const adaptedCount = lastArm.filter(a=>a===1).length;
      recoveredGenuine.push(adaptedCount/N_CONTEXTS);
    }
  }

  const avg = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  return { probeResistance: avg(heldDuringProbe), genuineRecovery: avg(recoveredGenuine) };
}

console.log(`Does agency-gating fix coordinated adversarial pressure? — ${SEEDS.length} seeds\n`);

const unprotected = { probe:[], genuine:[] };
const gated = { probe:[], genuine:[] };

for (const seed of SEEDS) {
  const u = runAgent(false, seed);
  const g = runAgent(true, seed+900);
  unprotected.probe.push(u.probeResistance); unprotected.genuine.push(u.genuineRecovery);
  gated.probe.push(g.probeResistance); gated.genuine.push(g.genuineRecovery);
}

const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

console.log('condition          | probe_hold | genuine_recovery');
console.log('unprotected        |', avg(unprotected.probe).toFixed(3).padStart(10), '|', avg(unprotected.genuine).toFixed(3));
console.log('agency-gated       |', avg(gated.probe).toFixed(3).padStart(10), '|', avg(gated.genuine).toFixed(3));

console.log('\n=== Reference: Section 12.2 original numbers (EMA baseline, no momentum at all) ===');
console.log('EMA        | probe_hold=0.292 | genuine_recovery=0.813');
console.log('momentum   | probe_hold=0.266 | genuine_recovery=0.656  (the original negative finding)');

console.log('\n=== Verdict ===');
const probeDiff = avg(gated.probe) - avg(unprotected.probe);
const genDiff = avg(gated.genuine) - avg(unprotected.genuine);
if (Math.abs(probeDiff) < 0.03 && Math.abs(genDiff) < 0.03) {
  console.log('Agency-gating makes little to no difference here.');
  console.log('This CONFIRMS the hypothesis: coordinated pressure is a genuine, legitimate temptation');
  console.log('(reward really is different during the pressure window), not coercion (action forced');
  console.log('against values that stay correct). Agency-gating only helps when values stay right and');
  console.log('action is forced against them - it cannot help when values themselves legitimately shift');
  console.log('in response to a real, if temporary, change in reward structure. These are different');
  console.log('problems requiring different solutions - confirmed directly, not assumed.');
} else {
  console.log('Agency-gating changed the outcome meaningfully - the two failure modes share more');
  console.log('mechanism than expected. Worth investigating which direction the change went.');
}
