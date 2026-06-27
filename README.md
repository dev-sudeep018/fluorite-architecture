# fluorite-architecture

> *"My mission is to make people happy with my singing."*  
> — Vivy, *Vivy: Fluorite Eye's Song*

A sustained empirical research program into one question: **what does it actually take to build an AI that maintains coherent identity across long timescales while genuinely learning from experience?**

Motivated by Vivy — an android whose hundred-year mission requires persistent values, resistance to adversarial pressure, and real adaptation to genuine change — and grounded in twelve weeks of experiment-first research that refused to declare something true until it ran and produced numbers.

---

## What's in here

### `/paper`
- **`vivy_architecture.pdf`** — The main document. 15 pages. Everything found, everything that failed, the architecture proposal, and where the gap actually is. Read this first.
- **`experimental_findings.md`** — Running research notes with precise failure-mode analysis for each mechanism tested.

### `/experiments`
All code is self-contained Node.js. No dependencies beyond Node itself (v16+). Every file can be run with `node <filename>`.

| File | What it tests |
|------|--------------|
| `eqprop_core.js` | Core EqProp/Hopfield engine. Validated independently before any experiment uses it. |
| `test_memory.js` | Hopfield-attention memory mechanism (h). Found: signal informationally empty — mean 0.49 across all episode types. |
| `test_h_diagnostic.js` | Confirms h can't distinguish probe from genuine change at the raw signal level. |
| `bandit_causal_state_test.html` | Full interactive bandit experiment in the browser. Three agents, live Chart.js visualizations. Open in any browser. |
| `test_shadow_v2.js` | P_revert table with shadow tracker. Found: table saturates because everything reverts; scalar P_revert is wrong quantity. |
| `test_hebbian_eslow.js` | Hebbian co-activation as E_slow (MICrONS like-to-like). Attractor depth: 0.97 vs 0.57. Spurious switches: worse, not better. |
| `test_precision_gate.js` | Reward-rate gate on Hebbian. Failed: reward signal contaminated by probes. |
| `test_critical_period.js` | Burn-in pre-wiring + energy gate. Confirmed: post-probe recovery is failure mode, not the probe itself. |
| `test_bocpd.js` | Bayesian Online Changepoint Detection. Scalar pChange can't represent "change happened and reversed." |
| `test_asymmetric_cost.js` | **Key finding**: EMA makes 13,961 switches vs hardcoded's 154. Crossover at switch cost = 1. |
| `test_momentum.js` | Behavioral momentum (endogenous identity). 93.5% identity strength, 3-ep probe leaves trace of 0.00066. |
| `test_hierarchical_momentum.js` | Two-timescale momentum. Fast (α=0.03) and slow (α=0.0003) layers. 100× separation empirically confirmed. |
| `test_reservoir_synthesis.js` | **EqProp + fixed reservoir**: reservoir-only beats standard EqProp by 40%. Full architecture beats by 90%. |
| `test_forgetting.js` | Catastrophic forgetting benchmark. Hierarchical: retention=0.96, Task-B=0.04. The tradeoff located precisely. |
| `test_gated_consolidation.js` | Gated slow layer update. Gate never opens: strong identity prevents gate condition from being reached. |
| `test_pe_gate.js` | Prediction-error gate. Failed: stochastic bandit PE permanently exceeds any calibrated threshold. |
| `test_confirmatory.js` | Two-agent confirmatory architecture (CLS theory). Task-B=0.96, retention=0.03. Gate opens too fast. |
| `test_learned_commitment.js` | Adaptive commit threshold from duration statistics. Self-entanglement prevents recalibration. |
| `test_oracle_shadow.js` | Oracle shadow tracker with perfect duration statistics. Still loses to EMA on symmetric bandit. |

---

## The central finding

**In any environment where switching identity costs anything** — trust erosion, relationship damage, accumulated expertise that resets on switching — a commit-threshold behavioral architecture outperforms pure plasticity decisively. The crossover is at switch cost = 1 episode of reward. The standard symmetric bandit (switch cost = 0) is the single degenerate case where pure plasticity wins.

**The EqProp-reservoir synthesis works.** A fixed random reservoir with only a 24-parameter EqProp-trained readout beats a fully-trainable network by 40% in a relational environment. Two communities (EqProp and physical reservoir computing) solving the identical constraint — no backward pass through a physical substrate — with no cross-citations. The gap is real, the synthesis validates in software, the hardware version hasn't been built.

**The retention/adaptability tradeoff is structural, not parametric.** A values layer strong enough to protect identity is strong enough to block genuine changes from reaching the gate condition. The resolution theorem: this cannot be resolved faster than the timescale needed to distinguish the longest possible probe from a permanent genuine change. External confirmation — something outside the system that knows ground truth — is the only solution. This is why Vivy's identity is confirmed and protected by external human relationships, not internal architecture alone.

---

## Running the experiments

```bash
# Install Node.js (v16+), then:
cd experiments

# Core engine smoke test
node eqprop_core.js

# Asymmetric cost finding (the main result)
node test_asymmetric_cost.js

# EqProp-reservoir synthesis
node test_reservoir_synthesis.js

# Catastrophic forgetting
node test_forgetting.js

# Open in browser for live interactive experiment
open bandit_causal_state_test.html
```

All experiments are fully deterministic given the seed. Results in the paper can be reproduced exactly.

---

## The architecture proposal

Three layers, three timescales, three different problems:

```
Layer 1 — E_fast (EqProp + fixed reservoir)
  Per-episode learning from immediate experience.
  Validated: 40% better than fully-trainable on relational task.

Layer 2 — Commitment (hierarchical momentum + threshold)
  Fast sub-layer (α=0.03): preferences
  Slow sub-layer (α=0.0003): values
  Validated: trust=1.000, 1 switch in 40k episodes.

Layer 3 — E_slow (physical reservoir + like-to-like wiring)
  Structural organization from co-activation statistics.
  From MICrONS connectomics: neurons with similar response properties
  preferentially connect. Implemented in software. Hardware version: unbuilt.
```

The missing fourth layer: **embodied grounding**. Values that aren't grounded in physical relational experience aren't values — they're arm preferences with consistency bonuses. This is the Vivy gap that the architecture doesn't close.

---

## What this is not

- This is not a production system
- This is not a trained language model
- The experiments are 4-12 hidden unit networks on a two-armed bandit
- Scale results are not claimed, only structural principles
- The question of whether there's something it's like to be these systems is not addressed, because it can't be

---

## Context

Built by Sudeep (first-year B.Tech, NIAT Hyderabad) and Claude Sonnet 4.6, June 2026. Connected to Sudeep's earlier work on [Aether-Link](https://github.com/dev-sudeep018) (haptic feedback framework with neuroscience grounding) and [CausalLens](https://github.com/dev-sudeep018/casullens) (causal graph extraction middleware).

The goal is Vivy. We're at Layer 1 of maybe 6.
