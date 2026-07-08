# fluorite-architecture

> *"My mission is to make people happy with my singing."*  
> — Vivy, *Vivy: Fluorite Eye's Song*

A sustained empirical research program into one question: **what does it actually take to build an AI that maintains coherent identity across long timescales while genuinely learning from experience?**

Motivated by Vivy — an android whose hundred-year mission requires persistent values, resistance to adversarial pressure, and real adaptation to genuine change — and grounded in twelve weeks of experiment-first research that refused to declare something true until it ran and produced numbers.

---

## What's in here

### `/paper`
- **`vivy_architecture.pdf`** — The main document. 25 pages. Everything found, everything that failed, the architecture proposal, cross-context values and their limits, independent validation against BDH, associative memory capacity, a correction to earlier momentum claims with a verified fix, and — the central finding — identity under coercion, not just temptation. Read this first.
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
| `test_relational_identity.js` | **Cross-context values**: momentum beats EMA (35,206 vs 32,859 reward), min(R) 2× higher. Slow layer hits 91–93% identity uniformly across all 4 contexts — the closest thing to a value this program has produced. |
| `test_coordinated_adversarial.js` | The limit of the above: coordinated pressure across all contexts simultaneously defeats momentum's advantage. EMA recovers faster from coordinated genuine change (0.81 vs 0.66). |
| `test_bdh_separated_hebbian.js` | BDH-inspired fix to Wall 1 attempt #1: separate the Hebbian matrix from gradient-trained weights. Failed — the problem is closed-loop reward coupling, not shared object identity. Includes a caught-and-fixed bug (H computed but never wired into the decision). |
| `test_oja_hebbian.js` | Verified against Oja (1982): pure Hebbian learning is provably unstable (Δw ∝ w·x²). Oja-stabilized version is the first Hebbian-family mechanism in this program to be net-positive vs plain EqProp (10/16 cells). |
| `test_gated_oja.js` | Attempt to gate the Oja channel with the commit threshold. Appeared to fail catastrophically — but the failure was a methodological confound (gated updates, but reported unprotected raw choice). Retracted rather than reported as a finding. |
| `test_cs_agent_oja.js` | The corrected comparison: commit-threshold agent alone already gets 0.00–0.06 spurious switches. Adding the Oja channel is a ceiling effect — 14/16 tied, slight reward cost, no room left to improve. **Conclusion: the commit threshold IS the fix for Wall 1, not one option alongside a Hebbian one.** |
| `test_hopfield_capacity.js` | Direct verification of Ramsauer et al. (2020): classical Hebbian storage collapses exactly at the predicted ~0.138×D boundary (100%→0% exact retrieval, K=3→14). Modern (softmax) Hopfield holds 100% retrieval all the way to K=64 in the same space. Explains *why* the BDH ceiling effect happened — our toy channel was classical/scalar, nothing like BDH's actual attention-shaped mechanism. |
| `audit_relational_identity.js` | Self-audit: does the earlier cross-context values result hide bootstrap lock-in too? Checked per-context, not just the aggregate. Cleared — all 24 context-instances locked correctly, 0.957–0.964. |
| `test_accumulated_reputation.js` | Testing an unbounded environmental memory against momentum's bounded one. First version found nothing (stable policies make bounded/unbounded memory equivalent). Adding a forced-coercion window surfaced something more important — see below. |
| `test_agency_gated_momentum.js` | **The central finding.** Fix and full verification: 6/6 seeds, unprotected momentum permanently captures forced behavior as identity (reputation stays at 0.000 forever); agency-gated momentum (updates only when the action agrees with the agent's own uncoerced judgment) recovers 6/6 seeds to ~1.000 immediately once free choice resumes. |

---

## The central finding, updated: identity should not update on what you were forced to do

Every temptation/probe result in this repo tests whether identity resists being *chosen away from*. `test_agency_gated_momentum.js` tests something closer to the actual dramatic question Vivy's story asks: does identity survive being **forced** into behavior that contradicts it. Found by accident while testing an unrelated hypothesis about memory horizons — a scripted 5,000-episode window forcing shallow engagement, then releasing the agent back to free choice.

Unprotected momentum absorbed the forced behavior as if it were genuinely chosen, and never recovered — 6 of 6 seeds ended at reputation 0.000, permanently, for the remaining 40,000+ episodes. This despite the agent's own underlying value estimates never once stopping being correct: 5,000 episodes after the coercion ended, the learned preference for the right choice was still unambiguous (0.679 vs −0.882). The agent knew better the entire time. It acted on the corrupted identity anyway, because momentum's bonus term was large enough to override its own correct judgment.

The fix: gate momentum's update on whether the chosen action agrees with what the agent's own uncoerced values already preferred. If the action was imposed against that judgment, momentum doesn't update — the behavior is recognized as not genuinely chosen. 6 of 6 seeds recovered to ~1.000, immediately, once free choice returned.

**This is not automatic in any mechanism validated elsewhere in this repo.** It requires being built in specifically. Full details in Section 16 of the paper.
| `test_relationship_staging.js` | **Correction found here**: a multi-stage task where past behavior gates future options, not just reward magnitude. Surfaced a previously undiscovered momentum vulnerability — see correction notice below. |
| `test_protected_momentum.js` | The verified fix for that vulnerability: protect momentum's bootstrap with commit-threshold logic until a baseline stabilizes, then hand off. 100% pass rate (up from 38%), matching commit-threshold exactly. |

---

## ⚠️ Correction (v3): momentum has a bootstrap vulnerability under ambiguous starting conditions

Every earlier momentum result in this repo (`test_momentum.js`, `test_hierarchical_momentum.js`, `test_relational_identity.js`) used a **large, unambiguous reward margin** (0.78–0.80 vs 0.20–0.22) from episode zero. Under that condition, momentum reliably bootstraps toward the correct identity.

`test_relationship_staging.js` used a more realistic **ambiguous** initial margin (0.50 vs 0.35 — a new relationship doesn't announce its trustworthiness on day one) and found that momentum's self-reinforcing bonus locks onto the *wrong* arm roughly 50–62% of the time, permanently, before any adversarial pressure ever occurs. Controlled proof: the same random seed, with only the margin changed, produces `fastMom = 0.012` (locked wrong) vs `fastMom = 0.935` (locked right) by episode 150 — the entire outcome is decided in the first ~150 episodes, purely by how ambiguous the initial signal is.

This is the same structural signature as the Oja instability (Section 13 of the paper): any mechanism whose output feeds its own update can converge to *a* stable attractor without it being the *correct* one. `test_protected_momentum.js` verifies the fix: run commit-threshold's clean, self-correcting bootstrap first, hand off to momentum only once a baseline is confirmed stable. Every previously-failing seed now passes 100%.

**The earlier momentum claims are not wrong — they are correctly reported for the condition tested, which was not the realistic one.** Full details in Section 15 of the paper.

---

## The central finding

**In any environment where switching identity costs anything** — trust erosion, relationship damage, accumulated expertise that resets on switching — a commit-threshold behavioral architecture outperforms pure plasticity decisively. The crossover is at switch cost = 1 episode of reward. The standard symmetric bandit (switch cost = 0) is the single degenerate case where pure plasticity wins.

**The EqProp-reservoir synthesis works.** A fixed random reservoir with only a 24-parameter EqProp-trained readout beats a fully-trainable network by 40% in a relational environment. Two communities (EqProp and physical reservoir computing) solving the identical constraint — no backward pass through a physical substrate — with no cross-citations. The gap is real, the synthesis validates in software, the hardware version hasn't been built.

**The retention/adaptability tradeoff is structural, not parametric.** A values layer strong enough to protect identity is strong enough to block genuine changes from reaching the gate condition. The resolution theorem: this cannot be resolved faster than the timescale needed to distinguish the longest possible probe from a permanent genuine change. External confirmation — something outside the system that knows ground truth — is the only solution. This is why Vivy's identity is confirmed and protected by external human relationships, not internal architecture alone.

**Cross-context values emerge, but only against local pressure.** In a 4-context relational task with a cross-context consistency bonus, the momentum architecture developed a uniform 91–93% identity strength in every context simultaneously — the closest thing to a genuine value this program has produced, since it generalizes across situations rather than optimizing each one separately. But coordinated pressure across all contexts at once defeats this advantage entirely (EMA recovers from coordinated genuine change 0.81 vs momentum's 0.66) — cross-context comparison cannot be the external ground-truth signal the resolution theorem requires, since coordinated pressure is exactly what defeats that comparison.

**Independent validation from BDH (Pathway, 2025).** The Dragon Hatchling architecture's own authors, working independently at far greater scale, confirm verbatim that bridging fast Hebbian state to genuine long-term memory is unsolved — external validation of the resolution theorem from the field's actual frontier. A 5-experiment arc testing BDH-inspired fixes to the Hebbian/gradient conflict (Wall 1) found that the already-validated commit-threshold mechanism doesn't need a Hebbian supplement — it already solves the problem alone (0.00–0.06 spurious switches at every tested length), making an Oja-stabilized Hebbian channel redundant on this task despite being a genuine, verified improvement over a weaker baseline.

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
