# Correction to test_agency_gate_supersedes_bootstrap.js commit message

The original commit speculated that the combined mechanism's failures
(seeds 7777, 31337) were caused by "the bootstrap handoff firing at a
moment that looks stable but isn't robust."

Direct trace of seed 7777 through episode 3000 REFUTES this. The handoff
at episode 780 is correct in every way checked: committedArm=0 (the right
choice), val0=-0.251 > val1=-0.457 (correct direction), fastMom seeded at
0.85 (confident, correct). The speculative explanation does not hold.

What the trace actually shows: agency-gating-alone and the combined
mechanism fail the SAME trust test (the second one, episode 16000) - they
do not diverge there. The divergence is in what happens AFTER that
failure: agency-gating-alone recovers and passes the third test, reaching
confidant. The combined mechanism does not recover, and stays stuck.

The real question is about post-failure recovery dynamics, not the
bootstrap handoff. This has not yet been traced to a confirmed mechanism.
The result itself (combined: 75%, different failure pattern than
agency-gating-alone) stands as reported and verified - only the proposed
EXPLANATION for the specific failing seeds is retracted here.
