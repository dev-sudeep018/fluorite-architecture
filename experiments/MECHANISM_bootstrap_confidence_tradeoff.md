# The actual mechanism (verified, replacing the retracted speculation)

Direct side-by-side trace of seed 7777 at the second trust test (episode
16000), agency-gating-alone vs the combined mechanism:

| | fastMom at ep16000 | val0 vs val1 at ep16000 |
|---|---|---|
| Agency-gating alone | 1.000 | 0.486 vs 0.436 (agrees) |
| Combined | 0.578 | -0.312 vs 0.546 (disagrees) |

**Why the gap:** in agency-gating-alone, the momentum bonus feeds into
the CHOICE from episode 0, even though its UPDATE is gated. This creates
a filtered self-reinforcing loop - bonus nudges choice, gate only lets
correct-direction reinforcement through, confidence climbs to 1.0
relatively fast (well before ep16000).

The combined mechanism's bootstrap phase deliberately uses ZERO
bonus-to-choice coupling (pure value-based choice) specifically to avoid
wrong-direction lock-in under ambiguous conditions - the right design
choice for safety. But this means momentum only starts accumulating
confidence from the handoff point (ep780) onward, at the same slow gated
rate, with no accelerating loop. By ep16000 it's only moderately
confident (0.578), not fully locked (1.000) - and moderate confidence
isn't enough resistance against a real, large, one-off trust-test
temptation (reward=5.0). Full confidence is.

**This is a genuine, verified tradeoff, not a bug.** The mechanism that
keeps protected-bootstrap safe during ambiguous early conditions (no
self-reinforcing bonus-choice coupling during bootstrap) is the same one
that makes it converge more slowly to maximum resistance afterward. Both
sides of this tradeoff are now precisely understood and traced, not
guessed.
