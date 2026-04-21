# Idea Log

Undecided possibilities — things worth remembering but not yet decided.
Promote to an ADR when ready to decide; discard when no longer relevant.

---

## 2026-04-20 — Two-track drools-beliefs: Java 21 LTS stable + Java 27 Valhalla experimental

**Priority:** high  
**Status:** active

Maintain two parallel implementations of the drools-beliefs Bayesian inference
engine. Track 1 targets Java 21 LTS (incremental inference Phases 1–3 already
done, virtual threads for parallel distribute, Vector API opt-in incubator).
Track 2 targets Java 27+ and exploits Valhalla value classes — `CliqueState`,
`SeparatorState`, `BayesVariableState` as flat-layout value objects (JEP 401,
first preview est. Sept 2026), finalized Vector API, and scalarized clique
potentials with zero heap allocation per update. Expected 3–5× improvement over
Track 1 on arithmetic-heavy paths.

The two tracks are warranted because Valhalla value classes have API
restrictions (no identity, no synchronized, different null semantics) that are
not backward-compatible. Track 2 proves the performance thesis against live
benchmarks and feeds back into Track 1 once Valhalla stabilises (est. Java
28–29 for production). Current benchmark context: Java 25 JVM delivers ~8%
free improvement over Java 17 with no code changes; Vector API + Valhalla
together likely deliver 3–5× on the hot paths.

**Context:** Arose from Java platform feature analysis during drools-beliefs
incremental inference work. Benchmarks (SimpleBenchmark, 20k warmup / 100k
measurement) on a chain-topology network showed Java 25 gave 8.7% gain on
full-reset baseline at n=128 vs Java 17. The arithmetic paths in
`PotentialMultiplier`, `BayesProjection`, `BayesAbsorption` are the obvious
targets for Vector API SIMD and Valhalla flat arrays. Epic tracking the
individual features: https://github.com/apache/incubator-kie-drools/issues/6668

**Promoted to:**
