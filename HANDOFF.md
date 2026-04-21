# Session Handoff

**Project:** smallrye-bayesian  
**Extracted from:** Apache Drools `drools-beliefs` module  
**Session date:** 2026-04-19 / 2026-04-20 / 2026-04-21  
**GitHub:** https://github.com/mdproctor/smallrye-bayesian

---

## What was built this session

### Core deliverable
A three-phase incremental Bayesian inference engine, implemented in `BayesInstance.java`:

- **Phase 1** — Calibration guard: `if (calibrated && dirty.isEmpty()) return;` — skip entirely when nothing changed. ~0.11 µs overhead at all network sizes.
- **Phase 2** — Dirty-subtree incremental collect: snapshot separator potentials before reset; skip `passMessage` for clean subtrees and inject the snapshot instead. ~22% improvement over full baseline at n=128.
- **Phase 3** — Fast evidence retraction: divide out old likelihood, multiply in new, skip reset entirely. Opt-in via `new BayesInstance(jt, true)`. Falls back to Phase 2 on hard evidence (zero in potential) or first insertion.

### Supporting work
- `BitMask` migration: `long dirty/decided` → `BitMask` using `LongBitMask` (≤64 vars) / `OpenBitSet` (>64 vars)
- `JunctionTreeBuilder` fix: `new OpenBitSet()` → `new OpenBitSet(numCliques)` / `new OpenBitSet(graph.size())` — was failing for networks > 64 variables
- Full test suite: 98 tests covering unit, correctness (vs `globalUpdateForced()` oracle), message-count, edge cases, and 3 end-to-end KieSession tests (those require `-am` in drools; equivalent tests pending in this project)
- JMH benchmark infrastructure + `SimpleBenchmark` (hand-rolled, works on Java 17–26)
- Maven upgraded to Java 25 target; JMH upgraded from 1.21 → 1.37

---

## Current state of this repo

```
98 tests passing, 0 failures
Maven: Java 21 minimum, Java 25 current dev target
JMH: 1.37 (annotation processor generates BenchmarkList correctly)
```

The project was just extracted from drools-beliefs. It compiles and tests cleanly. **No new features have been added yet post-extraction** — the code is exactly what was in drools-beliefs, repackaged to `io.smallrye.bayesian`.

---

## What was NOT done / open work

### Immediate next steps (in priority order)

1. **Add a proper README.md** — the repo has no README. Needs: what it is, how to use it, Maven coordinates, basic API example (build a BayesNetwork, create a JunctionTree, create a BayesInstance, set evidence, globalUpdate, marginalize).

2. **Fix XStream dependency** — `XmlBifParser` uses XStream for BIF file parsing. XStream is a transitive pull-in with a security history. Either: replace with a standard JAXB/StAX parser (better), or add XStream explicitly to pom.xml and document it.

3. **Distribute phase pruning** — Phase 2 only optimises the collect phase. Distribute is still a full O(N) pass. For specific-variable queries (not querying all variables), you can skip distributing to subtrees that don't contain the query target. This is the largest remaining algorithmic gain.

4. **End-to-end tests** — The `IncrementalBayesIntegrationTest` from drools used KieSession. This project needs standalone integration tests that build a full network from a BIF file, set evidence via rules or direct API, and verify posteriors.

5. **Vector API on PotentialMultiplier** — `PotentialMultiplier.multiple()` and `divide()` are inner loops over `double[]`. Vectorising with `DoubleVector` (JEP 529, incubating Java 26) is the single biggest remaining performance opportunity. Requires `--add-modules jdk.incubator.vector`.

6. **Parallel distribute with Virtual Threads** — Independent subtrees in the distribute phase have no data dependencies. Fan out to virtual threads (Java 21, finalized). Only meaningful at n > ~50.

### Medium-term

7. **Maven coordinates / publishing** — GroupId `io.smallrye` requires SmallRye project membership. For now publishes nothing. Consider `io.github.mdproctor` for interim snapshots.

8. **Alarm / Hepar2 BIF networks** — All current benchmarks use synthetic chain topology. Dense-DAG real-world networks (Alarm 37 vars, Hepar2 70 vars) will show different incremental savings characteristics.

9. **Transfer to `smallrye/smallrye-bayesian`** — When ready, transfer via GitHub repo settings. Will need: CONTRIBUTING.md, proper license headers on all files, SmallRye CI pipeline.

### Long-term (parked)

10. **Two-track strategy** (see `IDEAS.md`) — Java 21 LTS stable track + Java 27 Valhalla experimental track. `CliqueState`/`SeparatorState` as value classes for flat heap layout. Earliest: Java 27 preview (est. Sept 2026).

---

## Key design decisions (don't re-litigate without reading context)

**Why two snapshot arrays?**
`sepPotSnapshots` = post-COLLECT (upward messages only). `calibratedSepPots` = post-DISTRIBUTE (fully calibrated). These serve different purposes in Phase 2 and Phase 3 respectively. Using either for the wrong purpose silently corrupts marginals. Read `docs/incremental-bayes-design.md` §5.3 before touching this.

**Why snapshot AFTER collect, not before reset?**
Phase 2 needs the upward messages (post-collect), not the fully calibrated separators (post-distribute). This was non-obvious and required debugging to discover. The plan said "before reset" — the implementation correctly moved it.

**Why BitMask instead of long?**
The original `long dirty` was limited to 64 variables. `BitMask.getEmpty(n)` returns `LongBitMask` for ≤64 (same performance, raw long internally) and `OpenBitSet` for >64. `LongBitMask.set(id)` auto-upgrades to `OpenBitSet` if id≥64. The JunctionTreeBuilder's OpenBitSet sizing was a separate bug (no-arg constructor = 64 bits).

**Why Phase 3 is opt-in?**
Doubles memory for clique and separator potentials (`calibratedCliquePots`, `calibratedSepPots`). Default off. Enable: `new BayesInstance(junctionTree, true)`.

**Why distribute is still full?**
For the all-variables-queried case (the common case), every distribute message is needed. Partial distribute pruning only helps for specific-variable queries. Deferred pending benchmark evidence it's worth the complexity.

**Why not Lazy Propagation (Madsen & Jensen 1999)?**
Requires replacing each clique's single `double[]` with a list of unevaluated factor references. Incompatible with the flat-array architecture. See `LITERATURE.md` for full analysis.

---

## Files to know

| File | Why it matters |
|---|---|
| `src/main/java/io/smallrye/bayesian/BayesInstance.java` | Everything — all three phases, ~600 lines |
| `src/test/java/io/smallrye/bayesian/IncrementalUpdateTest.java` | Primary correctness + Phase 1/2/3 tests |
| `src/test/java/io/smallrye/bayesian/IncrementalUpdateEdgeCaseTest.java` | Edge cases incl. n=100 (post BitMask fix) |
| `src/test/java/io/smallrye/bayesian/benchmark/SimpleBenchmark.java` | Benchmark runner (use this, not BenchmarkMain) |
| `BENCHMARKS.md` | All results: Java 17, 22, 25 comparison |
| `docs/incremental-bayes-design.md` | Full design spec with decision log |
| `IDEAS.md` | Two-track Java 21/27 idea |

---

## Drools relationship

The `drools-beliefs` module in Apache Drools (`~/projects/drools`) still has all this code. It has NOT been deleted — deletion was deliberately deferred. The two codebases will diverge. When this project is ready for SmallRye submission, the drools-beliefs copy can be deleted and replaced with a dependency on this library.

The journal module design (`drools-journal`) is an independent initiative for durable Drools sessions (append-only log). It's in `~/projects/drools/docs/superpowers/specs/2026-04-09-drools-journal-design.md` — someone else is implementing it. Not related to this project.
