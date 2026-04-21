# smallrye-bayesian — Claude Code Project Guide

## What this is

Standalone exact Bayesian inference library extracted from Apache Drools drools-beliefs.
Zero Drools/KieApi dependencies. Intended for submission to the SmallRye project.
Algorithm: junction tree (Lauritzen-Spiegelhalter) with three-phase incremental inference.

## Build & test

```bash
mvn test                          # run full test suite (98 tests)
mvn test-compile                  # compile only
mvn test -Dtest=IncrementalUpdateTest    # run a specific test class
```

Run benchmarks (no JMH fat-jar setup yet — use SimpleBenchmark):
```bash
mvn test-compile -q
java -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -q -DforceStdout)" \
     io.smallrye.bayesian.benchmark.SimpleBenchmark
```

## Java version

- **Minimum**: Java 21 (LTS)
- **Current dev target**: Java 25 (LTS, installed at `/Users/mdproctor/Library/Java/JavaVirtualMachines/openjdk-25.0.1`)
- Java 26 also installed (`/Library/Java/JavaVirtualMachines/jdk-26.jdk`)

## Package structure

```
io.smallrye.bayesian
├── graph/                   # Generic directed graph — Graph<T>, GraphNode<T>, Edge
│   └── impl/                # EdgeImpl, GraphImpl, GraphNodeImpl, list/map stores
├── model/                   # BIF/XMLBIF file format — XmlBifParser, Bif, Network, Variable
├── util/bitmask/            # Copied from drools-util — BitMask, LongBitMask, OpenBitSet etc.
├── BayesInstance            # *** PRIMARY CLASS — runtime inference state, all incremental logic
├── JunctionTree             # Compiled network (cliques + separators + root)
├── JunctionTreeBuilder      # Builds junction tree from a BayesNetwork (moralize→triangulate→build)
├── JunctionTreeClique       # Node in the junction tree, holds double[] potential array
├── JunctionTreeSeparator    # Edge in the junction tree, holds double[] potential array
├── BayesVariable            # Variable definition with CPT (double[][])
├── BayesVariableState       # Runtime marginal distribution (double[])
├── BayesLikelyhood          # Evidence likelihood for one variable
├── BayesInstance            # Runtime inference state
├── BayesNetwork             # Extends GraphImpl<BayesVariable>
├── CliqueState              # Runtime clique potentials (double[])
├── SeparatorState           # Runtime separator potentials (double[])
├── BayesProjection          # Marginalization (project clique → separator)
├── BayesAbsorption          # Message absorption (separator → clique)
├── PotentialMultiplier      # CPT multiplication into clique potentials (also has divide())
├── Marginalizer             # Extract single-variable marginal from clique potential
├── EliminationCandidate     # Used during triangulation (variable elimination)
├── SeparatorSet             # Pair of cliques sharing a separator (used in Prim's-like tree build)
├── GlobalUpdateListener     # Hook: called before/after each full propagation
└── PassMessageListener      # Hook: called on each project+absorb message pass
```

## Key algorithm — three-phase incremental inference in BayesInstance

All three phases are implemented and active. They compose in `globalUpdate()`:

**Phase 1 — Calibration guard** (`calibrated` flag)
- If `calibrated=true` and `dirty.isEmpty()`: return immediately. Free.
- Cost: one branch. Eliminates redundant calls in hot loops.

**Phase 2 — Dirty subtree incremental collect** (`subtreeHasDirty[]`, `sepPotSnapshots[][]`)
- Snapshot separator potentials before reset.
- Mark which cliques have dirty evidence (walk upward from dirty vars).
- During collect (bottom-up): skip `passMessage` for clean subtrees, inject snapshot instead.
- Distribute is always a full pass (future optimisation: query-specific pruning).
- `injectSnapshotAndAbsorb()`: uses post-collect separator snapshots (NOT post-distribute).
- `calibratedSepPots[][]` (Phase 3 only) holds post-distribute values — different invariant.

**Phase 3 — Fast evidence retraction** (`enableFastRetract` flag, opt-in constructor)
- Avoids full reset: divides out old evidence likelihood, multiplies in new.
- Uses `PotentialMultiplier.divide()` — exact structural mirror of `multiple()`.
- Falls back to Phase 2 when: first insertion (null previous), or any clique potential is 0 (hard evidence → division undefined).
- Separate collect path: `collectEvidenceRetract()` uses `calibratedSepPots` (post-distribute) not `sepPotSnapshots` (post-collect) — these serve different invariants.

## Critical implementation notes

**Two snapshot arrays with different semantics:**
- `sepPotSnapshots[sep]` — post-COLLECT separator values (upward messages only). Used in Phase 2 `injectSnapshotAndAbsorb()`.
- `calibratedSepPots[sep]` — post-DISTRIBUTE separator values (fully calibrated). Used in Phase 3 `collectEvidenceRetract()` as the neutral ratio reference.
Mixing these silently corrupts marginals. This was the hardest bug to find during implementation.

**Snapshot timing in `globalUpdate()`:**
`snapshotSeparators()` is called AFTER the collect phase, not before reset. It captures the upward messages, which is what Phase 2's inject step needs.

**BitMask for dirty/decided:**
`BayesInstance.dirty` and `decided` are `BitMask` (from `util/bitmask/`). `BitMask.getEmpty(n)` returns `LongBitMask` (raw long) for n ≤ 64 and `OpenBitSet` (long[]) for n > 64.  `LongBitMask.set(id)` auto-upgrades to `OpenBitSet` if id ≥ 64.

**OpenBitSet in JunctionTreeBuilder:**
The clique/separator bitsets inside `JunctionTreeBuilder` use `OpenBitSet` with explicit sizing: `new OpenBitSet(numCliques)` and `new OpenBitSet(graph.size())`. The default no-arg constructor creates a 64-bit set and will fail for networks > 64 variables — this was a bug we fixed.

**`globalUpdateForced()`:**
Always runs full reset+applyEvidence+collect+distribute regardless of incremental state. Benchmarks use this as the permanent baseline. After a forced update, `snapshotSeparators()` and `saveCalibrated()` are also called so subsequent `globalUpdate()` calls can take the incremental path.

## Test files

| File | What it tests |
|---|---|
| `IncrementalUpdateTest` | All three phases: correctness (marginals vs forced baseline), message count, Phase 1 listener counts |
| `IncrementalUpdateEdgeCaseTest` | n=2 minimum, all-dirty, n=100, first call, hard→soft transitions |
| `GlobalUpdateTest` | Pre-existing full propagation tests |
| `JunctionTreeBuilderTest` | Triangulation, moralization, clique/separator construction |
| `JunctionTreeTest` | Clique potential initialization from CPTs |
| `example/SprinkerTest` | Classic sprinkler Bayesian network |
| `example/EarthQuakeTest` | Earthquake Bayesian network |
| `benchmark/SimpleBenchmark` | Hand-rolled benchmark (JMH 1.21/Java 17 annotation processor incompatibility) |

## Benchmark results summary

See `BENCHMARKS.md` for full tables. Key numbers at n=128:
- Java 17 baselineFull: 74.69 µs → Java 25: 68.21 µs (**-8.7% free from JVM upgrade**)
- Phase 2 singleEvidenceChange vs baselineFull: **-22% at n=128** (incremental collect savings)
- Phase 1 noChangeUpdate: **~0.11 µs** at all sizes — effectively free

## What's NOT in this project (stripped during extraction from drools-beliefs)

- `BayesBeliefSystem` — Drools TMS integration (BeliefSet, LogicalDependency etc.)
- `BayesHardEvidence`, `NonConflictingModeSet` — Drools TMS types
- `BayesModeFactory`, `BayesModeFactoryImpl` — Drools Mode integration
- `runtime/BayesRuntimeImpl`, `BayesRuntimeService` — KieSession integration
- `assembler/` — drools-compiler integration
- `weaver/` — KieWeaverService integration
- Integration tests using KieSession (`BayesBeliefSystemTest` etc.)

## Roadmap

See `docs/incremental-bayes-design.md` for the full design.
See `IDEAS.md` for the two-track Java 21/Java 27 Valhalla idea.
See GitHub issue #1 for the Java platform features epic.

**Next algorithm work:**
1. Distribute phase pruning for specific-variable queries (skip subtrees with no query target)
2. Vector API SIMD on `PotentialMultiplier` / `BayesProjection` / `BayesAbsorption`
3. Parallel distribute using Virtual Threads (Java 21)
4. Valhalla value classes on `CliqueState`, `SeparatorState` (Java 27 preview, est. Sept 2026)

## GitHub

- Repo: https://github.com/mdproctor/smallrye-bayesian (transfer to `smallrye/` when ready)
- Java platform features epic: https://github.com/mdproctor/smallrye-bayesian/issues/1
- Original drools work: https://github.com/apache/incubator-kie-drools (drools-beliefs module)
