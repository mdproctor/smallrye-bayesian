# smallrye-bayesian — Incremental Bayesian Update Design Spec

**Date:** 2026-04-19
**Author:** Mark Proctor
**Status:** Draft — approved for implementation planning

---

## 1. Problem Statement

`smallrye-bayesian` implements exact Bayesian inference over a junction tree. When any evidence
changes, `BayesInstance.globalUpdate()` performs a full reset + full collect + full distribute
pass over every clique and separator in the tree, regardless of how many variables actually
changed. For networks larger than ~15 variables this is unnecessarily expensive: a single
evidence change can affect only O(depth) messages out of O(N) total, yet all O(N) are
recomputed.

The goal of this design is to make Bayesian updating **incremental**: only the portions of the
junction tree affected by an evidence change are recomputed, while the rest reuse calibrated
values from the previous propagation pass.

**Constraints:**
- Maintain the existing highly-optimised flat `double[]` array data structures throughout.
- No structural changes to `CliqueState`, `SeparatorState`, `JunctionTreeClique`, or
  `JunctionTreeSeparator`.
- Retain a permanently preserved `globalUpdateForced()` baseline for benchmarking.
- All three phases must be independently measurable via JMH benchmarks.

---

## 2. Background: Current Evaluation Pipeline

The junction tree is built once at `KieBase` load time:
- Variables with CPTs → moralize → triangulate → cliques → junction tree

At runtime, `BayesInstance` holds:

| Field | Type | Purpose |
|---|---|---|
| `cliqueStates` | `CliqueState[]` | Runtime clique potentials (`double[]` per clique) |
| `separatorStates` | `SeparatorState[]` | Runtime separator potentials (`double[]` per separator) |
| `likelyhoods` | `BayesLikelyhood[]` | Evidence per variable, indexed by variable ID |
| `dirty` | `long` | Bitset: which variables have changed evidence since last update |

`JunctionTreeClique.resetState()` resets a clique to its CPT-initialized potentials (copies
from the clique's stored `double[] potentials`). `JunctionTreeSeparator.resetState()` fills
separator potentials with `1.0`. This is the invariant Phase 2 relies on.

**Current `globalUpdate()` sequence:**
1. If `dirty > 0`: reset all clique and separator potentials to initial values
2. Apply all evidence likelihoods into their host clique potentials
3. Full collect (bottom-up): every clique sends a message to its parent
4. Full distribute (top-down): every clique sends a message to each child
5. `dirty = 0`

---

## 3. Benchmarking (Cross-Cutting — Implemented First)

Benchmarks are first-class and are written before any incremental code so that a true baseline
is captured. They live in:

```
smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/
    BayesNetworkFixtures.java    # synthetic + BIF network generators
    BayesBenchmark.java          # JMH @Benchmark methods
    BenchmarkMain.java           # standalone main() entry point
```

### 3.1 `BayesNetworkFixtures`

Generates reproducible synthetic networks at four sizes:

| Label | Variables | Topology |
|---|---|---|
| S | 8 | chain + sprinkler/garden BIF |
| M | 20 | balanced binary tree |
| L | 50 | mixed DAG |
| XL | 100 | mixed DAG |

Also loads the standard Alarm network (37 variables) from BIF via the existing `XmlBifParser`.

`@Param("networkSize")` cycles through `{S, M, L, XL, ALARM}`.

### 3.2 Benchmark methods

| Method | `@Param` | What it measures |
|---|---|---|
| `baselineFull` | networkSize | `globalUpdateForced()` — always full reset+propagate, never removed |
| `noChangeUpdate` | networkSize | `globalUpdate()` twice, second call has no evidence change |
| `singleEvidenceChange` | networkSize | Change one variable's likelihood, then `globalUpdate()` |
| `batchEvidenceChange` | networkSize, dirtyCount∈{1,3,10}% | Change N variables, then `globalUpdate()` |
| `queryAll` | networkSize | `globalUpdate()` then marginalize all variables |
| `querySpecific` | networkSize | `globalUpdate()` then marginalize one named variable |

### 3.3 Baseline preservation

`BayesInstance` gets a `globalUpdateForced()` method that permanently bypasses all incremental
logic and always runs the full reset + applyEvidence + collect + distribute sequence. This method
is never modified as incremental phases are added. `baselineFull` benchmarks this method
exclusively, providing a stable reference across all phases.

---

## 4. Phase 1: Calibration Guard

### 4.1 Purpose

Skip `globalUpdate()` entirely when the network is already calibrated and no evidence has
changed. Currently the full propagation runs on every call even with `dirty == 0`.

### 4.2 Data structure

One new field on `BayesInstance`:

```java
private boolean calibrated;
```

### 4.3 Algorithm changes

**`globalUpdate()`** — add short-circuit at the top:

```java
if (calibrated && !isDirty()) return;
// ... existing body unchanged ...
dirty = 0;
calibrated = true;
```

**`setLikelyhood()`** and **`unsetLikelyhood()`** — add `calibrated = false` alongside the
existing `dirty` bit-set.

**Constructor** — `calibrated` initialises to `false`.

### 4.4 Correctness

`calibrated` is false on construction and after any evidence change. It becomes true only after
a successful full propagation. The guard fires only when both `calibrated == true` and
`dirty == 0`, meaning the posterior is already exact for the current evidence state.

### 4.5 Expected benchmark impact

- `noChangeUpdate`: second call drops from O(N·K) propagation to a single branch-not-taken.
- All other benchmarks: no change (evidence always changes before calling `globalUpdate()`).
- `baselineFull`: unchanged (calls `globalUpdateForced()`).

### 4.6 Code delta

~5 lines across `BayesInstance.java`. No new classes, no new arrays.

---

## 5. Phase 2: Dirty Subtree Tracking + Incremental Collect

### 5.1 Purpose

When evidence changes on a subset of variables, only the collect-phase messages on the path
from those variables' cliques to the root are stale. All other separator values from the
previous calibration are correct. Phase 2 snapshots those separator values before the reset and
reinjects them for clean subtrees, avoiding recomputation.

### 5.2 Data structures

Two new pre-allocated fields on `BayesInstance`, both allocated at construction:

```java
private boolean[] subtreeHasDirty;   // one slot per clique
private double[][] sepPotSnapshots;  // one double[] per separator, same length as sep potentials
```

`subtreeHasDirty[cliqueId]` is true if any clique in the downward subtree of that clique
(away from root) contains a variable with dirty evidence. Cleared to false after each update.

`sepPotSnapshots[sepId]` is a pre-allocated array of the same length as the corresponding
separator's potentials. It holds a snapshot of the calibrated separator taken before the reset.

**Memory cost:** equal to the combined size of all separator potential arrays. For a 50-variable
network this is typically a few kilobytes.

### 5.3 Algorithm

`globalUpdate()` takes the incremental path when `calibrated && isDirty()`:

```
if (calibrated && !isDirty()) return;                   // Phase 1 guard

if (calibrated) {                                       // Phase 2 setup
    snapshotSeparators();
    computeDirtySubtrees();
}

if (isDirty()) reset();                                 // unchanged
applyEvidence();                                        // unchanged
collectIncremental(root);                               // modified
distributeEvidence(root);                               // unchanged — full pass
dirty = 0;
calibrated = true;
Arrays.fill(subtreeHasDirty, false);
```

**`snapshotSeparators()`:** for each separator, `System.arraycopy` its current potentials into
`sepPotSnapshots[sep.getId()]`. Called before `reset()` so the calibrated values are preserved.

**`computeDirtySubtrees()`:** for each set bit in `dirty`, look up `var.getFamily()` to find
the host clique ID, then walk upward via `clique.getParentSeparator().getParent()` to the root,
setting `subtreeHasDirty[cliqueId] = true` at each step. Early-exit if a clique is already
marked (prevents redundant traversal on shared ancestors for batch changes). O(dirtyCount ×
depth).

**`collectIncremental()`:** before calling `passMessage(source, sep, target)`, check
`subtreeHasDirty[source.getId()]`:

- **True (dirty subtree):** call `passMessage` normally — project source into sep, absorb into
  target.
- **False (clean subtree):** instead of recomputing:
  1. Copy `sepPotSnapshots[sep.getId()]` into `separatorStates[sep.getId()].getPotentials()`.
  2. Call `absorb()` directly with `oldSepPots` = the post-reset separator state (all 1.0, per
     `JunctionTreeSeparator.resetState()`).
  
  The absorb formula becomes: `target[i] = targetInit[i] × snapshot[j] / 1.0`, which correctly
  injects the previously calibrated message without any project computation.

**Why this is correct:** after `reset()`, clique potentials are CPT-initialized and separator
potentials are all 1.0. Absorbing a clean subtree's calibrated separator into its adjacent
clique is mathematically identical to re-running that message — because the CPTs in that subtree
are unchanged and so is its evidence. Dirty subtrees always go through normal `passMessage`, so
no stale values reach the root.

**Distribute phase:** unchanged — full pass. For the all-variables-queried case, all distribute
messages are required. Partial distribute pruning (query-specific) is deferred until benchmarks
quantify whether it is worth the complexity.

### 5.4 Expected benchmark impact

For a single variable change in a balanced 50-clique tree:
- Collect messages: drops from ~49 to ~log₂(50) ≈ 6 (only the dirty path).
- Distribute messages: unchanged (~49).
- Net for all-variables query: ~40–50% reduction in total message work.
- Net for specific-variable query: ~70–80% (distribute pruning to follow in a later pass).

Batch changes touching 10% of cliques: proportionally fewer clean subtrees but still
significant savings for large networks.

### 5.5 Code delta

- `BayesInstance`: 2 new fields, `snapshotSeparators()`, `computeDirtySubtrees()`,
  `injectAndAbsorb()`, modified `globalUpdate()`, modified collect traversal.
- No changes to `JunctionTreeClique`, `JunctionTreeSeparator`, `CliqueState`, `SeparatorState`.

---

## 6. Phase 3: Fast Evidence Retraction

### 6.1 Purpose

Phases 1 and 2 still call `reset()` on every dirty update, reinitialising all clique potentials
from CPTs and re-applying all evidence from scratch. Phase 3 eliminates this: it divides out
the old evidence factor directly from the calibrated clique potentials and multiplies in the
new factor, then propagates only outward from the changed clique. No reset is needed.

### 6.2 Mathematical basis

After full calibration, clique C_x's potentials encode: CPT contribution × all absorbed child
messages × applied evidence likelihood × normalisation. The evidence on variable X entered as a
likelihood multiplication at C_x. Retracting that evidence is exact division; inserting new
evidence is multiplication. The remainder of C_x's state (CPT + non-dirty child messages) is
unchanged and does not need to be recomputed.

### 6.3 Data structures

Two new pre-allocated fields on `BayesInstance`, both allocated at construction:

```java
private double[][] calibratedCliquePots;  // one double[] per clique, same length as clique potentials
private double[][] calibratedSepPots;     // one double[] per separator, same length as sep potentials
```

These are written after every successful calibration (both full and retraction updates), forming
a "last known good" snapshot. They are distinct from Phase 2's `sepPotSnapshots`, which are
taken immediately before a reset.

**Memory cost:** doubles the storage for all clique and separator potentials. For networks with
wide cliques (treewidth ≥ 6, potential arrays of 64+ elements) this is non-trivial. Phase 3
is therefore opt-in via a constructor flag `BayesInstance(JunctionTree, boolean enableFastRetract)`;
it defaults to disabled and must be benchmarked before enabling by default.

### 6.4 Retraction update path

Phase 3 activates only when all hold:
- `calibrated == true`
- Every dirty variable's previous likelihood is non-null (we are changing, not first-inserting)
- No entry in any affected clique potential is zero (division by zero guard — see §6.5)

```
for each dirty variable X with old likelihood L_old and new likelihood L_new:
    load calibratedCliquePots[X.getFamily()] into working clique state
    divide out L_old contribution element-wise
    multiply in L_new contribution element-wise
    normalise

propagate outward from each dirty clique:
    collect upward to root (dirty path only — same as Phase 2 subtree logic)
    distribute downward from root (full pass)

save new calibrated clique + separator potentials
```

The collect pass in Phase 3 uses the same dirty-subtree tracking as Phase 2 — both are active
simultaneously. Phase 3 replaces the reset+reapply cycle; Phase 2 handles which collect
messages to skip.

### 6.5 Fallback discipline

If any clique potential entry is zero (which occurs when hard evidence sets a state's likelihood
to 0), exact division is undefined. In this case the retraction path silently falls back to the
Phase 2 path (reset + snapshot-injected collect). The fallback is transparent to the caller.
The benchmark will measure fallback frequency under hard-evidence workloads.

### 6.6 Expected additional gain over Phase 2

Eliminates the O(N·K) reset loop where N is the number of cliques and K is the average clique
potential array length. For treewidth 4 (16-element arrays) and 50 cliques: ~800 array writes
saved per update. For treewidth 6 (64-element arrays) the saving is ~3200 writes. The benchmark
will quantify this against the Phase 2 baseline.

### 6.7 Code delta

- `BayesInstance`: 2 new fields (conditional on flag), `saveCalibrated()`,
  `retractionUpdate()`, `divideOutLikelihood()`, modified `globalUpdate()`.
- No changes to `JunctionTreeClique`, `JunctionTreeSeparator`, `CliqueState`, `SeparatorState`.

---

## 7. Interaction Between Phases

The phases compose cleanly and activate in order:

```
globalUpdate() {
    if (calibrated && !isDirty())          → Phase 1: return immediately
    if (calibrated && enableFastRetract && retractable())
                                           → Phase 3: divide/multiply + dirty collect
    else if (calibrated)                   → Phase 2: snapshot + reset + dirty collect
    else                                   → Full path (first call)
}
```

`globalUpdateForced()` bypasses all of this unconditionally and is used only by benchmarks.

---

## 8. File Structure

```
smallrye-bayesian/
  src/main/java/org/drools/beliefs/bayes/
    BayesInstance.java              [primary changes across all phases]
  src/test/java/org/drools/beliefs/bayes/
    benchmark/
      BayesNetworkFixtures.java     [synthetic + BIF network generators]
      BayesBenchmark.java           [JMH benchmark class]
      BenchmarkMain.java            [standalone entry point]
```

No new production classes. No changes to any class other than `BayesInstance`.

---

## 9. Implementation Order

1. **Benchmarks first** — `BayesNetworkFixtures` + `BayesBenchmark` with `baselineFull` only.
   Capture baseline numbers across all network sizes.
2. **Phase 1** — add `calibrated` flag. Re-run benchmarks, confirm `noChangeUpdate` win.
3. **Phase 2** — add snapshot arrays + dirty subtree tracking + incremental collect.
   Re-run benchmarks, confirm `singleEvidenceChange` and `batchEvidenceChange` improvements.
4. **Phase 3** — add calibrated snapshots + retraction path behind opt-in flag.
   Re-run benchmarks, quantify additional gain vs Phase 2. Decide whether to enable by default
   based on memory/throughput trade-off the numbers reveal.

---

## 10. Decision Log

### 10.1 Why IJTI-style dirty subtree tracking over Lazy Propagation

Lazy Propagation (Madsen & Jensen 1999) requires each clique to store a list of unevaluated
factor references rather than a single combined `double[]`. This is incompatible with the
existing architecture's single-array-per-clique design without a major restructure.

IJTI (Agli et al. 2016) achieves equivalent collect-side savings by adding only a
`boolean[]` per clique (subtree dirty flags) and a `double[][]` snapshot for separators.
The existing `CliqueState` and `SeparatorState` arrays are untouched.

### 10.2 Why snapshot-inject rather than "skip and divide"

An alternative to snapshotting is to divide out old messages from adjacent cliques
(Cowell & Dawid 1992 fast retraction applied at the message level). This avoids the snapshot
array but requires division safety checks on every non-dirty separator, and the division must
happen before reset — adding ordering complexity. Snapshotting before reset is simpler, always
safe, and has negligible cost (one `System.arraycopy` per separator).

Phase 3 uses division at the clique level (for evidence likelihood only), where the structure
of the likelihood factor makes the zero-check straightforward.

### 10.3 Why distribute is left as a full pass in Phase 2

For the all-variables-queried case (which the user identified as common), every clique is a
query target, so every distribute message is required. Partial distribute pruning only helps
for specific-variable queries. Benchmarks will measure the specific-variable case after Phase 2
is complete and guide whether a Phase 4 distribute optimisation is warranted.

### 10.4 Why Phase 3 is opt-in

Phase 3 doubles the memory footprint for clique and separator potential storage. For wide
treewidths this can be significant. Defaulting to opt-out ensures no regression for existing
users until benchmarks confirm the throughput gain justifies the memory cost for their network
sizes.
