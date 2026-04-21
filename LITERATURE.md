# Literature Review

Academic papers and algorithms that informed the design of `smallrye-bayesian`.
For each, the implementation status in this codebase is noted.

---

## Foundational: Junction Tree Algorithm

### Lauritzen & Spiegelhalter (1988)
**"Local computations with probabilities on graphical structures and their application to expert systems"**  
*Journal of the Royal Statistical Society B, 50(2):157–224*

The foundational paper for exact Bayesian inference on general graphs. Introduces the junction tree (join tree) algorithm: moralize the DAG, triangulate, build a clique tree, run collect+distribute message passing to calibrate all clique potentials. Every clique potential is a `double[]` array indexed by the Cartesian product of the clique's variable states.

**Status: Fully implemented.**  
`JunctionTreeBuilder` (moralize → triangulate → Prim-like spanning tree → calibrate).
`JunctionTreeClique`, `JunctionTreeSeparator` store the `double[]` potentials.
`BayesInstance.collectChildEvidence()` and `distributeEvidence()` implement the message passing.

---

### Pearl (1988)
**"Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference"**  
*Morgan Kaufmann*

Introduces belief propagation for polytrees and the foundation of message passing in probabilistic graphical models. The HUGIN algorithm (collect+distribute on junction trees) generalises Pearl's polytree propagation to arbitrary graphs.

**Status: Indirectly implemented** via the Lauritzen-Spiegelhalter junction tree approach. No direct polytree optimisation (not needed for general graphs).

---

## Incremental Inference

### Agli, Bonnard, Gonzales & Wuillemin (2016)
**"Incremental Junction Tree Inference"**  
*IPMU 2016, Communications in Computer and Information Science vol. 610, Springer*  
[Open-access PDF — HAL](https://hal.sorbonne-universite.fr/hal-01345418v1/document)

**The primary paper for Phase 2.** Introduces a binary invalidation indicator δ_{i→j} per directed edge of the junction tree. After evidence changes, only messages on the path from dirty cliques to query targets are recomputed. Messages on clean subtrees are reused from the previous calibration.

The key correctness theorem: a message ψ_{i→j} can be reused if and only if the subtree rooted at i (directed away from j) contains no variable with changed evidence.

**Status: Partially implemented, with adaptation.**  
Phase 2 (`subtreeHasDirty[]`, `injectSnapshotAndAbsorb()`) is inspired by IJTI but differs in a key way: IJTI reuses the old separator value directly; this implementation snapshots separator potentials before reset and reinjects them. This was necessary because `JunctionTreeSeparator.resetState()` fills with 1.0 — we can't recover the old calibrated value without snapshotting.

The IJTI paper also optimises the distribute phase for specific-variable queries (skip subtrees with no query target). **This part is NOT yet implemented** — distribute is currently always a full pass.

---

### Madsen & Jensen (1999)
**"Lazy propagation: A junction tree inference algorithm based on lazy evaluation"**  
*Artificial Intelligence, 113(1–2):203–245*  
[arXiv:1301.7398](https://arxiv.org/abs/1301.7398)

Lazy Propagation (LP) keeps each clique's potential as a set of unevaluated factor references rather than a combined `double[]`. Messages are assembled just-in-time with aggressive pruning (barren variable elimination, d-separation tests). Most efficient when evidence is dense (many instantiated variables).

**Status: Considered and rejected.**  
LP requires replacing `double[]` per clique with a list of factor references. This is fundamentally incompatible with the flat-array architecture (and with the Phase 2 snapshot approach). Rejected during design. See `docs/incremental-bayes-design.md` §10.1 for full rationale.

---

### D'Ambrosio (1993)
**"Incremental Probabilistic Inference"**  
*UAI 1993, pp. 301–308*  
[arXiv:1303.1490](https://arxiv.org/abs/1303.1490)

The earliest systematic treatment of incremental inference. Argued that standard inference operates at too coarse a grain and proposed interleaving probability evaluation with model construction. Foundational/conceptual — established the justification for all later incremental approaches.

**Status: Foundational only. Not directly implemented.**  
Motivated the design philosophy but no specific algorithm was adopted from this paper.

---

## Fast Evidence Retraction

### Cowell & Dawid (1992)
**"Fast retraction of evidence in a probabilistic expert system"**  
*Statistics and Computing, 2:37–40*  
[Springer](https://link.springer.com/article/10.1007/BF01890547)

### Dawid (1992)
**"Applications of a general propagation algorithm for probabilistic expert systems"**  
*Statistics and Computing, 2:25–36*  
[Springer](https://link.springer.com/article/10.1007/BF01890546)

These two papers establish fast evidence retraction: after a full calibration, retracting evidence on variable X is equivalent to dividing out X's likelihood from X's host clique potential, then re-propagating outward from that clique — no full reset needed. HUGIN implements this as "PropSumFastRetract".

**Status: Phase 3 implements the core idea.**  
`BayesInstance.retractionUpdate()` restores calibrated clique potentials, divides out old likelihood (`BayesLikelyhood.divideFrom()`), multiplies in new likelihood (`multiplyInto()`), normalises, then runs an incremental collect+distribute. Division is implemented by `PotentialMultiplier.divide()` — exact structural mirror of `multiple()` with `/=` instead of `*=`.

**Notable deviation from paper:** The Cowell/Dawid approach retracted on the HUGIN separator tables. This implementation retracks on the clique potentials directly and then uses the Phase 2 incremental collect to propagate. This was necessary because the separator values serve different roles in collect vs distribute.

---

## Differential / Arithmetic Circuit Approach

### Darwiche (2000)
**"A Differential Approach to Inference in Bayesian Networks"**  
*UAI 2000, pp. 123–132*  
[arXiv:1301.3847](https://arxiv.org/abs/1301.3847)

### Darwiche (2003)
**"A differential approach to inference in Bayesian networks"**  
*Journal of the ACM, 50(3):280–305*  
[ACM DL](https://dl.acm.org/doi/10.1145/765568.765570)

Compiles the Bayesian network into a multivariate polynomial f(λ, θ) and then into an arithmetic circuit (AC) — a DAG of add/multiply operations. Once compiled, a single forward+backward pass through the AC gives all posterior marginals simultaneously. Incremental updates recompute only the affected sub-circuit.

**Status: Considered, not implemented.**  
The AC approach requires an offline compilation step (building the AC from the junction tree) that restructures the data model entirely. The existing `double[]` clique/separator arrays don't map onto an AC without a new data structure. Deferred as a potential future Track 2 optimisation. If implemented, it would be the fastest option for repeated queries on the same evidence pattern.

---

## HUGIN Algorithm (Reference Implementation Context)

### Jensen, Lauritzen & Olesen (1990)
**"Bayesian updating in causal probabilistic networks by local computations"**  
*Computational Statistics Quarterly, 4:269–282*

The HUGIN algorithm paper — direct inspiration for the collect+distribute implementation, the separator table design, and the normalisation step in `BayesAbsorption`. HUGIN's data structures map directly onto this codebase: clique tables = `CliqueState.potentials`, separator tables = `SeparatorState.potentials`.

**Status: Core algorithm fully implemented.**  
`BayesInstance.globalUpdateForced()` is a faithful HUGIN implementation. Phases 1–3 add incremental behaviour on top.

---

## Implementation Status Summary

| Paper / Algorithm | Phase | Status |
|---|---|---|
| Lauritzen & Spiegelhalter (1988) — Junction tree | Core | **Fully implemented** |
| Pearl (1988) — Belief propagation | Core | Indirectly (via L&S) |
| Jensen et al. (1990) — HUGIN | Core | **Fully implemented** (`globalUpdateForced`) |
| D'Ambrosio (1993) — Incremental inference (framework) | — | Foundational only |
| Cowell & Dawid (1992) — Fast retraction | Phase 3 | **Implemented** (adapted) |
| Agli et al. (2016) — IJTI collect | Phase 2 | **Implemented** (adapted: snapshot-inject) |
| Agli et al. (2016) — IJTI distribute pruning | — | **NOT YET IMPLEMENTED** |
| Madsen & Jensen (1999) — Lazy Propagation | — | Rejected (incompatible architecture) |
| Darwiche (2000/2003) — Arithmetic Circuit | — | Not implemented (requires new data model) |

---

## Papers to Read Next

For **distribute phase pruning** (next algorithmic priority):
- Re-read Agli et al. §4.2 on the distribute-side δ indicator — the same dirty-path logic applies in reverse for query-variable pruning.

For **Vector API / SIMD**:
- Intel Intrinsics Guide for AVX2 double-precision FMA operations
- JEP 529 (Java 26 Vector API incubator) — API reference for `DoubleVector`

For **Valhalla value classes**:
- JEP 401 (Value Classes and Objects) — current EA spec
- [Inside.java — Try JEP 401](https://inside.java/2025/10/27/try-jep-401-value-classes/)
- Consider: can `CliqueState` become a value class if it holds a `double[]` (a mutable array)? Probably not directly — value classes with mutable fields are restricted. May need `CliqueState` to become a view into a flat shared array.

For **parallel distribute**:
- JEP 444 (Virtual Threads, Java 21 final) — check overhead for short-lived tasks vs platform threads
- Consider: is the junction tree separator access truly race-free per subtree? Need to verify no clique is shared between two independent subtrees during distribute.
