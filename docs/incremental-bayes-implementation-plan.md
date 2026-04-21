# smallrye-bayesian Incremental Bayesian Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BayesInstance.globalUpdate()` incremental — only recomputing the affected portion of the junction tree when evidence changes — while preserving the existing flat `double[]` array data structures.

**Architecture:** Three additive phases in `BayesInstance.java`: a calibration guard (skip entirely when nothing changed), dirty-subtree collect optimisation (snapshot clean separator values and skip recomputing their messages), and fast evidence retraction (divide out old evidence in-place, eliminating the full reset cycle). A JMH benchmark suite is written first to capture the baseline and measure each phase's gain.

**Tech Stack:** Java 17, JUnit Jupiter 5, AssertJ, JMH 1.21 (already in build-parent), Maven multi-module build.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `smallrye-bayesian/pom.xml` | Modify | Add JMH test dependencies |
| `smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java` | Modify | All incremental logic + `globalUpdateForced()` |
| `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesNetworkFixtures.java` | Create | Synthetic chain network builder at configurable sizes |
| `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java` | Create | JMH benchmark class — all `@Benchmark` methods |
| `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java` | Create | Standalone `main()` entry point for running outside Maven |
| `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java` | Create | Unit, correctness (marginal equality vs forced baseline), message-count, happy-path, and phase-interaction tests |
| `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateEdgeCaseTest.java` | Create | Edge cases, sequential changes, unset evidence |
| `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/integration/IncrementalBayesIntegrationTest.java` | Create | End-to-end via KieSession + BayesRuntime + DRL rules |

---

## Test Strategy

Four layers of tests, each in a distinct file:

### Layer 1 — Unit tests (`IncrementalUpdateTest.java`)
Test each new method in isolation: `computeDirtySubtrees`, `snapshotSeparators`, `injectSnapshotAndAbsorb`, `saveCalibrated`, `isRetractable`. Verify internal state via the public API (`marginalize`, `globalUpdateForced` as oracle) and the `PassMessageListener`/`GlobalUpdateListener` hooks already on `BayesInstance`.

### Layer 2 — Happy path tests (`IncrementalUpdateTest.java`)
Standard successful flows: prior distribution (no evidence), single soft evidence, single hard evidence, multi-variable evidence, evidence at root/middle/leaf positions, sequential changes always matching the forced baseline.

### Layer 3 — Edge cases (`IncrementalUpdateEdgeCaseTest.java`)
Boundary conditions: all variables dirty simultaneously, evidence set to the same value twice (no dirty bit), `unsetLikelyhood` then re-set, Phase 3 fallback triggers (zero potential, null previous), first ever call (not calibrated), large chain networks with evidence at extreme ends.

### Layer 4 — End-to-end (`IncrementalBayesIntegrationTest.java`)
Full Drools KieSession using the Garden network (`.xmlbif`) and DRL rules that insert/retract evidence via `insertLogical(new PropertyReference(...), bsFactory.create(...))`. Verifies the belief system's `BayesInstance` produces correct posteriors after `fireAllRules()` + `globalUpdate()`, including incremental updates triggered by retraction of rule-causing facts.

---

## Task 1: Add JMH to smallrye-bayesian pom.xml

**Files:**
- Modify: `smallrye-bayesian/pom.xml`

- [ ] **Step 1: Add JMH dependencies inside the `<dependencies>` block**

Add after the existing `assertj-core` dependency:

```xml
    <dependency>
      <groupId>org.openjdk.jmh</groupId>
      <artifactId>jmh-core</artifactId>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.openjdk.jmh</groupId>
      <artifactId>jmh-generator-annprocess</artifactId>
      <scope>test</scope>
    </dependency>
```

- [ ] **Step 2: Verify the module compiles**

```bash
mvn test-compile -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`

- [ ] **Step 3: Commit**

```bash
git add smallrye-bayesian/pom.xml
git commit -m "feat(beliefs): add JMH test dependency for benchmarks"
```

---

## Task 2: Create BayesNetworkFixtures

**Files:**
- Create: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesNetworkFixtures.java`

- [ ] **Step 1: Create the fixtures class**

```java
package io.smallrye.bayesian.bayes.benchmark;

import io.smallrye.bayesian.bayes.BayesNetwork;
import io.smallrye.bayesian.bayes.BayesVariable;
import io.smallrye.bayesian.bayes.JunctionTree;
import io.smallrye.bayesian.bayes.JunctionTreeBuilder;
import io.smallrye.bayesian.graph.Graph;
import io.smallrye.bayesian.graph.GraphNode;
import io.smallrye.bayesian.graph.impl.EdgeImpl;

/**
 * Builds reproducible synthetic Bayesian networks for benchmarking.
 */
public class BayesNetworkFixtures {

    /**
     * Builds a chain of {@code n} binary variables: var_0 → var_1 → ... → var_{n-1}.
     * var_0 has prior [0.5, 0.5]. Each subsequent variable has its predecessor as parent
     * with CPT [[0.7, 0.3], [0.4, 0.6]].
     */
    public static JunctionTree buildChain(int n) {
        if (n < 2) throw new IllegalArgumentException("Chain must have at least 2 variables");

        Graph<BayesVariable> graph = new BayesNetwork();

        @SuppressWarnings("unchecked")
        GraphNode<BayesVariable>[] nodes = new GraphNode[n];
        for (int i = 0; i < n; i++) {
            nodes[i] = graph.addNode();
        }

        // Root variable: no parents
        nodes[0].setContent(new BayesVariable<String>(
                varName(0), nodes[0].getId(),
                new String[]{"false", "true"},
                new double[][]{{0.5, 0.5}}));

        // Remaining variables: one parent each
        for (int i = 1; i < n; i++) {
            nodes[i].setContent(new BayesVariable<String>(
                    varName(i), nodes[i].getId(),
                    new String[]{"false", "true"},
                    new double[][]{{0.7, 0.3}, {0.4, 0.6}}));

            EdgeImpl edge = new EdgeImpl();
            edge.setOutGraphNode(nodes[i - 1]);
            edge.setInGraphNode(nodes[i]);
        }

        return new JunctionTreeBuilder(graph).build();
    }

    public static String varName(int index) {
        return "var_" + index;
    }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
mvn test-compile -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`

- [ ] **Step 3: Commit**

```bash
git add smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesNetworkFixtures.java
git commit -m "feat(beliefs): add BayesNetworkFixtures for benchmark network generation"
```

---

## Task 3: Add `globalUpdateForced()` to BayesInstance

**Files:**
- Modify: `smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java`

- [ ] **Step 1: Write failing tests that call `globalUpdateForced()` and cover multiple scenarios**

Create `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java`:

```java
package io.smallrye.bayesian.bayes;

import io.smallrye.bayesian.bayes.benchmark.BayesNetworkFixtures;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class IncrementalUpdateTest {

    // ── Helper ──────────────────────────────────────────────────────────────

    static void assertSameMarginals(BayesInstance<?> expected, BayesInstance<?> actual, int n) {
        for (int i = 0; i < n; i++) {
            double[] exp = expected.marginalize(BayesNetworkFixtures.varName(i)).getDistribution();
            double[] act = actual.marginalize(BayesNetworkFixtures.varName(i)).getDistribution();
            for (int j = 0; j < exp.length; j++) {
                assertThat(act[j]).as("var_%d[%d]", i, j).isCloseTo(exp[j], within(1e-10));
            }
        }
    }

    // ── Happy path: globalUpdateForced ───────────────────────────────────────

    @Test
    void globalUpdateForced_withNoEvidence_producesPriorDistribution() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(4);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.globalUpdateForced();
        // var_0 has prior [0.5, 0.5]; with no evidence all vars should reflect prior
        double[] dist = instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution();
        assertThat(dist[0]).isCloseTo(0.5, within(1e-10));
        assertThat(dist[1]).isCloseTo(0.5, within(1e-10));
    }

    @Test
    void globalUpdateForced_producesIdenticalMarginalsToGlobalUpdate_singleEvidence() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> normal = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        normal.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});

        forced.globalUpdateForced();
        normal.globalUpdate();

        assertSameMarginals(forced, normal, 8);
    }

    @Test
    void globalUpdateForced_producesIdenticalMarginalsToGlobalUpdate_multipleEvidence() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> normal = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        forced.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.3, 0.7});
        normal.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        normal.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.3, 0.7});

        forced.globalUpdateForced();
        normal.globalUpdate();

        assertSameMarginals(forced, normal, 8);
    }

    @Test
    void globalUpdateForced_isIdempotent() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});

        instance.globalUpdateForced();
        double[] first = instance.marginalize(BayesNetworkFixtures.varName(7)).getDistribution().clone();

        instance.globalUpdateForced(); // call again — must produce same result
        double[] second = instance.marginalize(BayesNetworkFixtures.varName(7)).getDistribution();

        for (int j = 0; j < first.length; j++) {
            assertThat(second[j]).isCloseTo(first[j], within(1e-10));
        }
    }

    @Test
    void globalUpdateForced_withHardEvidence_producesCorrectMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(4);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        // Hard evidence: var_0 is definitely "false"
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        instance.globalUpdateForced();
        double[] dist = instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution();
        assertThat(dist[0]).isCloseTo(1.0, within(1e-10));
        assertThat(dist[1]).isCloseTo(0.0, within(1e-10));
    }
}
```

- [ ] **Step 2: Run the test to confirm it fails with "cannot find symbol globalUpdateForced"**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress -Dtest=IncrementalUpdateTest#globalUpdateForced_producesIdenticalMarginalsToGlobalUpdate
```

Expected: compilation error — `globalUpdateForced()` does not exist yet.

- [ ] **Step 3: Add `globalUpdateForced()` to `BayesInstance.java`**

Add after the existing `globalUpdate()` method (around line 244):

```java
/**
 * Always performs the full reset + applyEvidence + collect + distribute sequence,
 * bypassing all incremental optimisations. Used by benchmarks as the permanent baseline.
 */
public void globalUpdateForced() {
    if ( !isDecided() ) {
        throw new IllegalStateException("Cannot perform global update while one or more variables are undecided");
    }
    reset();
    applyEvidence();
    globalUpdate(tree.getRoot());
    dirty = 0;
    calibrated = true;
}
```

Note: `calibrated` field does not exist yet — add it as a `private boolean calibrated;` field alongside `dirty` and `decided` (around line 42). It will be used fully in Task 4.

- [ ] **Step 4: Run the test and confirm it passes**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress -Dtest=IncrementalUpdateTest#globalUpdateForced_producesIdenticalMarginalsToGlobalUpdate
```

Expected: `BUILD SUCCESS`, test PASSES.

- [ ] **Step 5: Run the full smallrye-bayesian test suite to confirm no regressions**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`, all tests pass.

- [ ] **Step 6: Commit**

```bash
git add smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java
git commit -m "feat(beliefs): add globalUpdateForced() as permanent benchmark baseline"
```

---

## Task 4: Create BayesBenchmark and BenchmarkMain

**Files:**
- Create: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java`
- Create: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java`

- [ ] **Step 1: Create BayesBenchmark with `baselineFull` only**

```java
package io.smallrye.bayesian.bayes.benchmark;

import io.smallrye.bayesian.bayes.BayesInstance;
import io.smallrye.bayesian.bayes.JunctionTree;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class BayesBenchmark {

    @Param({"8", "16", "32", "64"})
    private int networkSize;

    // Separate instance per benchmark to avoid cross-contamination of calibrated state
    private BayesInstance<?> baselineInstance;

    private boolean toggle;

    @Setup(Level.Trial)
    public void setup() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(networkSize);

        baselineInstance = new BayesInstance<>(jt);
        baselineInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        baselineInstance.globalUpdateForced();

        toggle = false;
    }

    /**
     * Permanent baseline — always full reset+propagate. Never modified.
     * Alternates evidence each iteration so dirty is always set.
     */
    @Benchmark
    public void baselineFull(Blackhole bh) {
        toggle = !toggle;
        baselineInstance.setLikelyhood(BayesNetworkFixtures.varName(0),
                toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
        baselineInstance.globalUpdateForced();
    }
}
```

- [ ] **Step 2: Create BenchmarkMain**

```java
package io.smallrye.bayesian.bayes.benchmark;

import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

public class BenchmarkMain {

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BayesBenchmark.class.getSimpleName())
                .build();
        new Runner(opt).run();
    }
}
```

- [ ] **Step 3: Verify the benchmark compiles**

```bash
mvn test-compile -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`

- [ ] **Step 4: Record baseline numbers — run the benchmark**

```bash
mvn test-compile -pl smallrye-bayesian -am --no-transfer-progress
java -cp $(mvn -pl smallrye-bayesian -am dependency:build-classpath -q -DforceStdout 2>/dev/null):smallrye-bayesian/target/test-classes:smallrye-bayesian/target/classes \
    io.smallrye.bayesian.bayes.benchmark.BenchmarkMain 2>&1 | tee /tmp/baseline-results.txt
```

If the classpath command is too cumbersome, run from IDE or use:
```bash
mvn exec:java -pl smallrye-bayesian -Dexec.mainClass=io.smallrye.bayesian.bayes.benchmark.BenchmarkMain \
    -Dexec.classpathScope=test --no-transfer-progress
```

**Record the output in a comment in `BenchmarkMain.java`** — e.g.:
```java
// Baseline results captured 2026-04-19:
// baselineFull  n=8:  X.XX µs/op
// baselineFull  n=16: X.XX µs/op
// baselineFull  n=32: X.XX µs/op
// baselineFull  n=64: X.XX µs/op
```

- [ ] **Step 5: Commit**

```bash
git add smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java
git commit -m "feat(beliefs): add JMH benchmark skeleton with baselineFull"
```

---

## Task 5: Phase 1 — Calibration Guard

**Files:**
- Modify: `smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java`
- Modify: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java`
- Modify: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java`

- [ ] **Step 1: Write failing tests for Phase 1**

Add to `IncrementalUpdateTest.java`:

```java
// ── Phase 1: Calibration guard ───────────────────────────────────────────

static AtomicInteger countingListener(BayesInstance<?> instance) {
    AtomicInteger count = new AtomicInteger();
    instance.setGlobalUpdateListener(new GlobalUpdateListener() {
        public void beforeGlobalUpdate(CliqueState clique) { count.incrementAndGet(); }
        public void afterGlobalUpdate(CliqueState clique) {}
    });
    return count;
}

@Test
void phase1_secondCallWithNoEvidenceChange_doesNotPropagate() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> instance = new BayesInstance<>(jt);
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    AtomicInteger count = countingListener(instance);

    instance.globalUpdate();
    assertThat(count.get()).isEqualTo(1);

    instance.globalUpdate(); // nothing changed
    assertThat(count.get()).isEqualTo(1); // still 1 — returned early
}

@Test
void phase1_afterEvidenceChange_propagatesAgain() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> instance = new BayesInstance<>(jt);
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    AtomicInteger count = countingListener(instance);

    instance.globalUpdate();                                                         // 1st
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7}); // changed
    instance.globalUpdate();                                                         // 2nd
    assertThat(count.get()).isEqualTo(2);
}

@Test
void phase1_multipleNoOpCallsAllSkip() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> instance = new BayesInstance<>(jt);
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    AtomicInteger count = countingListener(instance);

    instance.globalUpdate(); // calibrates
    instance.globalUpdate(); // skip
    instance.globalUpdate(); // skip
    instance.globalUpdate(); // skip
    assertThat(count.get()).isEqualTo(1);
}

@Test
void phase1_settingSameEvidenceValueDoesNotMarkDirty() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> instance = new BayesInstance<>(jt);
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    AtomicInteger count = countingListener(instance);

    instance.globalUpdate();
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2}); // same value
    instance.globalUpdate(); // should still skip — evidence didn't change
    assertThat(count.get()).isEqualTo(1);
}

@Test
void phase1_unsetEvidenceMarksDirty() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> instance = new BayesInstance<>(jt);
    BayesVariable var0 = instance.getVariables().get(BayesNetworkFixtures.varName(0));
    instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    AtomicInteger count = countingListener(instance);

    instance.globalUpdate();
    instance.unsetLikelyhood(var0); // removes evidence → dirty
    instance.globalUpdate();        // must re-propagate
    assertThat(count.get()).isEqualTo(2);
}

@Test
void phase1_noOpSkipPreservesCorrectMarginals() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> guarded = new BayesInstance<>(jt);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    guarded.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    guarded.globalUpdate();
    guarded.globalUpdate(); // second no-op call — must not corrupt state

    assertSameMarginals(forced, guarded, 8);
}
```

- [ ] **Step 2: Run to confirm both tests fail**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalUpdateTest#globalUpdate_secondCallWithNoEvidenceChange_doesNotPropagate+globalUpdate_afterEvidenceChange_propagatesAgain"
```

Expected: both FAIL — second `globalUpdate()` call currently always propagates.

- [ ] **Step 3: Implement the calibration guard in `BayesInstance.java`**

The `calibrated` field was added in Task 3. Now wire it fully.

**In `setLikelyhood(BayesLikelyhood likelyhood)`** — add `calibrated = false` when dirty is set:

```java
public void setLikelyhood(BayesLikelyhood likelyhood) {
    int id = likelyhood.getVariable().getId();
    BayesLikelyhood old = this.likelyhoods[id];
    if ( old == null || !old.equals( likelyhood ) ) {
        this.likelyhoods[likelyhood.getVariable().getId()] = likelyhood;
        dirty = BitMaskUtil.set(dirty, id);
        calibrated = false;   // ADD THIS LINE
    }
}
```

**In `unsetLikelyhood(BayesVariable var)`** — add `calibrated = false`:

```java
public void unsetLikelyhood(BayesVariable var) {
    int id = var.getId();
    this.likelyhoods[id] = null;
    dirty = BitMaskUtil.set(dirty, id);
    calibrated = false;   // ADD THIS LINE
}
```

**In `globalUpdate()`** — add early-return guard and set `calibrated = true` at the end:

```java
public void globalUpdate() {
    if ( !isDecided() ) {
        throw new IllegalStateException("Cannot perform global update while one or more variables are undecided");
    }
    if ( calibrated && !isDirty() ) {   // ADD: Phase 1 guard
        return;
    }
    if ( isDirty() ) {
        reset();
    }
    applyEvidence();
    globalUpdate(tree.getRoot());
    dirty = 0;
    calibrated = true;   // ADD THIS LINE
}
```

- [ ] **Step 4: Run the failing tests and confirm they pass**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalUpdateTest#globalUpdate_secondCallWithNoEvidenceChange_doesNotPropagate+globalUpdate_afterEvidenceChange_propagatesAgain"
```

Expected: both PASS.

- [ ] **Step 5: Run the full test suite**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`, all tests pass.

- [ ] **Step 6: Add `noChangeUpdate` benchmark to `BayesBenchmark.java`**

Add a new `noChangeInstance` field, initialise it in `setup()`, and add the benchmark method:

```java
// Add field:
private BayesInstance<?> noChangeInstance;

// In setup(), after baselineInstance setup:
JunctionTree jt2 = BayesNetworkFixtures.buildChain(networkSize);
noChangeInstance = new BayesInstance<>(jt2);
noChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
noChangeInstance.globalUpdate(); // pre-calibrate — Phase 1 will skip subsequent calls

// New benchmark method:
/**
 * Phase 1 win: second call with no evidence change returns immediately.
 */
@Benchmark
public void noChangeUpdate(Blackhole bh) {
    noChangeInstance.globalUpdate(); // should be near-zero cost after Phase 1
}
```

- [ ] **Step 7: Run benchmarks and record Phase 1 numbers**

```bash
mvn exec:java -pl smallrye-bayesian -Dexec.mainClass=io.smallrye.bayesian.bayes.benchmark.BenchmarkMain \
    -Dexec.classpathScope=test --no-transfer-progress
```

Record `noChangeUpdate` results alongside baseline in the `BenchmarkMain.java` comment block. Expect `noChangeUpdate` to be ~50–100× faster than `baselineFull` for all network sizes.

- [ ] **Step 8: Commit**

```bash
git add smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java
git commit -m "feat(beliefs): Phase 1 calibration guard — skip globalUpdate when already calibrated"
```

---

## Task 6: Phase 2 — Dirty Subtree Fields and Snapshot

**Files:**
- Modify: `smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java`

- [ ] **Step 1: Write failing correctness, message-count, and unit tests for Phase 2**

Add to `IncrementalUpdateTest.java`. The helper `countMessages(instance)` returns an `AtomicInteger` wired to `beforeProjectAndAbsorb`:

```java
// ── Helper ──────────────────────────────────────────────────────────────

static AtomicInteger countMessages(BayesInstance<?> instance) {
    AtomicInteger count = new AtomicInteger();
    instance.setPassMessageListener(new PassMessageListener() {
        public void beforeProjectAndAbsorb(JunctionTreeClique s, JunctionTreeSeparator sep,
                                           JunctionTreeClique t, double[] old) { count.incrementAndGet(); }
        public void afterProject(JunctionTreeClique s, JunctionTreeSeparator sep,
                                 JunctionTreeClique t, double[] old) {}
        public void afterAbsorb(JunctionTreeClique s, JunctionTreeSeparator sep,
                                JunctionTreeClique t, double[] old) {}
    });
    return count;
}

// ── Phase 2: correctness (marginals match forced baseline) ───────────────

@Test
void phase2_singleEvidenceChangeAtMiddle_producesSameMarginalsAsForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    inc.globalUpdate();

    forced.setLikelyhood(BayesNetworkFixtures.varName(8), new double[]{0.9, 0.1});
    inc.setLikelyhood(BayesNetworkFixtures.varName(8), new double[]{0.9, 0.1});
    forced.globalUpdateForced();
    inc.globalUpdate();

    assertSameMarginals(forced, inc, 16);
}

@Test
void phase2_singleEvidenceChangeAtLeaf_producesSameMarginalsAsForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
    forced.globalUpdateForced();
    inc.globalUpdate();

    // Change evidence at the leaf (end of chain)
    forced.setLikelyhood(BayesNetworkFixtures.varName(15), new double[]{0.9, 0.1});
    inc.setLikelyhood(BayesNetworkFixtures.varName(15), new double[]{0.9, 0.1});
    forced.globalUpdateForced();
    inc.globalUpdate();

    assertSameMarginals(forced, inc, 16);
}

@Test
void phase2_singleEvidenceChangeAtRoot_producesSameMarginalsAsForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
    forced.globalUpdateForced();
    inc.globalUpdate();

    // Change the root variable's evidence
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.9, 0.1});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.9, 0.1});
    forced.globalUpdateForced();
    inc.globalUpdate();

    assertSameMarginals(forced, inc, 16);
}

@Test
void phase2_batchEvidenceChange_producesSameMarginalsAsForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
    forced.globalUpdateForced();
    inc.globalUpdate();

    // Change three variables simultaneously
    for (int varIdx : new int[]{2, 8, 14}) {
        forced.setLikelyhood(BayesNetworkFixtures.varName(varIdx), new double[]{0.9, 0.1});
        inc.setLikelyhood(BayesNetworkFixtures.varName(varIdx), new double[]{0.9, 0.1});
    }
    forced.globalUpdateForced();
    inc.globalUpdate();

    assertSameMarginals(forced, inc, 16);
}

@Test
void phase2_sequentialEvidenceChanges_alwaysMatchForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);

    // Round 1
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    inc.globalUpdate();
    assertSameMarginals(forced, inc, 16);

    // Round 2 — different variable
    forced.setLikelyhood(BayesNetworkFixtures.varName(5), new double[]{0.3, 0.7});
    inc.setLikelyhood(BayesNetworkFixtures.varName(5), new double[]{0.3, 0.7});
    forced.globalUpdateForced();
    inc.globalUpdate();
    assertSameMarginals(forced, inc, 16);

    // Round 3 — yet another variable
    forced.setLikelyhood(BayesNetworkFixtures.varName(12), new double[]{0.6, 0.4});
    inc.setLikelyhood(BayesNetworkFixtures.varName(12), new double[]{0.6, 0.4});
    forced.globalUpdateForced();
    inc.globalUpdate();
    assertSameMarginals(forced, inc, 16);
}

@Test
void phase2_unsetEvidenceThenReSet_producesSameMarginalsAsForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);
    BayesVariable var4 = inc.getVariables().get(BayesNetworkFixtures.varName(4));

    forced.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.9, 0.1});
    inc.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.9, 0.1});
    forced.globalUpdateForced();
    inc.globalUpdate();

    // Unset then re-set to a different value
    forced.unsetLikelyhood(forced.getVariables().get(BayesNetworkFixtures.varName(4)));
    inc.unsetLikelyhood(var4);
    forced.globalUpdateForced();
    inc.globalUpdate();
    assertSameMarginals(forced, inc, 8);

    forced.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.2, 0.8});
    inc.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.2, 0.8});
    forced.globalUpdateForced();
    inc.globalUpdate();
    assertSameMarginals(forced, inc, 8);
}

// ── Phase 2: message-count (incremental does less work) ─────────────────

@Test
void phase2_singleEvidenceChange_computesFewerCollectMessages() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(32);
    BayesInstance<?> forced = new BayesInstance<>(jt);
    BayesInstance<?> inc = new BayesInstance<>(jt);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    inc.globalUpdate();

    AtomicInteger forcedCount = countMessages(forced);
    AtomicInteger incCount    = countMessages(inc);

    forced.setLikelyhood(BayesNetworkFixtures.varName(16), new double[]{0.9, 0.1});
    inc.setLikelyhood(BayesNetworkFixtures.varName(16), new double[]{0.9, 0.1});
    forced.globalUpdateForced();
    inc.globalUpdate();

    assertThat(incCount.get())
            .as("incremental must compute fewer messages than forced full re-eval")
            .isLessThan(forcedCount.get());
}

@Test
void phase2_largerNetworkSavesMoreMessages() {
    // For a chain: collect savings grow with N. At N=64, changing var_32
    // means only ~32 collect messages vs 63 forced. Check savings increase with N.
    int[] sizes = {16, 32, 64};
    int[] savings = new int[sizes.length];

    for (int idx = 0; idx < sizes.length; idx++) {
        int n = sizes[idx];
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        forced.globalUpdateForced();
        inc.globalUpdate();

        AtomicInteger fc = countMessages(forced);
        AtomicInteger ic = countMessages(inc);

        forced.setLikelyhood(BayesNetworkFixtures.varName(n / 2), new double[]{0.9, 0.1});
        inc.setLikelyhood(BayesNetworkFixtures.varName(n / 2), new double[]{0.9, 0.1});
        forced.globalUpdateForced();
        inc.globalUpdate();

        savings[idx] = fc.get() - ic.get();
    }

    // Savings should increase (or at least not decrease) with network size
    assertThat(savings[1]).isGreaterThanOrEqualTo(savings[0]);
    assertThat(savings[2]).isGreaterThanOrEqualTo(savings[1]);
}
```

Also create `IncrementalUpdateEdgeCaseTest.java`:

```java
package io.smallrye.bayesian.bayes;

import io.smallrye.bayesian.bayes.benchmark.BayesNetworkFixtures;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static io.smallrye.bayesian.bayes.IncrementalUpdateTest.assertSameMarginals;

public class IncrementalUpdateEdgeCaseTest {

    @Test
    void phase2_minimumNetworkSize_twoVariables_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(2);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.9, 0.1});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.9, 0.1});
        forced.globalUpdateForced();
        inc.globalUpdate();

        // Change same variable
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
        forced.globalUpdateForced();
        inc.globalUpdate();

        assertSameMarginals(forced, inc, 2);
    }

    @Test
    void phase2_allVariablesDirty_correctMarginals() {
        int n = 8;
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        // Initial calibration
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        forced.globalUpdateForced();
        inc.globalUpdate();

        // Make ALL variables dirty
        for (int i = 0; i < n; i++) {
            forced.setLikelyhood(BayesNetworkFixtures.varName(i), new double[]{0.8, 0.2});
            inc.setLikelyhood(BayesNetworkFixtures.varName(i), new double[]{0.8, 0.2});
        }
        forced.globalUpdateForced();
        inc.globalUpdate();

        assertSameMarginals(forced, inc, n);
    }

    @Test
    void phase2_largishNetwork_100variables_correctMarginals() {
        int n = 100;
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        forced.globalUpdateForced();
        inc.globalUpdate();

        forced.setLikelyhood(BayesNetworkFixtures.varName(50), new double[]{0.6, 0.4});
        inc.setLikelyhood(BayesNetworkFixtures.varName(50), new double[]{0.6, 0.4});
        forced.globalUpdateForced();
        inc.globalUpdate();

        assertSameMarginals(forced, inc, n);
    }

    @Test
    void phase2_firstCallIsAlwaysFullPath_correctMarginals() {
        // On the very first globalUpdate() call, calibrated=false so the full path runs.
        // This should produce the same result as globalUpdateForced().
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(3), new double[]{0.7, 0.3});
        inc.setLikelyhood(BayesNetworkFixtures.varName(3), new double[]{0.7, 0.3});

        forced.globalUpdateForced(); // first call on forced
        inc.globalUpdate();           // first call on inc — must take full path

        assertSameMarginals(forced, inc, 8);
    }

    @Test
    void phase2_hardEvidenceFollowedBySoftEvidence_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        // Round 1: hard evidence
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        forced.globalUpdateForced();
        inc.globalUpdate();
        assertSameMarginals(forced, inc, 8);

        // Round 2: change to soft evidence on same variable
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.6, 0.4});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.6, 0.4});
        forced.globalUpdateForced();
        inc.globalUpdate();
        assertSameMarginals(forced, inc, 8);
    }
}
```

- [ ] **Step 2: Run to confirm both Phase 2 tests fail**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalUpdateTest#incrementalCollect_afterSingleEvidenceChange_producesSameMarginalsAsForced+incrementalCollect_afterSingleEvidenceChange_computesFewerMessages"
```

Expected: correctness test PASSES (full path still runs — same result), message-count test FAILS (same count as forced — incremental not yet implemented).

- [ ] **Step 3: Add Phase 2 fields to `BayesInstance.java`**

Add alongside the existing `cliqueStates`, `separatorStates` fields (around line 45):

```java
// Phase 2: dirty subtree tracking + separator snapshots
private boolean[] subtreeHasDirty;
private double[][] sepPotSnapshots;
```

- [ ] **Step 4: Allocate Phase 2 fields in the constructor**

In the constructor `BayesInstance(JunctionTree tree)`, after `separatorStates` is populated (after line 78), add:

```java
subtreeHasDirty = new boolean[tree.getJunctionTreeNodes().length];

sepPotSnapshots = new double[tree.getJunctionTreeSeparators().length][];
for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
    sepPotSnapshots[sep.getId()] = new double[separatorStates[sep.getId()].getPotentials().length];
}
```

- [ ] **Step 5: Implement `snapshotSeparators()`**

Add after the constructor:

```java
private void snapshotSeparators() {
    for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
        double[] current = separatorStates[sep.getId()].getPotentials();
        System.arraycopy(current, 0, sepPotSnapshots[sep.getId()], 0, current.length);
    }
}
```

- [ ] **Step 6: Implement `computeDirtySubtrees()`**

```java
private void computeDirtySubtrees() {
    Arrays.fill(subtreeHasDirty, false);
    for (int varId = 0; varId < likelyhoods.length; varId++) {
        if (BitMaskUtil.isSet(dirty, varId)) {
            int cliqueId = varStates[varId].getVariable().getFamily();
            markSubtreeDirtyUpward(tree.getJunctionTreeNodes()[cliqueId]);
        }
    }
}

private void markSubtreeDirtyUpward(JunctionTreeClique clique) {
    if (subtreeHasDirty[clique.getId()]) {
        return; // already marked — ancestor path already processed
    }
    subtreeHasDirty[clique.getId()] = true;
    JunctionTreeSeparator parentSep = clique.getParentSeparator();
    if (parentSep != null) {
        markSubtreeDirtyUpward(parentSep.getParent());
    }
}
```

- [ ] **Step 7: Verify compilation**

```bash
mvn test-compile -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`

- [ ] **Step 8: Commit Phase 2 fields and helpers (before wiring — tests still fail on message count)**

```bash
git add smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java
git commit -m "feat(beliefs): Phase 2 — add dirty subtree fields, snapshotSeparators, computeDirtySubtrees"
```

---

## Task 7: Phase 2 — Incremental Collect

**Files:**
- Modify: `smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java`
- Modify: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java`

- [ ] **Step 1: Implement `injectSnapshotAndAbsorb()`**

This method replaces `passMessage` for clean subtrees. It restores the snapshotted separator value and absorbs it into the target clique using the post-reset (all-1.0) separator as `oldSepPots`.

Add to `BayesInstance.java`:

```java
/**
 * For a clean (non-dirty) subtree: restore the calibrated separator snapshot
 * and absorb it into the target clique. After reset(), separator potentials are
 * all 1.0, so oldSepPots = {1.0,...} and absorb gives:
 *   target[i] = targetInit[i] * snapshot[j] / 1.0
 * which correctly injects the previously calibrated message.
 */
private void injectSnapshotAndAbsorb(JunctionTreeSeparator sep, JunctionTreeClique targetClique) {
    double[] sepPots = separatorStates[sep.getId()].getPotentials();

    // oldSepPots = current state before injection = post-reset = all 1.0
    double[] oldSepPots = Arrays.copyOf(sepPots, sepPots.length); // all 1.0 after reset

    // Restore the calibrated snapshot into the separator
    System.arraycopy(sepPotSnapshots[sep.getId()], 0, sepPots, 0, sepPots.length);

    // Absorb snapshot into target (no project — source clique not involved)
    BayesVariable[] sepVars = sep.getValues().toArray(new BayesVariable[sep.getValues().size()]);
    absorb(sepVars, cliqueStates[targetClique.getId()], separatorStates[sep.getId()], oldSepPots);
}
```

- [ ] **Step 2: Implement `collectChildEvidenceIncremental()`**

This replaces `collectChildEvidence()` with a dirty-aware version that skips clean subtrees:

```java
private void collectChildEvidenceIncremental(JunctionTreeClique clique, JunctionTreeClique startClique) {
    List<JunctionTreeSeparator> seps = clique.getChildren();
    for (JunctionTreeSeparator sep : seps) {
        collectChildEvidenceIncremental(sep.getChild(), startClique);
    }

    if (clique.getParentSeparator() != null && clique != startClique) {
        JunctionTreeSeparator parentSep = clique.getParentSeparator();
        JunctionTreeClique parent = parentSep.getParent();

        if (subtreeHasDirty[clique.getId()]) {
            passMessage(clique, parentSep, parent); // dirty: recompute
        } else {
            injectSnapshotAndAbsorb(parentSep, parent); // clean: inject snapshot
        }
    }
}
```

- [ ] **Step 3: Implement `collectEvidenceIncremental()`**

`globalUpdate()` always passes the root clique, which has no parent, so `collectParentEvidence` is never invoked in the normal flow. For the parent path we therefore fall back to the existing full `collectParentEvidence` — it is unreachable during `globalUpdate()` but kept for correctness if ever called with a non-root clique in the future.

```java
private void collectEvidenceIncremental(JunctionTreeClique clique) {
    if (clique.getParentSeparator() != null) {
        // Non-root start clique: parent path unchanged (full, not incremental).
        // globalUpdate() never reaches here because root has no parent.
        collectParentEvidence(clique.getParentSeparator().getParent(),
                clique.getParentSeparator(), clique, clique);
    }
    collectChildEvidenceIncremental(clique, clique);
}
```

- [ ] **Step 5: Wire Phase 2 into `globalUpdate()`**

Replace the existing `globalUpdate()` body with:

```java
public void globalUpdate() {
    if ( !isDecided() ) {
        throw new IllegalStateException("Cannot perform global update while one or more variables are undecided");
    }
    if ( calibrated && !isDirty() ) {                   // Phase 1: skip entirely
        return;
    }

    if ( calibrated ) {                                 // Phase 2: setup incremental collect
        snapshotSeparators();
        computeDirtySubtrees();
    }

    if ( isDirty() ) {
        reset();
    }
    applyEvidence();

    if ( calibrated ) {                                 // Phase 2: incremental collect
        collectEvidenceIncremental(tree.getRoot());
    } else {
        collectEvidence(tree.getRoot());                // first call: full collect
    }
    distributeEvidence(tree.getRoot());                 // always full distribute

    dirty = 0;
    calibrated = true;
    Arrays.fill(subtreeHasDirty, false);
}
```

Wait — there is a subtlety: `calibrated` starts as `false` on the first call, so the first call goes through the full path. After that, `calibrated = true`, so subsequent dirty calls use Phase 2. This is correct.

- [ ] **Step 6: Run the Phase 2 tests and confirm both pass**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalUpdateTest#incrementalCollect_afterSingleEvidenceChange_producesSameMarginalsAsForced+incrementalCollect_afterSingleEvidenceChange_computesFewerMessages"
```

Expected: both PASS.

- [ ] **Step 7: Run the full test suite**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`, all tests pass.

- [ ] **Step 8: Add `singleEvidenceChange` and `batchEvidenceChange` benchmarks to `BayesBenchmark.java`**

Add fields and setup:

```java
private BayesInstance<?> singleChangeInstance;
private BayesInstance<?> batchChangeInstance;
private boolean toggleSingle;
private boolean toggleBatch;

// In setup(), add:
JunctionTree jt3 = BayesNetworkFixtures.buildChain(networkSize);
singleChangeInstance = new BayesInstance<>(jt3);
singleChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
singleChangeInstance.globalUpdate();
toggleSingle = false;

JunctionTree jt4 = BayesNetworkFixtures.buildChain(networkSize);
batchChangeInstance = new BayesInstance<>(jt4);
batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
batchChangeInstance.globalUpdate();
toggleBatch = false;
```

Add benchmark methods:

```java
@Benchmark
public void singleEvidenceChange(Blackhole bh) {
    toggleSingle = !toggleSingle;
    singleChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0),
            toggleSingle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
    singleChangeInstance.globalUpdate();
}

@Benchmark
public void batchEvidenceChange(Blackhole bh) {
    toggleBatch = !toggleBatch;
    double[] ev = toggleBatch ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7};
    // Change variables at 0%, 33%, 66% positions through the chain
    batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), ev);
    batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(networkSize / 3), ev);
    batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(2 * networkSize / 3), ev);
    batchChangeInstance.globalUpdate();
}
```

- [ ] **Step 9: Run benchmarks and record Phase 2 numbers**

```bash
mvn exec:java -pl smallrye-bayesian -Dexec.mainClass=io.smallrye.bayesian.bayes.benchmark.BenchmarkMain \
    -Dexec.classpathScope=test --no-transfer-progress
```

Record `singleEvidenceChange` and `batchEvidenceChange` results alongside baseline in `BenchmarkMain.java`. Expect `singleEvidenceChange` to be noticeably faster than `baselineFull`, with larger gains at `n=64` than `n=8`.

- [ ] **Step 10: Commit**

```bash
git add smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java
git commit -m "feat(beliefs): Phase 2 incremental collect — skip clean subtree messages using separator snapshots"
```

---

## Task 8: Phase 3 — Fast Evidence Retraction

**Files:**
- Modify: `smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java`
- Modify: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java`
- Modify: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java`
- Modify: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java`

- [ ] **Step 1: Write failing correctness tests for Phase 3 — happy path, fallback, sequential, and edge cases**

Add to `IncrementalUpdateTest.java`:

```java
// ── Phase 3: correctness ─────────────────────────────────────────────────

@Test
void phase3_softToSoftEvidenceChange_takesRetractionPath_correctMarginals() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    retract.globalUpdate();

    // Soft → soft: retraction path should apply
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
    forced.globalUpdateForced();
    retract.globalUpdate();

    assertSameMarginals(forced, retract, 16);
}

@Test
void phase3_hardEvidenceFallsBackToPhase2_correctMarginals() {
    // Hard evidence (zero entry) → calibrated clique has a zero → isRetractable() = false
    // → falls back to Phase 2 path, must still produce correct result
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    retract.globalUpdate();

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0}); // hard
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
    forced.globalUpdateForced();
    retract.globalUpdate();

    assertSameMarginals(forced, retract, 8);
}

@Test
void phase3_firstEvidenceInsertionFallsBackToPhase2_correctMarginals() {
    // previousLikelyhoods[varId] == null on first insertion → isRetractable() = false
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    // Initial calibration with no evidence
    forced.globalUpdateForced();
    retract.globalUpdate();

    // Now insert evidence for the first time (no previous)
    forced.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.9, 0.1});
    retract.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.9, 0.1});
    forced.globalUpdateForced();
    retract.globalUpdate();

    assertSameMarginals(forced, retract, 8);
}

@Test
void phase3_multipleDirtyVariables_correctMarginals() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.setLikelyhood(BayesNetworkFixtures.varName(8), new double[]{0.6, 0.4});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    retract.setLikelyhood(BayesNetworkFixtures.varName(8), new double[]{0.6, 0.4});
    forced.globalUpdateForced();
    retract.globalUpdate();

    // Change both simultaneously — retraction applies to both
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
    forced.setLikelyhood(BayesNetworkFixtures.varName(8), new double[]{0.2, 0.8});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
    retract.setLikelyhood(BayesNetworkFixtures.varName(8), new double[]{0.2, 0.8});
    forced.globalUpdateForced();
    retract.globalUpdate();

    assertSameMarginals(forced, retract, 16);
}

@Test
void phase3_sequentialChanges_alwaysMatchForced() {
    JunctionTree jt = BayesNetworkFixtures.buildChain(16);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    // Seed
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced();
    retract.globalUpdate();
    assertSameMarginals(forced, retract, 16);

    // Several soft→soft retraction rounds
    double[][] evidenceSeq = {{0.3,0.7},{0.6,0.4},{0.9,0.1},{0.1,0.9}};
    for (double[] ev : evidenceSeq) {
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), ev);
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), ev);
        forced.globalUpdateForced();
        retract.globalUpdate();
        assertSameMarginals(forced, retract, 16);
    }
}

@Test
void phase3_retractionThenFallback_thenRetractionAgain_correctMarginals() {
    // soft → soft (retraction) → hard (fallback) → soft (retraction again)
    JunctionTree jt = BayesNetworkFixtures.buildChain(8);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
    forced.globalUpdateForced(); retract.globalUpdate();
    assertSameMarginals(forced, retract, 8);

    // Soft → soft (retraction)
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.4, 0.6});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.4, 0.6});
    forced.globalUpdateForced(); retract.globalUpdate();
    assertSameMarginals(forced, retract, 8);

    // Soft → hard (fallback to Phase 2)
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
    forced.globalUpdateForced(); retract.globalUpdate();
    assertSameMarginals(forced, retract, 8);

    // Hard → soft (retraction path if calibrated pots now have no zeros after the reset in fallback)
    forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.7, 0.3});
    retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.7, 0.3});
    forced.globalUpdateForced(); retract.globalUpdate();
    assertSameMarginals(forced, retract, 8);
}

@Test
void phase3_noEvidence_priorDistribution_correctMarginals() {
    // Phase 3 enabled but no evidence at all — first call, no retraction
    JunctionTree jt = BayesNetworkFixtures.buildChain(4);
    BayesInstance<?> forced  = new BayesInstance<>(jt);
    BayesInstance<?> retract = new BayesInstance<>(jt, true);

    forced.globalUpdateForced();
    retract.globalUpdate();

    assertSameMarginals(forced, retract, 4);
}
```

- [ ] **Step 2: Run to confirm Phase 3 tests fail (constructor doesn't accept boolean yet)**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalUpdateTest#fastRetraction_afterEvidenceChange_producesSameMarginalsAsForced+fastRetraction_withHardEvidence_fallsBackToPhase2WithCorrectResult"
```

Expected: compilation error — `BayesInstance(JunctionTree, boolean)` does not exist.

- [ ] **Step 3: Add Phase 3 fields and constructor to `BayesInstance.java`**

Add fields alongside the Phase 2 fields:

```java
// Phase 3: calibrated snapshots for fast evidence retraction
private final boolean enableFastRetract;
private double[][] calibratedCliquePots;
private double[][] calibratedSepPots;
```

Add a new constructor after the existing two-argument constructor (`BayesInstance(JunctionTree tree, Class<T> targetClass)`):

```java
public BayesInstance(JunctionTree tree, boolean enableFastRetract) {
    this(tree);
    this.enableFastRetract = enableFastRetract;
    if (enableFastRetract) {
        calibratedCliquePots = new double[tree.getJunctionTreeNodes().length][];
        for (JunctionTreeClique clique : tree.getJunctionTreeNodes()) {
            calibratedCliquePots[clique.getId()] =
                    new double[cliqueStates[clique.getId()].getPotentials().length];
        }
        calibratedSepPots = new double[tree.getJunctionTreeSeparators().length][];
        for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
            calibratedSepPots[sep.getId()] =
                    new double[separatorStates[sep.getId()].getPotentials().length];
        }
    }
}
```

Change the `enableFastRetract` field from `private final boolean` to `private boolean` and initialise it in the primary constructor:

```java
// In BayesInstance(JunctionTree tree), add:
this.enableFastRetract = false;
```

- [ ] **Step 4: Implement `saveCalibrated()`**

```java
private void saveCalibrated() {
    if (!enableFastRetract) return;
    for (JunctionTreeClique clique : tree.getJunctionTreeNodes()) {
        double[] src = cliqueStates[clique.getId()].getPotentials();
        System.arraycopy(src, 0, calibratedCliquePots[clique.getId()], 0, src.length);
    }
    for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
        double[] src = separatorStates[sep.getId()].getPotentials();
        System.arraycopy(src, 0, calibratedSepPots[sep.getId()], 0, src.length);
    }
}
```

- [ ] **Step 5: Add `previousLikelyhoods` field and update `setLikelyhood()` / `unsetLikelyhood()`**

Add field alongside `likelyhoods`:

```java
private BayesLikelyhood[] previousLikelyhoods;
```

Allocate in the primary constructor after `likelyhoods`:

```java
previousLikelyhoods = new BayesLikelyhood[graph.size()];
```

Update `setLikelyhood(BayesLikelyhood likelyhood)` to save the old likelihood before replacing:

```java
public void setLikelyhood(BayesLikelyhood likelyhood) {
    int id = likelyhood.getVariable().getId();
    BayesLikelyhood old = this.likelyhoods[id];
    if (old == null || !old.equals(likelyhood)) {
        previousLikelyhoods[id] = old;              // ADD: save previous
        this.likelyhoods[id] = likelyhood;
        dirty = BitMaskUtil.set(dirty, id);
        calibrated = false;
    }
}
```

Update `unsetLikelyhood(BayesVariable var)`:

```java
public void unsetLikelyhood(BayesVariable var) {
    int id = var.getId();
    previousLikelyhoods[id] = this.likelyhoods[id]; // ADD: save previous
    this.likelyhoods[id] = null;
    dirty = BitMaskUtil.set(dirty, id);
    calibrated = false;
}
```

- [ ] **Step 6: Implement `isRetractable()`**

Phase 3 activates only when every dirty variable had a non-null previous likelihood (evidence change, not first insertion) and no calibrated clique potential is zero (division safety):

```java
private boolean isRetractable() {
    for (int varId = 0; varId < likelyhoods.length; varId++) {
        if (!BitMaskUtil.isSet(dirty, varId)) continue;

        // Must have a previous likelihood to retract
        if (previousLikelyhoods[varId] == null) return false;

        // Check calibrated clique potentials have no zeros (division safety)
        int cliqueId = varStates[varId].getVariable().getFamily();
        for (double v : calibratedCliquePots[cliqueId]) {
            if (v == 0.0) return false;
        }
    }
    return true;
}
```

- [ ] **Step 7: Implement `retractionUpdate()`**

This divides out the old evidence, multiplies in the new evidence, then uses the Phase 2 collect path for the dirty subtrees:

```java
private void retractionUpdate() {
    // Apply retraction: divide out old, multiply in new, for each dirty variable's clique
    for (int varId = 0; varId < likelyhoods.length; varId++) {
        if (!BitMaskUtil.isSet(dirty, varId)) continue;

        int cliqueId = varStates[varId].getVariable().getFamily();
        double[] cliquePots = cliqueStates[cliqueId].getPotentials();

        // Restore calibrated potentials as starting point
        System.arraycopy(calibratedCliquePots[cliqueId], 0, cliquePots, 0, cliquePots.length);

        // Divide out old evidence
        BayesLikelyhood old = previousLikelyhoods[varId];
        if (old != null) {
            old.divideFrom(cliquePots);
        }

        // Multiply in new evidence
        BayesLikelyhood newL = likelyhoods[varId];
        if (newL != null) {
            newL.multiplyInto(cliquePots);
        }

        BayesAbsorption.normalize(cliquePots);
    }

    // Restore calibrated separator potentials for non-dirty subtrees
    // (replaces snapshotSeparators() — calibratedSepPots IS the snapshot)
    for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
        double[] src = calibratedSepPots[sep.getId()];
        System.arraycopy(src, 0, sepPotSnapshots[sep.getId()], 0, src.length);
    }

    // Use Phase 2 incremental collect (dirty subtree tracking already computed)
    collectEvidenceIncremental(tree.getRoot());
    distributeEvidence(tree.getRoot());
}
```

This requires `BayesLikelyhood.divideFrom(double[])`. Check if that method exists. If it does not, add it to `BayesLikelyhood.java` mirroring the existing `multiplyInto(double[])` but dividing instead of multiplying:

```java
// In BayesLikelyhood.java — add if not present:
public void divideFrom(double[] potentials) {
    // Mirror of multiplyInto but divides each element
    // Use the same index mapping as multiplyInto
    // (copy the multiplyInto implementation, replace *= with /=)
}
```

Check `BayesLikelyhood.multiplyInto()` first and replicate its index-mapping logic for division.

- [ ] **Step 8: Wire Phase 3 into `globalUpdate()`**

Replace the `globalUpdate()` body:

```java
public void globalUpdate() {
    if ( !isDecided() ) {
        throw new IllegalStateException("Cannot perform global update while one or more variables are undecided");
    }
    if ( calibrated && !isDirty() ) {                                    // Phase 1
        return;
    }

    if ( calibrated && enableFastRetract ) {
        computeDirtySubtrees();
        if ( isRetractable() ) {                                         // Phase 3
            retractionUpdate();
            saveCalibrated();
            dirty = 0;
            calibrated = true;
            Arrays.fill(subtreeHasDirty, false);
            return;
        }
    }

    if ( calibrated ) {                                                  // Phase 2 setup
        snapshotSeparators();
        computeDirtySubtrees();                                          // recompute if not done above
    }

    if ( isDirty() ) {
        reset();
    }
    applyEvidence();

    if ( calibrated ) {
        collectEvidenceIncremental(tree.getRoot());                      // Phase 2 collect
    } else {
        collectEvidence(tree.getRoot());
    }
    distributeEvidence(tree.getRoot());

    if ( enableFastRetract ) {
        saveCalibrated();
    }

    dirty = 0;
    calibrated = true;
    Arrays.fill(subtreeHasDirty, false);
}
```

Note: avoid calling `computeDirtySubtrees()` twice when Phase 3 is enabled but `isRetractable()` returns false. Guard with a local boolean or restructure:

```java
boolean dirtySubtreesComputed = false;
if ( calibrated && enableFastRetract ) {
    computeDirtySubtrees();
    dirtySubtreesComputed = true;
    if ( isRetractable() ) { ... return; }
}
if ( calibrated ) {
    snapshotSeparators();
    if ( !dirtySubtreesComputed ) computeDirtySubtrees();
}
```

Also update `globalUpdateForced()` to call `saveCalibrated()` when Phase 3 is enabled:

```java
public void globalUpdateForced() {
    ...
    if (enableFastRetract) saveCalibrated();
    dirty = 0;
    calibrated = true;
}
```

- [ ] **Step 9: Run the Phase 3 tests**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalUpdateTest#fastRetraction_afterEvidenceChange_producesSameMarginalsAsForced+fastRetraction_withHardEvidence_fallsBackToPhase2WithCorrectResult"
```

Expected: both PASS.

- [ ] **Step 10: Run the full test suite**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`, all tests pass.

- [ ] **Step 11: Add Phase 3 benchmark to `BayesBenchmark.java`**

Add field and setup:

```java
private BayesInstance<?> retractionInstance;
private boolean toggleRetract;

// In setup():
JunctionTree jt5 = BayesNetworkFixtures.buildChain(networkSize);
retractionInstance = new BayesInstance<>(jt5, true); // Phase 3 enabled
retractionInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
retractionInstance.globalUpdate();
toggleRetract = false;
```

Add benchmark method:

```java
/**
 * Phase 3: fast retraction — divides out old evidence, no full reset.
 */
@Benchmark
public void fastRetraction(Blackhole bh) {
    toggleRetract = !toggleRetract;
    retractionInstance.setLikelyhood(BayesNetworkFixtures.varName(0),
            toggleRetract ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
    retractionInstance.globalUpdate();
}
```

- [ ] **Step 12: Run final benchmarks and record Phase 3 numbers**

```bash
mvn exec:java -pl smallrye-bayesian -Dexec.mainClass=io.smallrye.bayesian.bayes.benchmark.BenchmarkMain \
    -Dexec.classpathScope=test --no-transfer-progress
```

Record `fastRetraction` results in `BenchmarkMain.java`. Compare all four benchmark methods at each network size: `baselineFull` vs `singleEvidenceChange` (Phase 2) vs `fastRetraction` (Phase 3). The Phase 3 gain over Phase 2 will be most visible at larger network sizes where the reset loop is more expensive.

- [ ] **Step 13: Commit**

```bash
git add smallrye-bayesian/src/main/java/org/drools/beliefs/bayes/BayesInstance.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/IncrementalUpdateTest.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BayesBenchmark.java \
        smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/benchmark/BenchmarkMain.java
git commit -m "feat(beliefs): Phase 3 fast evidence retraction — divide/multiply in-place, no full reset"
```

---

## Task 9: End-to-End Integration Tests via KieSession

**Files:**
- Create: `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/integration/IncrementalBayesIntegrationTest.java`

These tests use the full Drools stack: KnowledgeBuilder loads a `.bayes` file + DRL rules, a `KieSession` fires rules that insert evidence via `insertLogical(new PropertyReference(...), bsFactory.create(...))`, and the `BayesRuntime` computes posteriors. They verify that the incremental path produces the same results as the forced path in a real session, including after retraction of rule-triggering facts.

Reference pattern: see existing `BayesBeliefSystemTest.java` and `Garden.java` / `Garden.xmlbif` in `src/test/java/org/drools/beliefs/bayes/integration/`.

- [ ] **Step 1: Write the end-to-end tests**

Create `smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/integration/IncrementalBayesIntegrationTest.java`:

```java
package io.smallrye.bayesian.bayes.integration;

import io.smallrye.bayesian.bayes.BayesInstance;
import io.smallrye.bayesian.bayes.BayesModeFactory;
import io.smallrye.bayesian.bayes.BayesModeFactoryImpl;
import io.smallrye.bayesian.bayes.PropertyReference;
import io.smallrye.bayesian.bayes.runtime.BayesRuntime;
import org.drools.core.BeliefSystemType;
import org.drools.core.RuleSessionConfiguration;
import org.junit.jupiter.api.Test;
import org.kie.api.io.ResourceType;
import org.kie.api.runtime.KieSession;
import org.kie.api.runtime.KieSessionConfiguration;
import org.kie.api.runtime.rule.FactHandle;
import org.kie.internal.builder.KnowledgeBuilder;
import org.kie.internal.builder.KnowledgeBuilderFactory;
import org.kie.internal.io.ResourceFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * End-to-end tests: full KieSession + DRL rules + BayesRuntime + incremental globalUpdate().
 * Uses the Garden network (4 variables: Cloudy, Sprinkler, Rain, WetGrass).
 * Reference: BayesBeliefSystemTest for session-building boilerplate.
 */
public class IncrementalBayesIntegrationTest {

    // ── Session builder (mirrors BayesBeliefSystemTest boilerplate) ──────────

    private static KieSession buildSession(String drl) {
        KnowledgeBuilder kbuilder = KnowledgeBuilderFactory.newKnowledgeBuilder();
        kbuilder.add(ResourceFactory.newClassPathResource("Garden.xmlbif",
                IncrementalBayesIntegrationTest.class), ResourceType.BAYES);
        kbuilder.add(ResourceFactory.newByteArrayResource(drl.getBytes()), ResourceType.DRL);
        if (kbuilder.hasErrors()) {
            throw new RuntimeException(kbuilder.getErrors().toString());
        }

        KieSessionConfiguration ksconf = org.kie.api.KieServices.Factory.get()
                .newKieSessionConfiguration();
        ksconf.setOption(RuleSessionConfiguration.BeliefSystem.get(BeliefSystemType.BAYES));

        return kbuilder.newKnowledgeBase().newKieSession(ksconf, null);
    }

    private static final String GARDEN_PACKAGE = Garden.class.getPackageName();

    // ── Happy path: rule fires, evidence set, posteriors updated ─────────────

    @Test
    void e2e_ruleInsertsEvidence_posteriorUpdated() {
        String drl =
            "package " + GARDEN_PACKAGE + "; " +
            "import " + Garden.class.getCanonicalName() + "; " +
            "import " + PropertyReference.class.getCanonicalName() + "; " +
            "global " + BayesModeFactory.class.getCanonicalName() + " bsFactory; " +
            "rule 'cloudy' when String(this == 'cloudy') g : Garden() then " +
            "    insertLogical(new PropertyReference(g, 'cloudy'), bsFactory.create(new double[]{1.0, 0.0})); " +
            "end ";

        KieSession ksession = buildSession(drl);
        BayesModeFactory bsFactory = new BayesModeFactoryImpl(ksession);
        ksession.setGlobal("bsFactory", bsFactory);

        Garden garden = new Garden();
        ksession.insert(garden);
        ksession.fireAllRules(); // no trigger yet

        BayesRuntime bayesRuntime = ksession.getKieRuntime(BayesRuntime.class);
        BayesInstance<Garden> instance = bayesRuntime.createInstance(Garden.class);
        instance.globalUpdate();
        double[] cloudyPrior = instance.marginalize("cloudy").getDistribution().clone();

        // Insert trigger — rule fires, evidence inserted
        ksession.insert("cloudy");
        ksession.fireAllRules();
        instance.globalUpdate(); // incremental: only cloudy changed
        double[] cloudyPosterior = instance.marginalize("cloudy").getDistribution();

        // After hard evidence cloudy=true, P(cloudy=true) should be 1.0
        assertThat(cloudyPosterior[1]).isCloseTo(1.0, within(1e-6));
        // Posterior must differ from prior
        assertThat(cloudyPosterior[0]).isNotCloseTo(cloudyPrior[0], within(1e-3));

        ksession.dispose();
    }

    @Test
    void e2e_retractingTrigger_removesEvidence_posteriorReverts() {
        String drl =
            "package " + GARDEN_PACKAGE + "; " +
            "import " + Garden.class.getCanonicalName() + "; " +
            "import " + PropertyReference.class.getCanonicalName() + "; " +
            "global " + BayesModeFactory.class.getCanonicalName() + " bsFactory; " +
            "rule 'sprinkler' when String(this == 'sprinkler') g : Garden() then " +
            "    insertLogical(new PropertyReference(g, 'sprinkler'), bsFactory.create(new double[]{1.0, 0.0})); " +
            "end ";

        KieSession ksession = buildSession(drl);
        BayesModeFactory bsFactory = new BayesModeFactoryImpl(ksession);
        ksession.setGlobal("bsFactory", bsFactory);

        Garden garden = new Garden();
        ksession.insert(garden);
        ksession.fireAllRules();

        BayesRuntime bayesRuntime = ksession.getKieRuntime(BayesRuntime.class);
        BayesInstance<Garden> instance = bayesRuntime.createInstance(Garden.class);
        instance.globalUpdate();
        double[] sprinklerPrior = instance.marginalize("sprinkler").getDistribution().clone();

        // Insert trigger
        FactHandle triggerHandle = ksession.insert("sprinkler");
        ksession.fireAllRules();
        instance.globalUpdate();
        double[] withEvidence = instance.marginalize("sprinkler").getDistribution().clone();
        assertThat(withEvidence[1]).isCloseTo(1.0, within(1e-6)); // hard evidence

        // Retract trigger — evidence should be removed
        ksession.delete(triggerHandle);
        ksession.fireAllRules();
        instance.globalUpdate();
        double[] afterRetract = instance.marginalize("sprinkler").getDistribution();

        // Should revert close to prior (within floating-point rounding)
        assertThat(afterRetract[0]).isCloseTo(sprinklerPrior[0], within(1e-6));
        assertThat(afterRetract[1]).isCloseTo(sprinklerPrior[1], within(1e-6));

        ksession.dispose();
    }

    @Test
    void e2e_multipleRulesInsertSameEvidence_conflictResolved_correctPosterior() {
        // Two rules both set sprinkler=true — overlapping insertLogical
        String drl =
            "package " + GARDEN_PACKAGE + "; " +
            "import " + Garden.class.getCanonicalName() + "; " +
            "import " + PropertyReference.class.getCanonicalName() + "; " +
            "global " + BayesModeFactory.class.getCanonicalName() + " bsFactory; " +
            "rule 'r1' when String(this == 'r1') g : Garden() then " +
            "    insertLogical(new PropertyReference(g, 'sprinkler'), bsFactory.create(new double[]{1.0, 0.0})); " +
            "end " +
            "rule 'r2' when String(this == 'r2') g : Garden() then " +
            "    insertLogical(new PropertyReference(g, 'sprinkler'), bsFactory.create(new double[]{1.0, 0.0})); " +
            "end ";

        KieSession ksession = buildSession(drl);
        BayesModeFactory bsFactory = new BayesModeFactoryImpl(ksession);
        ksession.setGlobal("bsFactory", bsFactory);

        Garden garden = new Garden();
        ksession.insert(garden);

        // Both triggers fire
        ksession.insert("r1");
        ksession.insert("r2");
        ksession.fireAllRules();

        BayesRuntime bayesRuntime = ksession.getKieRuntime(BayesRuntime.class);
        BayesInstance<Garden> instance = bayesRuntime.createInstance(Garden.class);
        assertThat(instance.isDecided()).isTrue(); // non-conflicting: both agree
        instance.globalUpdate();

        double[] sprinkler = instance.marginalize("sprinkler").getDistribution();
        assertThat(sprinkler[1]).isCloseTo(1.0, within(1e-6));

        ksession.dispose();
    }

    @Test
    void e2e_incrementalMatchesForcedAfterEachRuleFire() {
        // Verify that after each rule trigger, globalUpdate() matches globalUpdateForced()
        String drl =
            "package " + GARDEN_PACKAGE + "; " +
            "import " + Garden.class.getCanonicalName() + "; " +
            "import " + PropertyReference.class.getCanonicalName() + "; " +
            "global " + BayesModeFactory.class.getCanonicalName() + " bsFactory; " +
            "rule 'cloudy' when String(this == 'cloudy') g : Garden() then " +
            "    insertLogical(new PropertyReference(g, 'cloudy'), bsFactory.create(new double[]{1.0, 0.0})); " +
            "end " +
            "rule 'rain' when String(this == 'rain') g : Garden() then " +
            "    insertLogical(new PropertyReference(g, 'rain'), bsFactory.create(new double[]{1.0, 0.0})); " +
            "end ";

        // Two sessions: one uses globalUpdateForced, one uses globalUpdate (incremental)
        KieSession forcedSession      = buildSession(drl);
        KieSession incrementalSession = buildSession(drl);

        BayesModeFactory bsForced = new BayesModeFactoryImpl(forcedSession);
        BayesModeFactory bsInc    = new BayesModeFactoryImpl(incrementalSession);
        forcedSession.setGlobal("bsFactory", bsForced);
        incrementalSession.setGlobal("bsFactory", bsInc);

        Garden gForced = new Garden();
        Garden gInc    = new Garden();
        forcedSession.insert(gForced);
        incrementalSession.insert(gInc);
        forcedSession.fireAllRules();
        incrementalSession.fireAllRules();

        BayesRuntime forcedRuntime = forcedSession.getKieRuntime(BayesRuntime.class);
        BayesRuntime incRuntime    = incrementalSession.getKieRuntime(BayesRuntime.class);
        BayesInstance<Garden> forcedInst = forcedRuntime.createInstance(Garden.class);
        BayesInstance<Garden> incInst    = incRuntime.createInstance(Garden.class);

        forcedInst.globalUpdateForced();
        incInst.globalUpdate();
        assertMarginalMatchGarden(forcedInst, incInst);

        // Trigger 'cloudy' rule in both sessions
        forcedSession.insert("cloudy");      forcedSession.fireAllRules();
        incrementalSession.insert("cloudy"); incrementalSession.fireAllRules();
        forcedInst.globalUpdateForced();
        incInst.globalUpdate();
        assertMarginalMatchGarden(forcedInst, incInst);

        // Trigger 'rain' rule in both sessions
        forcedSession.insert("rain");      forcedSession.fireAllRules();
        incrementalSession.insert("rain"); incrementalSession.fireAllRules();
        forcedInst.globalUpdateForced();
        incInst.globalUpdate();
        assertMarginalMatchGarden(forcedInst, incInst);

        forcedSession.dispose();
        incrementalSession.dispose();
    }

    private static void assertMarginalMatchGarden(BayesInstance<Garden> expected,
                                                   BayesInstance<Garden> actual) {
        for (String var : new String[]{"cloudy", "sprinkler", "rain", "wetGrass"}) {
            double[] exp = expected.marginalize(var).getDistribution();
            double[] act = actual.marginalize(var).getDistribution();
            for (int j = 0; j < exp.length; j++) {
                assertThat(act[j]).as("%s[%d]", var, j).isCloseTo(exp[j], within(1e-8));
            }
        }
    }
}
```

- [ ] **Step 2: Run to confirm tests fail (incremental classes not yet built)**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalBayesIntegrationTest"
```

Expected: compilation errors — `BayesInstance(JunctionTree, boolean)` and other Phase 3 APIs not yet present. Re-run this step after each Phase is implemented.

- [ ] **Step 3: Run after Phase 1 is complete — verify base e2e tests pass**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalBayesIntegrationTest#e2e_ruleInsertsEvidence_posteriorUpdated"
```

Expected: PASS — Phase 1 does not break the basic session flow.

- [ ] **Step 4: Run all e2e tests after Phase 2 is complete**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress \
    -Dtest="IncrementalBayesIntegrationTest"
```

Expected: all PASS except the Phase 3 test (`e2e_incrementalMatchesForcedAfterEachRuleFire` with `BayesInstance(jt, true)` constructor) which needs Task 8.

- [ ] **Step 5: Run full suite after all phases complete**

```bash
mvn test -pl smallrye-bayesian -am --no-transfer-progress
```

Expected: `BUILD SUCCESS`, every test in every file passes.

- [ ] **Step 6: Commit**

```bash
git add smallrye-bayesian/src/test/java/org/drools/beliefs/bayes/integration/IncrementalBayesIntegrationTest.java
git commit -m "test(beliefs): add end-to-end KieSession integration tests for incremental Bayesian update"
```

---

## Implementation Notes

### `BayesLikelyhood.divideFrom()` — check before implementing

Before implementing `divideFrom()` in Task 8 Step 7, read `BayesLikelyhood.multiplyInto()` to understand the index-mapping pattern it uses. The division method must use the exact same index-mapping logic with `/=` instead of `*=`. If `multiplyInto()` uses a recursive helper, replicate that helper for division.

### `computeDirtySubtrees()` — called twice risk

In Task 8, the `globalUpdate()` logic must not call `computeDirtySubtrees()` twice (once for Phase 3 check, once for Phase 2 fallback). The `dirtySubtreesComputed` boolean flag guards this. Keep the guard consistent when modifying `globalUpdate()`.

### Phase 3 `sepPotSnapshots` population

In `retractionUpdate()`, the `sepPotSnapshots` arrays are populated from `calibratedSepPots` rather than from the current (pre-reset) separator states. This is correct because `calibratedSepPots` holds the last fully-calibrated separator values, which is exactly what Phase 2's incremental collect needs as the "clean subtree" baseline.

### Running benchmarks

The `mvn exec:java` approach requires `exec-maven-plugin` in the pom. If not present, the simplest alternative is to run from an IDE, or to build a fat jar. For the initial benchmark runs, running from the IDE is acceptable — the numbers recorded in `BenchmarkMain.java` are the important artefact.
