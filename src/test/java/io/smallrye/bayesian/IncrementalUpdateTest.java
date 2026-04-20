package io.smallrye.bayesian;

import io.smallrye.bayesian.benchmark.BayesNetworkFixtures;
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

    static AtomicInteger countingListener(BayesInstance<?> instance) {
        AtomicInteger count = new AtomicInteger();
        instance.setGlobalUpdateListener(new GlobalUpdateListener() {
            public void beforeGlobalUpdate(CliqueState clique) { count.incrementAndGet(); }
            public void afterGlobalUpdate(CliqueState clique) {}
        });
        return count;
    }

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

    // ── Happy path: globalUpdateForced ───────────────────────────────────────

    @Test
    void globalUpdateForced_withNoEvidence_producesPriorDistribution() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(4);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.globalUpdateForced();
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

        instance.globalUpdateForced();
        double[] second = instance.marginalize(BayesNetworkFixtures.varName(7)).getDistribution();

        for (int j = 0; j < first.length; j++) {
            assertThat(second[j]).isCloseTo(first[j], within(1e-10));
        }
    }

    @Test
    void globalUpdateForced_withHardEvidence_producesCorrectMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(4);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        instance.globalUpdateForced();
        double[] dist = instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution();
        assertThat(dist[0]).isCloseTo(1.0, within(1e-10));
        assertThat(dist[1]).isCloseTo(0.0, within(1e-10));
    }

    // ── Phase 1: Calibration guard ───────────────────────────────────────────

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

        instance.globalUpdate();
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
        instance.globalUpdate();
        assertThat(count.get()).isEqualTo(2);
    }

    @Test
    void phase1_multipleNoOpCallsAllSkip() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        AtomicInteger count = countingListener(instance);

        instance.globalUpdate();
        instance.globalUpdate();
        instance.globalUpdate();
        instance.globalUpdate();
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
        instance.globalUpdate();
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
        instance.unsetLikelyhood(var0);
        instance.globalUpdate();
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

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        forced.globalUpdateForced();
        inc.globalUpdate();
        assertSameMarginals(forced, inc, 16);

        forced.setLikelyhood(BayesNetworkFixtures.varName(5), new double[]{0.3, 0.7});
        inc.setLikelyhood(BayesNetworkFixtures.varName(5), new double[]{0.3, 0.7});
        forced.globalUpdateForced();
        inc.globalUpdate();
        assertSameMarginals(forced, inc, 16);

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
        BayesVariable var4Forced = forced.getVariables().get(BayesNetworkFixtures.varName(4));
        BayesVariable var4Inc    = inc.getVariables().get(BayesNetworkFixtures.varName(4));

        forced.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.9, 0.1});
        inc.setLikelyhood(BayesNetworkFixtures.varName(4), new double[]{0.9, 0.1});
        forced.globalUpdateForced();
        inc.globalUpdate();

        forced.unsetLikelyhood(var4Forced);
        inc.unsetLikelyhood(var4Inc);
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

        assertThat(savings[1]).isGreaterThanOrEqualTo(savings[0]);
        assertThat(savings[2]).isGreaterThanOrEqualTo(savings[1]);
    }

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

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.3, 0.7});
        forced.globalUpdateForced();
        retract.globalUpdate();

        assertSameMarginals(forced, retract, 16);
    }

    @Test
    void phase3_hardEvidenceFallsBackToPhase2_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced  = new BayesInstance<>(jt);
        BayesInstance<?> retract = new BayesInstance<>(jt, true);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        forced.globalUpdateForced();
        retract.globalUpdate();

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        forced.globalUpdateForced();
        retract.globalUpdate();

        assertSameMarginals(forced, retract, 8);
    }

    @Test
    void phase3_firstEvidenceInsertionFallsBackToPhase2_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced  = new BayesInstance<>(jt);
        BayesInstance<?> retract = new BayesInstance<>(jt, true);

        forced.globalUpdateForced();
        retract.globalUpdate();

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

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        forced.globalUpdateForced();
        retract.globalUpdate();
        assertSameMarginals(forced, retract, 16);

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
    void phase3_retractionThenFallbackThenRetractionAgain_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced  = new BayesInstance<>(jt);
        BayesInstance<?> retract = new BayesInstance<>(jt, true);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        forced.globalUpdateForced(); retract.globalUpdate();
        assertSameMarginals(forced, retract, 8);

        // soft → soft (retraction)
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.4, 0.6});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.4, 0.6});
        forced.globalUpdateForced(); retract.globalUpdate();
        assertSameMarginals(forced, retract, 8);

        // soft → hard (fallback)
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        forced.globalUpdateForced(); retract.globalUpdate();
        assertSameMarginals(forced, retract, 8);

        // hard → soft (retraction or fallback depending on calibrated state)
        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.7, 0.3});
        retract.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.7, 0.3});
        forced.globalUpdateForced(); retract.globalUpdate();
        assertSameMarginals(forced, retract, 8);
    }

    @Test
    void phase3_noEvidence_priorDistribution_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(4);
        BayesInstance<?> forced  = new BayesInstance<>(jt);
        BayesInstance<?> retract = new BayesInstance<>(jt, true);

        forced.globalUpdateForced();
        retract.globalUpdate();

        assertSameMarginals(forced, retract, 4);
    }
}
