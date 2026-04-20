package io.smallrye.bayesian.benchmark;

import io.smallrye.bayesian.BayesInstance;
import io.smallrye.bayesian.JunctionTree;

/**
 * Hand-rolled benchmark to work around JMH 1.21 / Java 17 annotation-processor
 * incompatibility. Results are comparable to JMH AverageTime mode.
 *
 * Format: Benchmark  networkSize  Score  Units
 */
public class SimpleBenchmark {

    private static final int WARMUP_ITERS   = 20_000;
    private static final int MEASURE_ITERS  = 100_000;

    public static void main(String[] args) {
        System.out.printf("%-32s %6s  %10s  %s%n", "Benchmark", "n", "Score", "Units");
        System.out.println("-".repeat(65));

        for (int n : new int[]{8, 16, 32, 64, 128}) {
            run("baselineFull",         n, () -> baselineFull(n));
            run("noChangeUpdate",       n, () -> noChangeUpdate(n));
            run("singleEvidenceChange", n, () -> singleEvidenceChange(n));
            run("batchEvidenceChange",  n, () -> batchEvidenceChange(n));
            run("fastRetraction",       n, () -> fastRetraction(n));
            System.out.println();
        }
    }

    // ── Benchmark bodies ────────────────────────────────────────────────────

    private static double baselineFull(int n) {
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        instance.globalUpdateForced();
        boolean toggle = false;
        double sink = 0;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            toggle = !toggle;
            instance.setLikelyhood(BayesNetworkFixtures.varName(0),
                    toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
            instance.globalUpdateForced();
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) {
            toggle = !toggle;
            instance.setLikelyhood(BayesNetworkFixtures.varName(0),
                    toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
            instance.globalUpdateForced();
            sink += instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution()[0];
        }
        return (double)(System.nanoTime() - start) / MEASURE_ITERS / 1_000.0 + (sink * 1e-300);
    }

    private static double noChangeUpdate(int n) {
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.8, 0.2});
        instance.globalUpdate();
        double sink = 0;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            instance.globalUpdate();
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) {
            instance.globalUpdate();
            sink += instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution()[0];
        }
        return (double)(System.nanoTime() - start) / MEASURE_ITERS / 1_000.0 + (sink * 1e-300);
    }

    private static double singleEvidenceChange(int n) {
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        instance.globalUpdate();
        boolean toggle = false;
        double sink = 0;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            toggle = !toggle;
            instance.setLikelyhood(BayesNetworkFixtures.varName(0),
                    toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
            instance.globalUpdate();
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) {
            toggle = !toggle;
            instance.setLikelyhood(BayesNetworkFixtures.varName(0),
                    toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
            instance.globalUpdate();
            sink += instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution()[0];
        }
        return (double)(System.nanoTime() - start) / MEASURE_ITERS / 1_000.0 + (sink * 1e-300);
    }

    private static double batchEvidenceChange(int n) {
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> instance = new BayesInstance<>(jt);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        instance.globalUpdate();
        boolean toggle = false;
        double sink = 0;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            toggle = !toggle;
            double[] ev = toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7};
            instance.setLikelyhood(BayesNetworkFixtures.varName(0), ev);
            instance.setLikelyhood(BayesNetworkFixtures.varName(n / 3), ev);
            instance.setLikelyhood(BayesNetworkFixtures.varName(2 * n / 3), ev);
            instance.globalUpdate();
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) {
            toggle = !toggle;
            double[] ev = toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7};
            instance.setLikelyhood(BayesNetworkFixtures.varName(0), ev);
            instance.setLikelyhood(BayesNetworkFixtures.varName(n / 3), ev);
            instance.setLikelyhood(BayesNetworkFixtures.varName(2 * n / 3), ev);
            instance.globalUpdate();
            sink += instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution()[0];
        }
        return (double)(System.nanoTime() - start) / MEASURE_ITERS / 1_000.0 + (sink * 1e-300);
    }

    private static double fastRetraction(int n) {
        JunctionTree jt = BayesNetworkFixtures.buildChain(n);
        BayesInstance<?> instance = new BayesInstance<>(jt, true);
        instance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        instance.globalUpdate();
        boolean toggle = false;
        double sink = 0;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            toggle = !toggle;
            instance.setLikelyhood(BayesNetworkFixtures.varName(0),
                    toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
            instance.globalUpdate();
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) {
            toggle = !toggle;
            instance.setLikelyhood(BayesNetworkFixtures.varName(0),
                    toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
            instance.globalUpdate();
            sink += instance.marginalize(BayesNetworkFixtures.varName(0)).getDistribution()[0];
        }
        return (double)(System.nanoTime() - start) / MEASURE_ITERS / 1_000.0 + (sink * 1e-300);
    }

    // ── Runner ───────────────────────────────────────────────────────────────

    @FunctionalInterface
    interface BenchmarkFn { double run(); }

    private static void run(String name, int n, BenchmarkFn fn) {
        // Three independent runs; report the median to reduce noise
        double[] scores = new double[3];
        for (int r = 0; r < 3; r++) {
            System.gc();
            scores[r] = fn.run();
        }
        java.util.Arrays.sort(scores);
        System.out.printf("%-32s %6d  %10.3f  µs/op%n", name, n, scores[1]);
    }
}
