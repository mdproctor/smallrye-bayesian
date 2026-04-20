package io.smallrye.bayesian.benchmark;

import io.smallrye.bayesian.BayesInstance;
import io.smallrye.bayesian.JunctionTree;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class BayesBenchmark {

    @Param({"8", "16", "32", "64", "128"})
    private int networkSize;

    private BayesInstance<?> baselineInstance;
    private boolean toggle;

    private BayesInstance<?> singleChangeInstance;
    private BayesInstance<?> batchChangeInstance;
    private boolean toggleSingle;
    private boolean toggleBatch;

    private BayesInstance<?> retractionInstance;
    private boolean toggleRetract;

    @Setup(Level.Trial)
    public void setup() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(networkSize);
        baselineInstance = new BayesInstance<>(jt);
        baselineInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        baselineInstance.globalUpdateForced();
        toggle = false;

        JunctionTree jt2 = BayesNetworkFixtures.buildChain(networkSize);
        singleChangeInstance = new BayesInstance<>(jt2);
        singleChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        singleChangeInstance.globalUpdate();
        toggleSingle = false;

        JunctionTree jt3 = BayesNetworkFixtures.buildChain(networkSize);
        batchChangeInstance = new BayesInstance<>(jt3);
        batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        batchChangeInstance.globalUpdate();
        toggleBatch = false;

        JunctionTree jt4 = BayesNetworkFixtures.buildChain(networkSize);
        retractionInstance = new BayesInstance<>(jt4, true);
        retractionInstance.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        retractionInstance.globalUpdate();
        toggleRetract = false;
    }

    /**
     * Permanent baseline — always full reset+propagate. Never modified as phases are added.
     * Alternates evidence each iteration so dirty is always set.
     */
    @Benchmark
    public void baselineFull(Blackhole bh) {
        toggle = !toggle;
        baselineInstance.setLikelyhood(BayesNetworkFixtures.varName(0),
                toggle ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
        baselineInstance.globalUpdateForced();
    }

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
        batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(0), ev);
        batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(networkSize / 3), ev);
        batchChangeInstance.setLikelyhood(BayesNetworkFixtures.varName(2 * networkSize / 3), ev);
        batchChangeInstance.globalUpdate();
    }

    @Benchmark
    public void fastRetraction(Blackhole bh) {
        toggleRetract = !toggleRetract;
        retractionInstance.setLikelyhood(BayesNetworkFixtures.varName(0),
                toggleRetract ? new double[]{0.8, 0.2} : new double[]{0.3, 0.7});
        retractionInstance.globalUpdate();
    }
}
