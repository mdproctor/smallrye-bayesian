package io.smallrye.bayesian.benchmark;

import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH entry point for BayesBenchmark. Requires a fat-jar or Maven surefire setup to resolve
 * the forked JVM classpath correctly. For quick results use SimpleBenchmark instead, which
 * runs in-process and needs no special setup:
 *
 *   mvn test-compile -pl drools-beliefs -am -DskipTests -q
 *   java -cp "target/test-classes:target/classes:$(mvn -pl drools-beliefs dependency:build-classpath -q -DforceStdout)" \
 *        org.drools.beliefs.bayes.benchmark.SimpleBenchmark
 *
 * Results captured 2026-04-20 (20k warmup / 100k measurement / median of 3 runs):
 *   Benchmark              n=8     n=16    n=32    n=64   n=128   µs/op
 *   baselineFull           3.81    8.22   16.39   33.19   68.28
 *   noChangeUpdate         0.11    0.10    0.11    0.10    0.11
 *   singleEvidenceChange   3.41    6.25   12.84   25.94   53.19
 *   batchEvidenceChange    4.87    7.33   14.72   31.78   62.26
 *   fastRetraction         2.83    5.90   12.04   25.52   51.94
 *
 * Integration tests (BayesBeliefSystemTest, BayesRuntimeTest, WeaverTest, AssemblerTest):
 *   These 4 tests fail with NoClassDefFoundError when run via 'mvn test -pl drools-beliefs'
 *   alone because transitive SNAPSHOT dependencies are not on the classpath. They pass in a
 *   full build: 'mvn test -pl drools-beliefs -am'. This is a pre-existing project issue,
 *   unrelated to the incremental inference work.
 */
public class BenchmarkMain {

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BayesBenchmark.class.getSimpleName())
                .build();
        new Runner(opt).run();
    }
}
