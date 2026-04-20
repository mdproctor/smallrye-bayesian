package io.smallrye.bayesian;

import io.smallrye.bayesian.benchmark.BayesNetworkFixtures;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static io.smallrye.bayesian.IncrementalUpdateTest.assertSameMarginals;

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

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.5, 0.5});
        forced.globalUpdateForced();
        inc.globalUpdate();

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
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(3), new double[]{0.7, 0.3});
        inc.setLikelyhood(BayesNetworkFixtures.varName(3), new double[]{0.7, 0.3});

        forced.globalUpdateForced();
        inc.globalUpdate();

        assertSameMarginals(forced, inc, 8);
    }

    @Test
    void phase2_hardEvidenceFollowedBySoftEvidence_correctMarginals() {
        JunctionTree jt = BayesNetworkFixtures.buildChain(8);
        BayesInstance<?> forced = new BayesInstance<>(jt);
        BayesInstance<?> inc = new BayesInstance<>(jt);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{1.0, 0.0});
        forced.globalUpdateForced();
        inc.globalUpdate();
        assertSameMarginals(forced, inc, 8);

        forced.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.6, 0.4});
        inc.setLikelyhood(BayesNetworkFixtures.varName(0), new double[]{0.6, 0.4});
        forced.globalUpdateForced();
        inc.globalUpdate();
        assertSameMarginals(forced, inc, 8);
    }
}
