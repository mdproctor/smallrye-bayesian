package io.smallrye.bayesian.benchmark;

import io.smallrye.bayesian.BayesNetwork;
import io.smallrye.bayesian.BayesVariable;
import io.smallrye.bayesian.JunctionTree;
import io.smallrye.bayesian.JunctionTreeBuilder;
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
