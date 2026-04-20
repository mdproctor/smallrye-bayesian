/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package io.smallrye.bayesian;

import io.smallrye.bayesian.graph.Graph;
import io.smallrye.bayesian.graph.GraphNode;
import io.smallrye.bayesian.util.bitmask.BitMask;

import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BayesInstance<T> {
    private static final SecureRandom randomGenerator = new SecureRandom();

    private Graph<BayesVariable>       graph;
    private JunctionTree               tree;
    private Map<String, BayesVariable> variables;
    private Map<String, BayesVariable> fieldNames;
    private BayesLikelyhood[]          likelyhoods;
    private BayesLikelyhood[]          previousLikelyhoods;
    private BitMask                    dirty;
    private BitMask                    decided;
    private boolean                    calibrated;

    private CliqueState[]        cliqueStates;
    private SeparatorState[]     separatorStates;
    private BayesVariableState[] varStates;

    // Phase 2: dirty subtree tracking + separator snapshots
    private boolean[] subtreeHasDirty;
    private double[][] sepPotSnapshots;

    // Phase 3: calibrated snapshots for fast evidence retraction
    private boolean enableFastRetract;
    private double[][] calibratedCliquePots;
    private double[][] calibratedSepPots;

    private GlobalUpdateListener globalUpdateListener;
    private PassMessageListener  passMessageListener;

    private int[]          targetParameterMap;
    private Class<T>       targetClass;
    private Constructor<T> targetConstructor;

    public BayesInstance(JunctionTree tree, Class<T> targetClass) {
        this(tree);
        this.targetClass = targetClass;
        buildParameterMapping(targetClass);
        buildFieldMappings( targetClass );
    }

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

    public BayesInstance(JunctionTree tree) {
        this.graph = tree.getGraph();
        this.tree = tree;
        variables = new HashMap<>();
        fieldNames = new HashMap<>();
        likelyhoods = new BayesLikelyhood[graph.size()];
        previousLikelyhoods = new BayesLikelyhood[graph.size()];
        dirty = BitMask.getEmpty(graph.size());
        decided = BitMask.getEmpty(graph.size());
        this.enableFastRetract = false;

        cliqueStates = new CliqueState[tree.getJunctionTreeNodes().length];
        for (JunctionTreeClique clique : tree.getJunctionTreeNodes()) {
            cliqueStates[clique.getId()] = clique.createState();
        }

        separatorStates = new SeparatorState[tree.getJunctionTreeSeparators().length];
        for ( JunctionTreeSeparator sep : tree.getJunctionTreeSeparators() ) {
            separatorStates[sep.getId()] = sep.createState();
        }

        subtreeHasDirty = new boolean[tree.getJunctionTreeNodes().length];

        sepPotSnapshots = new double[tree.getJunctionTreeSeparators().length][];
        for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
            sepPotSnapshots[sep.getId()] = new double[separatorStates[sep.getId()].getPotentials().length];
        }

        varStates = new BayesVariableState[graph.size()];
        for (GraphNode<BayesVariable> node : graph) {
            BayesVariable var = node.getContent();
            variables.put(var.getName(), var);
            varStates[var.getId()] = var.createState();
        }
    }

    private void snapshotSeparators() {
        for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
            double[] current = separatorStates[sep.getId()].getPotentials();
            System.arraycopy(current, 0, sepPotSnapshots[sep.getId()], 0, current.length);
        }
    }

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

    private boolean isRetractable() {
        for (int varId = 0; varId < likelyhoods.length; varId++) {
            if (!dirty.isSet(varId)) continue;
            if (previousLikelyhoods[varId] == null) return false;
            // New evidence has hard evidence (zero) → will cause zeros post-update, not retractable
            BayesLikelyhood newL = likelyhoods[varId];
            if (newL != null) {
                for (double v : newL.getDistribution()) {
                    if (v == 0.0) return false;
                }
            }
            // Calibrated clique has zeros → division unsafe
            int cliqueId = varStates[varId].getVariable().getFamily();
            for (double v : calibratedCliquePots[cliqueId]) {
                if (v == 0.0) return false;
            }
        }
        // Also check separators on the dirty path for zeros (division safety)
        for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
            if (subtreeHasDirty[sep.getChild().getId()]) {
                for (double v : calibratedSepPots[sep.getId()]) {
                    if (v == 0.0) return false;
                }
            }
        }
        return true;
    }

    private void retractionUpdate() {
        for (int varId = 0; varId < likelyhoods.length; varId++) {
            if (!dirty.isSet(varId)) continue;

            int cliqueId = varStates[varId].getVariable().getFamily();
            double[] cliquePots = cliqueStates[cliqueId].getPotentials();

            // Restore calibrated clique potentials as starting point
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

        // Incremental collect using calibrated separator state as the no-op reference
        // (do NOT modify sepPotSnapshots — it must retain its post-collect values for Phase 2 fallback)
        collectEvidenceRetract(tree.getRoot());

        // After retraction collect, update sepPotSnapshots only for dirty-path separators
        // (those that passMessage actually updated). Clean-path seps keep their existing
        // post-collect snapshot for use in subsequent Phase 2 calls.
        snapshotDirtyPathSeparators();

        distributeEvidence(tree.getRoot());

        // Clean up subtreeHasDirty after retractionUpdate is complete
        Arrays.fill(subtreeHasDirty, false);
    }

    /**
     * Incremental collect for the Phase 3 retraction path.
     * Uses calibratedSepPots (post-distribute) as the no-op reference for clean separators,
     * leaving sepPotSnapshots untouched.
     */
    private void collectEvidenceRetract(JunctionTreeClique startClique) {
        collectChildEvidenceRetract(startClique, startClique);
    }

    private void collectChildEvidenceRetract(JunctionTreeClique clique, JunctionTreeClique startClique) {
        List<JunctionTreeSeparator> seps = clique.getChildren();
        for (JunctionTreeSeparator sep : seps) {
            collectChildEvidenceRetract(sep.getChild(), startClique);
        }

        if (clique.getParentSeparator() != null && clique != startClique) {
            JunctionTreeSeparator parentSep = clique.getParentSeparator();
            JunctionTreeClique parent = parentSep.getParent();

            if (subtreeHasDirty[clique.getId()]) {
                passMessage(clique, parentSep, parent); // dirty: recompute
            } else {
                injectCalibratedAndAbsorb(parentSep, parent); // clean: inject calibrated (no-op)
            }
        }
    }

    /**
     * Like injectSnapshotAndAbsorb, but uses calibratedSepPots instead of sepPotSnapshots.
     * Since calibratedSepPots equals the current separator state (both post-distribute),
     * the absorption ratio is 1.0 — effectively a no-op.
     */
    private void injectCalibratedAndAbsorb(JunctionTreeSeparator sep, JunctionTreeClique targetClique) {
        double[] sepPots = separatorStates[sep.getId()].getPotentials();
        double[] oldSepPots = Arrays.copyOf(sepPots, sepPots.length);
        System.arraycopy(calibratedSepPots[sep.getId()], 0, sepPots, 0, sepPots.length);
        BayesVariable[] sepVars = sep.getValues().toArray(new BayesVariable[sep.getValues().size()]);
        absorb(sepVars, cliqueStates[targetClique.getId()], separatorStates[sep.getId()], oldSepPots);
    }

    private void snapshotDirtyPathSeparators() {
        for (JunctionTreeSeparator sep : tree.getJunctionTreeSeparators()) {
            // A separator is on the dirty path if the child clique has a dirty subtree
            // (i.e., passMessage was called for it during retraction collect).
            if (sep.getChild() != null && subtreeHasDirty[sep.getChild().getId()]) {
                double[] current = separatorStates[sep.getId()].getPotentials();
                System.arraycopy(current, 0, sepPotSnapshots[sep.getId()], 0, current.length);
            }
        }
    }

    private void computeDirtySubtrees() {
        Arrays.fill(subtreeHasDirty, false);
        for (int varId = 0; varId < likelyhoods.length; varId++) {
            if (dirty.isSet(varId)) {
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

    public void reset() {
        for (JunctionTreeClique clique : tree.getJunctionTreeNodes()) {
            clique.resetState(cliqueStates[clique.getId()]);
        }

        for ( JunctionTreeSeparator sep : tree.getJunctionTreeSeparators() ) {
            sep.resetState(separatorStates[sep.getId()]);
        }

        for (GraphNode<BayesVariable> node : graph) {
            BayesVariable var = node.getContent();
            BayesVariableState varState =  varStates[var.getId()];
            varState.setDistribution( new double[ varState.getDistribution().length]);
        }
    }

    public void setTargetClass(Class<T> targetClass) {
        this.targetClass = targetClass;
        buildParameterMapping( targetClass );
        buildFieldMappings( targetClass );
    }

    public void buildFieldMappings(Class<T> target) {
        for ( Field field : target.getDeclaredFields() ) {
            Annotation[] anns = field.getDeclaredAnnotations();
            for ( Annotation ann : anns ) {
                if (ann.annotationType() == VarName.class) {
                    String varName = ((VarName)ann).value();
                    BayesVariable var = variables.get(varName);
                    fieldNames.put( field.getName(), var);
                }
            }
        }
    }

    public  void buildParameterMapping(Class<T> target) {
        Constructor[] cons = target.getConstructors();
        for ( Constructor con : cons ) {
            for ( Annotation ann : con.getDeclaredAnnotations() ) {
                if ( ann.annotationType() == BayesVariableConstructor.class ) {
                    Class[] paramTypes = con.getParameterTypes();

                    targetParameterMap = new int[paramTypes.length];
                    if ( paramTypes[0] != BayesInstance.class ) {
                        throw new RuntimeException( "First Argument must be " + BayesInstance.class.getSimpleName() );
                    }
                    Annotation[][] paramAnns = con.getParameterAnnotations();
                    for ( int j = 1; j < paramAnns.length; j++ ) {
                        if ( paramAnns[j][0].annotationType() == VarName.class ) {
                            String varName = ((VarName)paramAnns[j][0]).value();
                            BayesVariable var = variables.get(varName);
                            Object[] outcomes = new Object[ var.getOutcomes().length ];
                            if ( paramTypes[j].isAssignableFrom( Boolean.class) || paramTypes[j].isAssignableFrom( boolean.class) ) {
                                for ( int k = 0; k < var.getOutcomes().length; k++ ) {
                                    outcomes[k] = Boolean.valueOf( (String) var.getOutcomes()[k]);
                                }
                            }
                            varStates[var.getId()].setOutcomes( outcomes );
                            targetParameterMap[j] = var.getId();
                        }
                    }
                    targetConstructor = con;
                }
            }
        }
        if ( targetConstructor == null ) {
            throw new IllegalStateException( "Unable to find Constructor" );
        }
    }

    public GlobalUpdateListener getGlobalUpdateListener() {
        return globalUpdateListener;
    }

    public void setGlobalUpdateListener(GlobalUpdateListener globalUpdateListener) {
        this.globalUpdateListener = globalUpdateListener;
    }

    public PassMessageListener getPassMessageListener() {
        return passMessageListener;
    }

    public void setPassMessageListener(PassMessageListener passMessageListener) {
        this.passMessageListener = passMessageListener;
    }

    public Map<String, BayesVariable> getVariables() {
        return variables;
    }

    public Map<String, BayesVariable> getFieldNames() {
        return fieldNames;
    }

    public void setDecided(String varName, boolean bool) {

    }

    public void setDecided(BayesVariable var, boolean bool) {
        // note this is reversed, when the bit is on, the var is undecided. Default state is decided
        if ( !bool ) {
            decided = decided.set(var.getId());
        } else {
            decided = decided.reset(var.getId());
        }
    }

    public boolean isDecided() {
        return decided.isEmpty();
    }

    public boolean isDirty() {
        return !dirty.isEmpty();
    }

    public void setLikelyhood(String varName, double[] distribution) {
        BayesVariable var = variables.get( varName );
        if (  var == null ) {
            throw new IllegalArgumentException("Variable name does not exist: " + varName);
        }
        setLikelyhood( var, distribution );
    }

    public void unsetLikelyhood(BayesVariable var) {
        int id = var.getId();
        previousLikelyhoods[id] = this.likelyhoods[id];
        this.likelyhoods[id] = null;
        dirty = dirty.set(id);
    }

    public void setLikelyhood(BayesVariable var, double[] distribution) {
        GraphNode node = graph.getNode( var.getId() );
        JunctionTreeClique clique = tree.getJunctionTreeNodes( )[var.getFamily()];

        setLikelyhood( new BayesLikelyhood(graph, clique, node, distribution ) );
    }

    public void setLikelyhood(BayesLikelyhood likelyhood) {
        int id = likelyhood.getVariable().getId();
        BayesLikelyhood old = this.likelyhoods[id];
        if ( old == null || !old.equals( likelyhood ) ) {
            previousLikelyhoods[id] = old;
            this.likelyhoods[likelyhood.getVariable().getId()] = likelyhood;
            dirty = dirty.set(id);
        }
    }

    public void globalUpdate() {
        if ( !isDecided() ) {
            throw new IllegalStateException("Cannot perform global update while one or more variables are undecided");
        }
        if ( calibrated && !isDirty() ) {                    // Phase 1
            return;
        }

        boolean dirtySubtreesComputed = false;
        if ( calibrated && enableFastRetract ) {
            computeDirtySubtrees();
            dirtySubtreesComputed = true;
            if ( isRetractable() ) {                         // Phase 3
                if (globalUpdateListener != null) {
                    globalUpdateListener.beforeGlobalUpdate(cliqueStates[tree.getRoot().getId()]);
                }
                retractionUpdate();
                saveCalibrated();
                dirty = BitMask.getEmpty(likelyhoods.length);
                calibrated = true;
                if (globalUpdateListener != null) {
                    globalUpdateListener.afterGlobalUpdate(cliqueStates[tree.getRoot().getId()]);
                }
                return;
            }
        }

        if ( calibrated ) {                                  // Phase 2 setup
            if ( !dirtySubtreesComputed ) computeDirtySubtrees();
        }

        if ( isDirty() ) {
            reset();
        }
        applyEvidence();

        if (globalUpdateListener != null) {
            globalUpdateListener.beforeGlobalUpdate(cliqueStates[tree.getRoot().getId()]);
        }

        if ( calibrated ) {
            collectEvidenceIncremental(tree.getRoot());
        } else {
            collectEvidence(tree.getRoot());
        }

        snapshotSeparators();
        distributeEvidence(tree.getRoot());

        if ( enableFastRetract ) {
            saveCalibrated();
        }

        dirty = BitMask.getEmpty(likelyhoods.length);
        calibrated = true;
        Arrays.fill(subtreeHasDirty, false);

        if (globalUpdateListener != null) {
            globalUpdateListener.afterGlobalUpdate(cliqueStates[tree.getRoot().getId()]);
        }
    }

    /**
     * For a clean (non-dirty) subtree: restore the calibrated separator snapshot
     * and absorb it into the target clique. After reset(), separator potentials are
     * all 1.0, so oldSepPots = {1.0,...} and absorb gives:
     *   target[i] = targetInit[i] * snapshot[j] / 1.0
     * which correctly injects the previously calibrated message without recomputing it.
     */
    private void injectSnapshotAndAbsorb(JunctionTreeSeparator sep, JunctionTreeClique targetClique) {
        double[] sepPots = separatorStates[sep.getId()].getPotentials();

        // oldSepPots = current post-reset state = all 1.0
        double[] oldSepPots = Arrays.copyOf(sepPots, sepPots.length);

        // Restore the calibrated snapshot into the separator
        System.arraycopy(sepPotSnapshots[sep.getId()], 0, sepPots, 0, sepPots.length);

        // Absorb snapshot into target (no project — source clique not involved)
        BayesVariable[] sepVars = sep.getValues().toArray(new BayesVariable[sep.getValues().size()]);
        absorb(sepVars, cliqueStates[targetClique.getId()], separatorStates[sep.getId()], oldSepPots);
    }

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

    private void collectEvidenceIncremental(JunctionTreeClique clique) {
        if (clique.getParentSeparator() != null) {
            // Non-root start: fall back to full path (unreachable via globalUpdate())
            collectParentEvidence(clique.getParentSeparator().getParent(),
                    clique.getParentSeparator(), clique, clique);
        }
        collectChildEvidenceIncremental(clique, clique);
    }

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
        snapshotSeparators();
        saveCalibrated();
        dirty = BitMask.getEmpty(likelyhoods.length);
        calibrated = true;
    }

    public void applyEvidence() {
        for ( int i = 0; i < likelyhoods.length; i++ ) {
            BayesLikelyhood l = likelyhoods[i];
            if ( l != null ) {
                int family = likelyhoods[i].getVariable().getFamily();
                JunctionTreeClique node = tree.getJunctionTreeNodes()[family];
                likelyhoods[i].multiplyInto(cliqueStates[family].getPotentials());
                BayesAbsorption.normalize(cliqueStates[family].getPotentials());
            }
        }

    }

    public void globalUpdate(JunctionTreeClique clique) {
        if ( globalUpdateListener != null ) {
            globalUpdateListener.beforeGlobalUpdate(cliqueStates[clique.getId()]);
        }
        collectEvidence( clique );
        distributeEvidence( clique );
        if ( globalUpdateListener != null ) {
            globalUpdateListener.afterGlobalUpdate(cliqueStates[clique.getId()]);
        }
    }

    public void recurseGlobalUpdate(JunctionTreeClique clique) {
        globalUpdate(clique);

        List<JunctionTreeSeparator> seps = clique.getChildren();
        for ( JunctionTreeSeparator sep : seps ) {
            recurseGlobalUpdate(sep.getChild());
        }
    }

    public void collectEvidence(JunctionTreeClique clique) {
        if ( clique.getParentSeparator() != null ) {
            collectParentEvidence(clique.getParentSeparator().getParent(), clique.getParentSeparator(), clique, clique);
        }

        collectChildEvidence(clique, clique);
    }

    public void collectParentEvidence(JunctionTreeClique clique, JunctionTreeSeparator sep, JunctionTreeClique child, JunctionTreeClique startClique) {
        if ( clique.getParentSeparator() != null ) {
            collectParentEvidence(clique.getParentSeparator().getParent(), clique.getParentSeparator(),
                                  clique,
                                  startClique);
        }

        List<JunctionTreeSeparator> seps = clique.getChildren();
        for ( JunctionTreeSeparator childSep : seps ) {
            if ( childSep.getChild() == child )  {
                // ensure that when called from collectParentEvidence it does not re-enter the same node
                continue;
            }
            collectChildEvidence(childSep.getChild(), startClique);
        }

        passMessage(clique, child.getParentSeparator(), child );
    }


    public void collectChildEvidence(JunctionTreeClique clique, JunctionTreeClique startClique) {
        List<JunctionTreeSeparator> seps = clique.getChildren();
        for ( JunctionTreeSeparator sep : seps ) {
            collectChildEvidence(sep.getChild(), startClique);
        }

        if ( clique.getParentSeparator() != null && clique != startClique ) {
            // root has no parent, so we need to check.
            // Do not propogate the start node into another node
            passMessage(clique, clique.getParentSeparator(), clique.getParentSeparator().getParent() );
        }
    }

    public void distributeEvidence(JunctionTreeClique clique) {
        if ( clique.getParentSeparator() != null ) {
            distributeParentEvidence(clique.getParentSeparator().getParent(), clique.getParentSeparator(), clique, clique);
        }

        distributeChildEvidence(clique, clique);
    }

    public void distributeParentEvidence(JunctionTreeClique clique, JunctionTreeSeparator sep, JunctionTreeClique child, JunctionTreeClique startClique) {
        passMessage(child, child.getParentSeparator(), clique);

        if ( clique.getParentSeparator() != null ) {
            distributeParentEvidence(clique.getParentSeparator().getParent(), clique.getParentSeparator(),
                                     clique,
                                     startClique);
        }

        List<JunctionTreeSeparator> seps = clique.getChildren();
        for ( JunctionTreeSeparator childSep : seps ) {
            if ( childSep.getChild() == child )  {
                // ensure that when called from distributeParentEvidence it does not re-enter the same node
                continue;
            }
            distributeChildEvidence(childSep.getChild(), startClique);
        }
    }


    public void distributeChildEvidence(JunctionTreeClique clique, JunctionTreeClique startClique) {
        if ( clique.getParentSeparator() != null && clique != startClique ) {
            // root has no parent, so we need to check.
            // Do not propogate the start node into another node
            passMessage( clique.getParentSeparator().getParent(), clique.getParentSeparator(), clique );
        }

        List<JunctionTreeSeparator> seps = clique.getChildren();
        for ( JunctionTreeSeparator sep : seps ) {
            distributeChildEvidence(sep.getChild(), startClique);
        }
    }


    /**
     * Passes a message from node1 to node2.
     * node1 projects its trgPotentials into the separator.
     * node2 then absorbs those trgPotentials from the separator.
     * @param sourceClique
     * @param sep
     * @param targetClique
     */
    public void passMessage( JunctionTreeClique sourceClique, JunctionTreeSeparator sep, JunctionTreeClique targetClique) {
        double[] sepPots = separatorStates[sep.getId()].getPotentials();
        double[] oldSepPots = Arrays.copyOf(sepPots, sepPots.length);

        BayesVariable[] sepVars = sep.getValues().toArray(new BayesVariable[sep.getValues().size()]);

        if ( passMessageListener != null ) {
            passMessageListener.beforeProjectAndAbsorb(sourceClique, sep, targetClique, oldSepPots);
        }

        project(sepVars, cliqueStates[sourceClique.getId()], separatorStates[sep.getId()]);
        if ( passMessageListener != null ) {
            passMessageListener.afterProject(sourceClique, sep, targetClique, oldSepPots);
        }

        absorb(sepVars, cliqueStates[targetClique.getId()], separatorStates[sep.getId()], oldSepPots);
        if ( passMessageListener != null ) {
            passMessageListener.afterAbsorb(sourceClique, sep, targetClique, oldSepPots);
        }
    }

    //private static void project(BayesVariable[] sepVars, JunctionTreeNode node, JunctionTreeSeparator sep) {
    private static void project(BayesVariable[] sepVars, CliqueState clique, SeparatorState separator) {
        //JunctionTreeNode node, JunctionTreeSeparator sep
        BayesVariable[] vars = clique.getJunctionTreeClique().getValues().toArray(new BayesVariable[clique.getJunctionTreeClique().getValues().size()]);
        int[] sepVarPos = PotentialMultiplier.createSubsetVarPos(vars, sepVars);

        int sepVarNumberOfStates = PotentialMultiplier.createNumberOfStates(sepVars);
        int[] sepVarMultipliers = PotentialMultiplier.createIndexMultipliers(sepVars, sepVarNumberOfStates);

        BayesProjection p = new BayesProjection(vars, clique.getPotentials(), sepVarPos, sepVarMultipliers, separator.getPotentials());
        p.project();
    }

    //private static void absorb(BayesVariable[] sepVars, JunctionTreeNode node, JunctionTreeSeparator sep, double[] oldSepPots ) {
    private static void absorb(BayesVariable[] sepVars, CliqueState clique, SeparatorState separator, double[] oldSepPots ) {
        //BayesVariable[] vars = node.getValues().toArray( new BayesVariable[node.getValues().size()] );
        BayesVariable[] vars = clique.getJunctionTreeClique().getValues().toArray(new BayesVariable[clique.getJunctionTreeClique().getValues().size()]);

        int[] sepVarPos = PotentialMultiplier.createSubsetVarPos(vars, sepVars);

        int sepVarNumberOfStates = PotentialMultiplier.createNumberOfStates(sepVars);
        int[] sepVarMultipliers = PotentialMultiplier.createIndexMultipliers(sepVars, sepVarNumberOfStates);

        BayesAbsorption p = new BayesAbsorption(sepVarPos, oldSepPots, separator.getPotentials(), sepVarMultipliers, vars, clique.getPotentials());
        p.absorb();
    }

    public BayesVariableState marginalize(String name) {
        BayesVariable var = this.variables.get(name);
        if ( var == null ) {
            throw new IllegalArgumentException("Variable name does not exist '" + name + "'" );
        }
        BayesVariableState varState = varStates[var.getId()];
        marginalize( varState );
        return varState;
    }

    public T marginalize() {
        Object[] args = new Object[targetParameterMap.length];
        args[0] = this;
        for ( int i = 1; i < targetParameterMap.length; i++) {
            int id = targetParameterMap[i];
            BayesVariableState varState = varStates[id];
            marginalize(varState);
            int highestIndex = 0;
            double highestValue = 0;
            int maximalCounts = 1;
            for (int j = 0, length = varState.getDistribution().length;j < length; j++ ){
                if ( varState.getDistribution()[j] > highestValue ) {
                    highestValue = varState.getDistribution()[j];
                    highestIndex = j;
                    maximalCounts = 1;
                }  else  if ( j != 0 && varState.getDistribution()[j] == highestValue ) {
                    maximalCounts++;
                }
            }
            if ( maximalCounts > 1 ) {
                // have maximal conflict, so choose random one
                int picked = randomGenerator.nextInt( maximalCounts );
                int count = 0;
                for (int j = 0, length = varState.getDistribution().length;j < length; j++ ){
                    if ( varState.getDistribution()[j] == highestValue ) {
                        highestIndex = j;
                        if ( ++count > picked) {
                            break;
                        }
                    }
                }
            }
            args[i] = varState.getOutcomes()[highestIndex];
        }
        try {
            return targetConstructor.newInstance( args );
        } catch (Exception e) {
           throw new RuntimeException( "Unable to instantiate " + targetClass.getSimpleName() + " " + Arrays.asList( args ), e );
        }
    }

//    public T createBayesFact() {
//        Object[] args = new Object[targetParameterMap.length];
//        args[0] = this;
//        try {
//            return targetConstructor.newInstance( args );
//        } catch (Exception e) {
//            throw new RuntimeException( "Unable to instantiate " + targetClass.getSimpleName() + " " + Arrays.asList( args ), e );
//        }
//    }

    public void marginalize(BayesVariableState varState) {
        CliqueState cliqueState = cliqueStates[varState.getVariable().getFamily()];
        JunctionTreeClique jtNode = cliqueState.getJunctionTreeClique();
        new Marginalizer(jtNode.getValues().toArray( new BayesVariable[jtNode.getValues().size()]), cliqueState.getPotentials(), varState.getVariable(), varState.getDistribution() );
//        System.out.print( varState.getVariable().getName() + " " );
//        for ( double d : varState.getDistribution() ) {
//            System.out.print(d);
//            System.out.print(" ");
//        }
//        System.out.println(" ");
    }

    public SeparatorState[] getSeparatorStates() {
        return separatorStates;
    }

    public CliqueState[] getCliqueStates() {
        return cliqueStates;
    }

    public BayesVariableState[] getVarStates() {
        return varStates;
    }
}
