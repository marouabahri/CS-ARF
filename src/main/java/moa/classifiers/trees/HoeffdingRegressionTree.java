/*
 *    HoeffdingRegressionTree.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.trees;


import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import moa.classifiers.Regressor;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.StringUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/*
 * Implementation of HoeffdingRegressionTree, regression and model trees for data streams.
 * (must be used with HoeffdingNominalAttributeClassObserver and HoeffdingNumericAttributeClassObserver)
 */

public class HoeffdingRegressionTree extends HoeffdingTree  implements Regressor {


    public FlagOption meanPredictionNodeOption = new FlagOption(
            "regressionTree", 'k', "Build a regression tree instead of a model tree.");


    public FloatOption learningRatioPerceptronOption = new FloatOption(
            "learningRatioPerceptron", 'w', "Learning ratio to used for training the Perceptrons in the leaves.",
            0.02, 0, 1.00);

    public FloatOption learningRateDecayFactorOption = new FloatOption(
            "learningRatioDecayFactorPerceptron", 'x', "Learning rate decay factor (not used when learning rate is constant).",
            0.001, 0, 1.00);

    public FlagOption learningRatioConstOption = new FlagOption(
            "learningRatioConstPerceptron", 'v', "Keep learning rate constant instead of decaying.");


    private static final long serialVersionUID = 1L;
    //variables useful for the normalization of the perceptrons

    public double examplesSeen = 0.0;

    protected double sumOfValues = 0.0;

    protected double sumOfSquares = 0.0;

    protected DoubleVector sumOfAttrValues = new DoubleVector();

    protected DoubleVector sumOfAttrSquares = new DoubleVector();

    public static boolean regressionTree ;

    @Override
    public String getPurposeString() {
        return "Hoeffding Regression Tree .";
    }

    public static class InactiveLearningNodeForRegression extends InactiveLearningNode {

        LearningNodePerceptron learningModel;

        public InactiveLearningNodeForRegression(double[] initialClassObservations, LearningNodePerceptron p) {
            super(initialClassObservations);
            this.learningModel = p ;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {

            //**
            // The observed class distribution contains the number of instances seen by the node in the first slot ,
            // the sum of values in the second and the sum of squared values in the third
            // these statistics are useful to calculate the mean and to calculate the variance reduction
            //**

            this.observedClassDistribution.addToValue(0,
                    inst.weight());
            this.observedClassDistribution.addToValue(1,
                    inst.weight() * inst.classValue());
            this.observedClassDistribution.addToValue(2,
                    inst.weight() * inst.classValue() * inst.classValue());

            if (regressionTree==false) {
                learningModel.updatePerceptron(inst);
            }
        }
    }

    public static class ActiveLearningNodeForRegression extends ActiveLearningNode {

        LearningNodePerceptron learningModel;

        public ActiveLearningNodeForRegression(double[] initialClassObservations,LearningNodePerceptron p) {
            super(initialClassObservations);
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.isInitialized = false;
            this.learningModel = p ;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            if (this.isInitialized == false) {
                this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
                this.isInitialized = true;
            }
            //**
            // The observed class distribution contains the number of instances seen by the node in the first slot ,
            // the sum of values in the second and the sum of squared values in the third
            // these statistics are useful to calculate the mean and to calculate the variance reduction
            //**

            this.observedClassDistribution.addToValue(0,
                    inst.weight());
            this.observedClassDistribution.addToValue(1,
                    inst.weight() * inst.classValue());
            this.observedClassDistribution.addToValue(2,
                    inst.weight() * inst.classValue() * inst.classValue());

            if (regressionTree==false)  {
                learningModel.updatePerceptron(inst);
            }


            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    //we use HoeffdingNominalAttributeClassObserver for nominal attributes and HoeffdingNUmericAttributeClassObserver for numeric attributes
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();

                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeTarget(inst.value(instAttIndex),  inst.classValue());
            }
        }

        @Override
        public double getWeightSeen() {
            return this.observedClassDistribution.getValue(0);
        }
    }

    //Implementation of the mean learning node
    public static class MeanLearningNode extends ActiveLearningNodeForRegression {

        public MeanLearningNode(double[] initialClassObservations,LearningNodePerceptron p) {
            super(initialClassObservations,p);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {

            double numberOfExamplesSeen = 0;

            double sumOfValues = 0;

            double prediction = 0;

            double V[] = super.getClassVotes(inst, ht);
            sumOfValues = V[1];
            numberOfExamplesSeen = V[0];
            prediction = sumOfValues / numberOfExamplesSeen;
            return new double[]{prediction};
        }

    }

    //Implementation of the perceptron learning node
    public static class PerceptronLearningNode extends ActiveLearningNodeForRegression {

        public PerceptronLearningNode(double[] initialClassObservations, LearningNodePerceptron p) {
            super(initialClassObservations, p);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            return new double[] {learningModel.prediction(inst)};
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst,
                    null, -1);
            Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            return leafNode.getClassVotes(inst, this);

        } else {
            return new double[]{0};
        }
    }

    protected LearningNode newLearningNode() {
        LearningNodePerceptron p = new LearningNodePerceptron();
        return newLearningNode(new double[0],p);
    }

    protected LearningNode newLearningNode(double[] initialClassObservations,LearningNodePerceptron p) {
        if (meanPredictionNodeOption.isSet())
        {
            regressionTree=true ;
            return new MeanLearningNode(initialClassObservations, p);
        }
        regressionTree=false ;
        return new PerceptronLearningNode(initialClassObservations, p);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        //Updating the tree statistics
        examplesSeen += inst.weight();
        sumOfValues += inst.weight() * inst.classValue();
        sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            int aIndex = modelAttIndexToInstanceAttIndex(i, inst);
            sumOfAttrValues.addToValue(i, inst.weight() * inst.value(aIndex));
            sumOfAttrSquares.addToValue(i, inst.weight() * inst.value(aIndex) * inst.value(aIndex));
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed
                    && (learningNode instanceof ActiveLearningNodeForRegression)) {
                ActiveLearningNodeForRegression activeLearningNode = (ActiveLearningNodeForRegression) learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (weightSeen
                        - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
                    attemptToSplit(activeLearningNode, foundNode.parent,
                            foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }
    }

    protected void attemptToSplit(ActiveLearningNodeForRegression node, SplitNode parent,
                                  int parentIndex) {
        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                if ((  secondBestSuggestion.merit/bestSuggestion.merit < 1 - hoeffdingBound)
                        || (hoeffdingBound < this.tieThresholdOption.getValue())) {
                    shouldSplit = true;

                }
                // }
                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2 - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }
            if (shouldSplit) {
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest == null) {
                    // preprune - null wins
                    deactivateLearningNode(node, parent, parentIndex);

                } else {
                    SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(),splitDecision.numSplits() );
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i),new LearningNodePerceptron((LearningNodePerceptron) node.learningModel));
                        newSplit.setChild(i, newChild);
                    }
                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }
                // manage memory
                enforceTrackerLimit();
            }
        }
    }


    public class LearningNodePerceptron implements Serializable {

        private static final long serialVersionUID = 1L;

        // The Perception weights
        protected DoubleVector weightAttribute = new DoubleVector();

        // The number of instances contributing to this model
        protected double instancesSeen = 0;

        // If the model should be reset or not
        protected boolean reset;

        public String getPurposeString() {
            return "A perceptron regressor as specified by Ikonomovska et al. used for FIMTDD";
        }

        public LearningNodePerceptron(LearningNodePerceptron original) {
            this.instancesSeen = original.instancesSeen;
            weightAttribute = (DoubleVector) original.weightAttribute.copy();
            reset = false;
        }

        public LearningNodePerceptron() {
            reset = true;
        }

        public DoubleVector getWeights() {
            return weightAttribute;
        }

        /**
         * Update the model using the provided instance
         */
        public void updatePerceptron(Instance inst) {

            // Initialize perceptron if necessary
            if (reset == true) {
                reset = false;
                weightAttribute = new DoubleVector();
                instancesSeen = 0;
                for (int j = 0; j < inst.numAttributes(); j++) { // The last index corresponds to the constant b
                    weightAttribute.setValue(j, 2 * classifierRandom.nextDouble() - 1);
                }
            }

            // Update attribute statistics
            instancesSeen += inst.weight();
            // Update weights
            double learningRatio = 0.0;
            if (learningRatioConstOption.isSet()) { learningRatio = learningRatioPerceptronOption.getValue();
            } else {
                learningRatio = learningRatioPerceptronOption.getValue() / (1 + instancesSeen * learningRateDecayFactorOption.getValue());
            }

            // Loop for compatibility with bagging methods
            for (int i = 0; i < (int) inst.weight(); i++) {
                updateWeights(inst, learningRatio);
            }
        }

        public void updateWeights(Instance inst, double learningRatio) {
            // Compute the normalized instance and the delta
            DoubleVector normalizedInstance = normalizedInstance(inst);
            double normalizedPrediction = prediction(normalizedInstance);
            double normalizedValue = normalizeTargetValue(inst.classValue());
            double delta = normalizedValue - normalizedPrediction;
            normalizedInstance.scaleValues(delta * learningRatio);
            weightAttribute.addValues(normalizedInstance);

        }

        public DoubleVector normalizedInstance(Instance inst) {
            // Normalize Instance
            DoubleVector normalizedInstance = new DoubleVector();

            for (int j = 0; j < inst.numAttributes() - 1; j++) {
                int l ;
                 DoubleVector  v = new DoubleVector() ;
                int index =0;

                int instAttIndex = modelAttIndexToInstanceAttIndex(j, inst);
                double mean = sumOfAttrValues.getValue(j) / examplesSeen;
                double sd = computeSD(sumOfAttrSquares.getValue(j), sumOfAttrValues.getValue(j), examplesSeen);
                if (inst.attribute(instAttIndex).isNumeric() && examplesSeen > 1 && sd > 0)
                    normalizedInstance.setValue(j, (inst.value(instAttIndex) - mean) / (3 * sd));
                else
                    normalizedInstance.setValue(j, 0);
            }
            if (examplesSeen > 1)
                normalizedInstance.setValue(inst.numAttributes() - 1, 1.0); // Value to be multiplied with the constant factor
            else
                normalizedInstance.setValue(inst.numAttributes() - 1, 0.0);
            return normalizedInstance;
        }

        /**
         * Output the prediction made by this perceptron on the given instance
         */
        public double prediction(DoubleVector instanceValues) {
            return scalarProduct(weightAttribute, instanceValues);
        }

        protected double prediction(Instance inst) {
            DoubleVector normalizedInstance = normalizedInstance(inst);
            double normalizedPrediction = prediction(normalizedInstance);
            return denormalizePrediction(normalizedPrediction);
        }

        private double denormalizePrediction(double normalizedPrediction) {
            double mean = sumOfValues / examplesSeen;
            double sd = computeSD(sumOfSquares, sumOfValues, examplesSeen);
            if (examplesSeen > 1)
                return normalizedPrediction * sd * 3 + mean;
            else
                return 0.0;
        }

        public void getModelDescription(StringBuilder out, int indent) {
            StringUtils.appendIndented(out, indent, getClassNameString() + " =");
            if (getModelContext() != null) {
                for (int j = 0; j < getModelContext().numAttributes() - 1; j++) {
                    if (getModelContext().attribute(j).isNumeric()) {
                        out.append((j == 0 || weightAttribute.getValue(j) < 0) ? " " : " + ");
                        out.append(String.format("%.4f", weightAttribute.getValue(j)));
                        out.append(" * ");
                        out.append(getAttributeNameString(j));
                    }
                }
                out.append(" + " + weightAttribute.getValue((getModelContext().numAttributes() - 1)));
            }
            StringUtils.appendNewline(out);
        }
    }

    public double computeSD(double squaredVal, double val, double size) {
        if (size > 1)
            return Math.sqrt((squaredVal - ((val * val) / size)) / size);
        else
            return 0.0;
    }

    public double scalarProduct(DoubleVector u, DoubleVector v) {
        double ret = 0.0;
        for (int i = 0; i < Math.max(u.numValues(), v.numValues()); i++) {
            ret += u.getValue(i) * v.getValue(i);
        }
        return ret;
    }


    public double normalizeTargetValue(double value) {
        if (examplesSeen > 1) {
            double sd = Math.sqrt((sumOfSquares - ((sumOfValues * sumOfValues)/examplesSeen))/examplesSeen);
            double average = sumOfValues / examplesSeen;
            if (sd > 0 && examplesSeen > 1)
                return (value - average) / (3 * sd);
            else
                return 0.0;
        }
        return 0.0;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        this.inactiveLeafByteSizeEstimate = 0.0;
        this.activeLeafByteSizeEstimate = 0.0;
        this.byteSizeEstimateOverheadFraction = 1.0;
        this.growthAllowed = true;
        this.examplesSeen = 0;
        this.sumOfValues = 0.0;
        this.sumOfSquares = 0.0;

        if (this.leafpredictionOption.getChosenIndex()>0) {
            this.removePoorAttsOption = null;
        }
    }
    @Override
    public void enforceTrackerLimit() {

    }
    @Override
    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
                                          SplitNode parent, int parentBranch) {

    }



}


