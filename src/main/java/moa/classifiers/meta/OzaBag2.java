package moa.classifiers.meta;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import com.github.javacliparser.IntOption;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiﬁer to train</li>
 * <li>-s : The number of models in the bag</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class OzaBag2 extends AbstractClassifier implements MultiClassClassifier, Regressor {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    protected Classifier[] ensemble;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double sum = 0 ;
        double prediction = 0 ;
        DoubleVector combinedVote = new DoubleVector();


        for (int i = 0; i < this.ensemble.length; i++) {
                DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
                if (inst.classAttribute().isNumeric()) {

                combinedVote.addValues(vote);
                sum = combinedVote.sumOfValues() ;
                }
                else {
                    if (vote.sumOfValues() > 0.0) {
                        vote.normalize();
                        combinedVote.addValues(vote);
                    }

                }

        }

        if (inst.classAttribute().isNumeric()) {
            prediction = sum / this.ensemble.length;
            return new double[]{prediction};
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0)};
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }
}
