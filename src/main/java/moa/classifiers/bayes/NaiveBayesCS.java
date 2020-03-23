/*
 *    NaiveBayes.java
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
package moa.classifiers.bayes;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;

import moa.core.InstanceExample;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import java.util.Random;
import moa.core.InstanceExample;
import static moa.streams.filters.CompressSensingGaussian.multiply;

/**
 * Naive Bayes incremental learner.
 *
 * <p>Performs classic bayesian prediction while making naive assumption that
 * all inputs are independent.<br /> Naive Bayes is a classiﬁer algorithm known
 * for its simplicity and low computational cost. Given n different classes, the
 * trained Naive Bayes classiﬁer predicts for every unlabelled instance I the
 * class C to which it belongs with high accuracy.</p>
 *
 * <p>Parameters:</p> <ul> <li>-r : Seed for random behaviour of the
 * classifier</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class NaiveBayesCS extends AbstractClassifier  implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;
     public IntOption dim = new IntOption("FeatureDimension", 'd',
            "the target feature dimension.", 10);
    protected InstancesHeader streamHeader;
    @Override
    public String getPurposeString() {
        return "Naive Bayes classifier: performs classic bayesian prediction while making naive assumption that all inputs are independent.";
    }
    protected DoubleVector observedClassDistribution;

    protected AutoExpandVector<AttributeClassObserver> attributeObservers;
protected double[][] GaussMatrix = new double[this.dim.getValue()][857-1];
    @Override
    public void resetLearningImpl() {
        Random r = new Random();  
        
        
       
        for(int i=0; i < this.dim.getValue(); i++){
            for(int j = 0; j < 857-1; j++){
                this.GaussMatrix[i][j]= r.nextGaussian();
            }
        }
        this.observedClassDistribution = new DoubleVector();
        this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
    this.streamHeader = null;
    }
 
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        
        Instance in;
        in = transformedInstance(inst, GaussianProjection(inst,this.dim.getValue(), this.GaussMatrix));
        this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight()); 
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
           
            int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs == null) {
                obs = inst.attribute(instAttIndex).isNominal() ? newNominalClassObserver()
                        : newNumericClassObserver();
                this.attributeObservers.set(i, obs);
            }
            obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return doNaiveBayesPrediction(inst, this.observedClassDistribution,
                this.attributeObservers);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    public static double[] multiply(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += a[i][j] * x[j];
        return y;
    }
    
    public DenseInstance transformedInstance(Instance sparseInst, double [] val) {
        
        Instances header = this.streamHeader;
        double[] attributeValues = new double[this.dim.getValue()];

        for(int i = 0 ; i < header.numAttributes()-1 ; i++) {
            attributeValues[i] = val[i];
        }

        attributeValues[attributeValues.length-1] = sparseInst.classValue();
        DenseInstance newInstance = new DenseInstance(1.0, attributeValues);
        newInstance.setDataset(header);
        return newInstance;
    }
    
        public  double[] GaussianProjection(Instance instance, int n, double[][] gm) {
       

        double [] denseValues = new double[n];
      
        double[] ins = new double [instance.numAttributes()-1]; 
        for(int i = 0 ; i < instance.numAttributes()-1 ; i++) {
            ins[i] = instance.value(i);
        }
         denseValues = multiply(gm, ins);
         
        return denseValues;
    }
        
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        for (int i = 0; i < this.observedClassDistribution.numValues(); i++) {
            StringUtils.appendIndented(out, indent, "Observations for ");
            out.append(getClassNameString());
            out.append(" = ");
            out.append(getClassLabelString(i));
            out.append(":");
            StringUtils.appendNewlineIndented(out, indent + 1,
                    "Total observed weight = ");
            out.append(this.observedClassDistribution.getValue(i));
            out.append(" / prob = ");
            out.append(this.observedClassDistribution.getValue(i)
                    / this.observedClassDistribution.sumOfValues());
            for (int j = 0; j < this.attributeObservers.size(); j++) {
                StringUtils.appendNewlineIndented(out, indent + 1,
                        "Observations for ");
                out.append(getAttributeNameString(j));
                out.append(": ");
                // TODO: implement observer output
                out.append(this.attributeObservers.get(j));
            }
            StringUtils.appendNewline(out);
        }
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    protected AttributeClassObserver newNominalClassObserver() {
        return new NominalAttributeClassObserver();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }

    public static double[] doNaiveBayesPrediction(Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> attributeObservers) {
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = observedClassDistribution.getValue(classIndex)
                    / observedClassSum;
            for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,
                        inst);
                AttributeClassObserver obs = attributeObservers.get(attIndex);
                if ((obs != null) && !inst.isMissing(instAttIndex)) {
                    votes[classIndex] *= obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex);
                }
            }
        }
        // TODO: need logic to prevent underflow?
        return votes;
    }

    // Naive Bayes Prediction using log10 for VFDR rules 
    public static double[] doNaiveBayesPredictionLog(Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> observers, AutoExpandVector<AttributeClassObserver> observers2) {
        AttributeClassObserver obs;
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = Math.log10(observedClassDistribution.getValue(classIndex)
                    / observedClassSum);
            for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,
                        inst);
                if (inst.attribute(instAttIndex).isNominal()) {
                    obs = observers.get(attIndex);
                } else {
                    obs = observers2.get(attIndex);
                }

                if ((obs != null) && !inst.isMissing(instAttIndex)) {
                    votes[classIndex] += Math.log10(obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex));

                }
            }
        }
        return votes;

    }

    public void manageMemory(int currentByteSize, int maxByteSize) {
        // TODO Auto-generated method stub
    }
}
