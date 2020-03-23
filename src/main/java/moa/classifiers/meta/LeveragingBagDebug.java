/*
 *    LeveragingBag.java
 *    Copyright (C) 2010 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
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
package moa.classifiers.meta;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import moa.classifiers.core.driftdetection.ADWIN;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import moa.core.DoubleVector;
import moa.core.Example;

import moa.core.MiscUtils;

public class LeveragingBagDebug extends LeveragingBag{

    private static final long serialVersionUID = 1L;
    
    protected long instancesSeen;
    
    // Files
    public FileOption dumpFileOption = new FileOption("dumpFile", '1',
            "File to append debug results to.", "debug_LevBag.csv", "csv", true);

    public IntOption dumpPeriodOption = new IntOption("dumpPeriod", 'k',
            "The amount of instances before outputing metrics",
            10000, 1, Integer.MAX_VALUE);

    public FileOption predictionsFileOption = new FileOption("predictionsFile", 'e',
            "File to append predictions.", "predictions_.csv", "csv", true);

    public FlagOption outputPredictionsOption = new FlagOption("outputPredictions", '5',
            "Whether to print predictions or not.");

    protected PrintStream predictionsStream;
    
    protected PrintStream resultStream;

    protected C[][] pairsOutputsKappa;
    
    @Override
    public void resetLearningImpl() {
        super.resetLearningImpl();
        this.instancesSeen = 0;
        createOpenOutputFile();
        
        if(this.resultStream != null) {
            this.resultStream.println(this.describe());
            StringBuilder header = new StringBuilder();
            header.append("[qty] InstancesSeen,[qty] TotalLearners,"
                    + "[avg] KappaDiv, [max] KappaDiv, [min] KappaDiv,"
            );

            this.resultStream.print(header.toString());
            this.resultStream.println();
        }

        if(this.outputPredictionsOption.isSet()) {
            this.predictionsStream.print("instances,");
            for (int i = 0; i < this.ensembleSizeOption.getValue(); ++i) {
                this.predictionsStream.print("learner(" + i + "),");
            }
            this.predictionsStream.println();
        }
    }



    @Override
    public double[] getVotesForInstance(Instance inst) {

        if(this.outputPredictionsOption.isSet()) {
            if (this.outputCodesOption.isSet()) {
                return getVotesForInstanceBinary(inst);
            }
            DoubleVector combinedVote = new DoubleVector();

            this.predictionsStream.print(this.instancesSeen + ",");

            for (int i = 0; i < this.ensemble.length; i++) {
                DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));

                DoubleVector outputVote = (DoubleVector) vote.copy();
                outputVote.normalize();

                this.predictionsStream.printf("%.4f,", outputVote.getValue(0));

                if (vote.sumOfValues() > 0.0) {
                    vote.normalize();
                    combinedVote.addValues(vote);
                }
            }
            this.predictionsStream.println();

            return combinedVote.getArrayRef();
        } else {
            return super.getVotesForInstance(inst);
        }

    }
    
    void outputStatistics() {
        // TODO: Watch out for bogus Kappa statistics (i.e. the learner is 'new' and cannot yield any results).
        double sumOfAllKappa = 0.0;
        int iMinKappaIdx = 0, jMinKappaIdx = 1;
        int iMaxKappaIdx = 0, jMaxKappaIdx = 1;
        int totalSimilarity = 0;

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            for(int j = i+1 ; j < this.ensemble.length ; ++j) {

                if(!Double.isNaN(this.pairsOutputsKappa[i][j].k())) {
                    totalSimilarity++;
                    sumOfAllKappa += this.pairsOutputsKappa[i][j].k();
                }

                if(this.getKappa(iMinKappaIdx,jMinKappaIdx) > this.getKappa(i,j) || Double.isNaN(this.getKappa(iMinKappaIdx,jMinKappaIdx))) {
                    iMinKappaIdx = i;
                    jMinKappaIdx = j;
                }
                if(this.getKappa(iMaxKappaIdx,jMaxKappaIdx) < this.getKappa(i, j) || Double.isNaN(this.getKappa(iMaxKappaIdx,jMaxKappaIdx))) {
                    iMaxKappaIdx = i;
                    jMaxKappaIdx = j;
                }
            }
        }
        double avgKappa = (sumOfAllKappa / (double) totalSimilarity);
            
//            double ensembleAccPreq = this.evaluator.getPerformanceMeasurements()[1].getValue();
            this.resultStream.print(
                            this.instancesSeen + "," + 
//                            this.mFeaturesPerTreeSizeOption.getValue() + "," + 
                            this.ensembleSizeOption.getValue() + "," +
                            avgKappa + "," + 
                            this.pairsOutputsKappa[iMaxKappaIdx][jMaxKappaIdx].k() + "," + 
                            this.pairsOutputsKappa[iMinKappaIdx][jMinKappaIdx].k() + ",");
            this.resultStream.println();
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        int numClasses = inst.numClasses();
        
        if(this.instancesSeen == 0) {
            // Init the Kappa matrix
            this.pairsOutputsKappa = new C[this.ensembleSizeOption.getValue()][this.ensembleSizeOption.getValue()];
            for(int i = 0 ; i < this.pairsOutputsKappa.length ; ++i) {
                for(int j = 0 ; j < this.pairsOutputsKappa[i].length ; ++j) {
                    this.pairsOutputsKappa[i][j] = new C(numClasses);
                }
            }
        }
        
        ++this.instancesSeen;
            
        // Kappa statistic
        HashMap<Integer, Integer> classifiersExactPredictions = new HashMap<>();
        
        //Output Codes
        if (this.initMatrixCodes == true) {
            this.matrixCodes = new int[this.ensemble.length][inst.numClasses()];
            for (int i = 0; i < this.ensemble.length; i++) {
                int numberOnes;
                int numberZeros;

                do { // until we have the same number of zeros and ones
                    numberOnes = 0;
                    numberZeros = 0;
                    for (int j = 0; j < numClasses; j++) {
                        int result = 0;
                        if (j == 1 && numClasses == 2) {
                            result = 1 - this.matrixCodes[i][0];
                        } else {
                            result = (this.classifierRandom.nextBoolean() ? 1 : 0);
                        }
                        this.matrixCodes[i][j] = result;
                        if (result == 1) {
                            numberOnes++;
                        } else {
                            numberZeros++;
                        }
                    }
                } while ((numberOnes - numberZeros) * (numberOnes - numberZeros) > (this.ensemble.length % 2));

            }
            this.initMatrixCodes = false;
        }


        boolean Change = false;
        Instance weightedInst = (Instance) inst.copy();
        double w = this.weightShrinkOption.getValue();

        //Train ensemble of classifiers
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            
            double k = 0.0;
            switch (this.leveraginBagAlgorithmOption.getChosenIndex()) {
                case 0: //LeveragingBag
                    k = MiscUtils.poisson(w, this.classifierRandom);
                    break;
                case 1: //LeveragingBagME
                    double error = this.ADError[i].getEstimation();
                    k = !this.ensemble[i].correctlyClassifies(weightedInst) ? 1.0 : (this.classifierRandom.nextDouble() < (error / (1.0 - error)) ? 1.0 : 0.0);
                    break;
                case 2: //LeveragingBagHalf
                    w = 1.0;
                    k = this.classifierRandom.nextBoolean() ? 0.0 : w;
                    break;
                case 3: //LeveragingBagWT
                    w = 1.0;
                    k = 1.0 + MiscUtils.poisson(w, this.classifierRandom);
                    break;
                case 4: //LeveragingSubag
                    w = 1.0;
                    k = MiscUtils.poisson(1, this.classifierRandom);
                    k = (k > 0) ? w : 0;
                    break;
            }
            if (k > 0) {
                if (this.outputCodesOption.isSet()) {
                    weightedInst.setClassValue((double) this.matrixCodes[i][(int) inst.classValue()]);
                }
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
            boolean correctlyClassifies = this.ensemble[i].correctlyClassifies(weightedInst);
            double ErrEstim = this.ADError[i].getEstimation();
            if (this.ADError[i].setInput(correctlyClassifies ? 0 : 1)) {
                if (this.ADError[i].getEstimation() > ErrEstim) {
                    Change = true;
                }
            }
            
            
            // Kappa statistic
            int maxIndex = vote.maxIndex();
            if(maxIndex < 0) 
                maxIndex = this.classifierRandom.nextInt(inst.numClasses());
            classifiersExactPredictions.put(i, maxIndex);
        }
        if (Change) {
            numberOfChangesDetected++;
            double max = 0.0;
            int imax = -1;
            for (int i = 0; i < this.ensemble.length; i++) {
                if (max < this.ADError[i].getEstimation()) {
                    max = this.ADError[i].getEstimation();
                    imax = i;
                }
            }
            if (imax != -1) {
                this.ensemble[imax].resetLearning();
                //this.ensemble[imax].trainOnInstance(inst);
                this.ADError[imax] = new ADWIN((double) this.deltaAdwinOption.getValue());
                
                // Update kappa matrix
                for(int j = imax+1 ; j < this.ensemble.length ; ++j) {
                    this.pairsOutputsKappa[imax][j].reset();
                }
                
            }
        }
        
        // Actually update Kappa statistic
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            for(int j = i+1 ; j < this.ensemble.length ; ++j) {
                this.pairsOutputsKappa[i][j].update(classifiersExactPredictions.get(i), 
                    classifiersExactPredictions.get(j));
            }
        }
        
//        "[qty] InstancesSeen,[qty] mSize,[qty] TotalLearners,[avg] KappaDiv, [max] KappaDiv, [min] KappaDiv,"
        if(this.instancesSeen % this.dumpPeriodOption.getValue() == 0) {
            outputStatistics();
        }
        
    }

    
    
    // For Kappa calculation
    protected class C{
    	private final long[][] C;
    	private final int numClasses;
    	private int instancesSeen = 0;
    	
    	public C(int numClasses) {
            this.numClasses = numClasses;
            this.C = new long[numClasses][numClasses];
    	}
        
        public void reset() {
            for(int i = 0 ; i < C.length ; ++i)
                for(int j = 0 ; j < C.length ; ++j)
                    C[i][j] = 0;
            this.instancesSeen = 0;
        }
    	
    	public void update(int iOutput, int jOutput) {
//            System.out.println("iOutput = " + iOutput + " jOutput = " + jOutput);
            this.C[iOutput][jOutput]++;
            this.instancesSeen++;
    	}
    	
    	public double theta1() {
            double sum = 0.0;
            for(int i = 0 ; i < C.length ; ++i)
                sum += C[i][i]; 

            return sum / (double) instancesSeen;
    	}
    	
    	public double theta2() {
            double sum1 = 0.0, sum2 = 0.0, sum = 0.0;

            for(int i = 0 ; i < C.length ; ++i) {
                for(int j = 0 ; j < C.length ; ++j) {
                    sum1 += C[i][j];
                    sum2 += C[j][i];
                }
            //	System.out.println("column = (" + sum1 + "," + (sum1 / (double) instancesSeen) + ") row = (" + sum2 + "," + (sum2 / (double) instancesSeen) + ")");
                sum += (sum1 / (double) instancesSeen) * (sum2 / (double) instancesSeen);
                sum1 = sum2 = 0;
            }
            return sum;
    	}
    	
    	public int getInstancesSeen() {
            return this.instancesSeen;
    	}
    	
        @Override
    	public String toString() {
            StringBuilder buffer = new StringBuilder();

            buffer.append("Instances seen = ");
            buffer.append(this.instancesSeen);
            buffer.append(" Theta1 = ");
            buffer.append(this.theta1());
            buffer.append(" Theta2 = ");
            buffer.append(this.theta2());
            buffer.append(" K = ");
            buffer.append(k());
            buffer.append("\n");
            
            buffer.append('*');
            buffer.append('\t');
            for(int i = 0 ; i < numClasses ; ++i) {
                buffer.append(i);
                buffer.append('\t');
            }
            buffer.append('\n');
            for(int i = 0 ; i < numClasses ; ++i){
                buffer.append(i);
                buffer.append('\t');
                for(int j = 0 ; j < numClasses ; ++j) {
                    buffer.append(C[i][j]);
                    buffer.append('\t');
                }
                buffer.append('\n');
            }
            return buffer.toString();
    	}
    	
    	public double k() {
            double t1 = theta1(), t2 = theta2();
            return (t1 - t2) / (double) (1.0 - t2);
    	}
    }

    protected double getKappa(int i, int j) {
        assert(i != j);
        if(this.pairsOutputsKappa == null)
            return Double.NaN;
        return i > j ? this.pairsOutputsKappa[j][i].k() : this.pairsOutputsKappa[i][j].k();
    }
    
    protected void createOpenOutputFile() {
        File dumpFile = this.dumpFileOption.getFile();
        File predFile = this.predictionsFileOption.getFile();

        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    this.resultStream = new PrintStream(new FileOutputStream(dumpFile, true), true);
                } else {
                    this.resultStream = new PrintStream(new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException("Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        if (predFile != null && this.outputPredictionsOption.isSet()) {
            try {
                if (predFile.exists()) {
                    this.predictionsStream = new PrintStream(new FileOutputStream(predFile, true), true);
                } else {
                    this.predictionsStream = new PrintStream(new FileOutputStream(predFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException("Unable to open predictions file: " + predFile, ex);
            }
        }
    }
    
    protected String describe() {
        StringBuilder description = new StringBuilder();
        description.append("LevBag_s");
        description.append(this.ensembleSizeOption.getValue());
        
        
        return description.toString();
    }
}

