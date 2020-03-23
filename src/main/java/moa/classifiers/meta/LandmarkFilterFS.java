package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.rules.RuleClassifier;
import moa.classifiers.rules.RuleClassifierNBayes;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import weka.attributeSelection.*;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Jean Paul Barddal
 */
public class LandmarkFilterFS extends AbstractClassifier implements MultiClassClassifier {

    /////////////
    // OPTIONS //
    /////////////
    public IntOption chunkSizeOption = new IntOption("chunkSize",
            'w',
            "",
            1000,
            1,
            1000000);
    public ClassOption baseLeanerOption
            = new ClassOption("baseLeaner",
                    '1',
                    "Classifier to train.",
                    Classifier.class,
                    "trees.HoeffdingTree");
    public IntOption dMaxOption
            = new IntOption("dMax",
                    'd',
                    "Max amount of features to be selected.",
                    5,
                    1,
                    500);
    public MultiChoiceOption fsAlgorithmOption
            = new MultiChoiceOption("fsAlgorithm",
                    'f',
                    "",
                    new String[]{"Info_Gain",
                        "Correlation",
                        "Gain_Ratio",
                        "ReliefF",
                        "Symmetrical_Uncertainty"},
                    new String[]{"Info_Gain",
                        "Correlation",
                        "Gain_Ratio",
                        "ReliefF",
                        "Symmetrical_Uncertainty"},
                    0);

    ///////////////
    // INTERNALS //
    ///////////////
    ArrayList<Instance> currentChunk;
    Classifier classifier;
    ArrayList<Attribute> features;
    ArrayList<String> featureNames;
    InstancesHeader header;

    long instancesSeen;


    @Override
    public void resetLearningImpl() {
        classifier = (Classifier) getPreparedClassOption(baseLeanerOption);
        classifier.prepareForUse();
        classifier.resetLearning();
        currentChunk = new ArrayList<>();
        instancesSeen = 0;
        features = new ArrayList<>();
        featureNames = new ArrayList<>();
        header = null;

        if (classifier instanceof RuleClassifierNBayes) {
            classifier = new RuleClassifierNBayes();
            classifier.prepareForUse();
            classifier.resetLearning();
        } else if (classifier instanceof RuleClassifier) {
            classifier = new RuleClassifier();
            classifier.prepareForUse();
            classifier.resetLearning();
        }
    }


    @Override
    public void trainOnInstanceImpl(Instance instnc) {

        Instance converted = convertInstance(instnc);
        this.classifier.trainOnInstance(converted);
        if (currentChunk.size() == chunkSizeOption.getValue()) {
            //run FS algorithm
            List<String> newBestFS = featureSelection();
            //if newBestFS differs from current best fs
            //change the current best fs and resets classifier
            ArrayList<String> auxiliar = new ArrayList<>(featureNames);
            auxiliar.remove("class");
            if (!matches(newBestFS, auxiliar)) {
                featureNames = (ArrayList<String>) newBestFS;
                featureNames.add(instnc.classAttribute().name());
                features.clear();
                for (int i = 0; i < instnc.numAttributes(); i++) {
                    if (featureNames.contains(instnc.attribute(i).name())
                            || instnc.attribute(i).name().equals(instnc.classAttribute().name())) {
                        features.add(instnc.attribute(i));
                    }
                }
                header = new InstancesHeader(
                        new Instances(
                                getCLICreationString(InstanceStream.class),
                                features,
                                0));
                header.setClassIndex(features.size() - 1);
                classifier.resetLearning();
                classifier.setModelContext(header);
                classifier.prepareForUse();
                if (classifier instanceof RuleClassifier) {
                    classifier = new RuleClassifier();
                    classifier.prepareForUse();
                }

            }
            currentChunk.clear();
        }
        currentChunk.add(instnc);
        instancesSeen++;

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance instnc) {
        if (instancesSeen == 0) {
            featureNames = (ArrayList<String>) getWholeFS(instnc);
        }
        Instance converted = convertInstance(instnc);
        double ret[] = classifier.getVotesForInstance(converted);
        return ret;
    }

    private Instance convertInstance(Instance original) {

        if (header == null) {
            features = new ArrayList<>();
            featureNames = new ArrayList<>();

            for (int i = 0; i < original.numAttributes(); i++) {
                features.add(original.attribute(i));
                featureNames.add(original.attribute(i).name());
            }

            header = new InstancesHeader(
                    new Instances(getCLICreationString(InstanceStream.class),
                            features,
                            0));
            header.setClassIndex(features.size() - 1);
        }

        Instance converted = new DenseInstance(featureNames.size());
        converted.setDataset(header);
        int qtdFilled = 0;
        if (original.numAttributes() != featureNames.size()) {
            for (int i = 0; i < original.numAttributes(); i++) {
                if (this.featureNames.contains(original.attribute(i).name())) {
                    converted.setValue(qtdFilled,
                            original.value(original.attribute(i)));
                    qtdFilled++;
                }
            }
        } else {
            converted = original;
        }

        converted.setClassValue((int) original.classValue());
        return converted;
    }

    private boolean matches(List<String> listA, List<String> listB) {
        if (listA.size() != listB.size()) {
            return false;
        }
        return listA.stream().noneMatch((strA) -> (!listB.contains(strA)));
    }

    private List<String> getWholeFS(Instance instnc) {
        List<String> fs = new ArrayList<>();
        for (int i = 0; i < instnc.numAttributes(); i++) {
            fs.add(instnc.attribute(i).name());
        }
        fs.remove("class");
        return fs;
    }

    private List<String> featureSelection() {
        Ranker searcher = new Ranker();
        ASEvaluation eval = null;
        if (fsAlgorithmOption.getValueAsCLIString().equals("Info_Gain")) {
            eval = new InfoGainAttributeEval();
        } else if (fsAlgorithmOption.getValueAsCLIString().equals("Correlation")) {
            eval = new CorrelationAttributeEval();
        } else if (fsAlgorithmOption.getValueAsCLIString().equals("ReliefF")) {
            eval = new ReliefFAttributeEval();
        } else if (fsAlgorithmOption.getValueAsCLIString().equals("Gain_Ratio")) {
            eval = new GainRatioAttributeEval();
        } else if (fsAlgorithmOption.
                getValueAsCLIString().equals("Symmetrical_Uncertainty")) {
            eval = new SymmetricalUncertAttributeEval();
        }

        ArrayList<Attribute> allAttributes = new ArrayList<>();
        for (int i = 0; i < currentChunk.get(0).numAttributes(); i++) {
            allAttributes.add(currentChunk.get(0).attribute(i));
        }
        Instances instances = new Instances("name",
                allAttributes,
                currentChunk.size());
        instances.setClassIndex(currentChunk.get(0).numAttributes() - 1);
        currentChunk.stream().forEach((instnc) -> {
            instances.add(instnc);
        });

        int atts[] = null;
        try {
            SamoaToWekaInstanceConverter cvt = new SamoaToWekaInstanceConverter();            
            eval.buildEvaluator(cvt.wekaInstances(instances));
            atts = searcher.search(eval, cvt.wekaInstances(instances));
        } catch (Exception ex) {
            Logger.getLogger(LandmarkFilterFS.class.getName()).
                    log(Level.SEVERE, null, ex);
        }
        ArrayList<String> chosenSubset = new ArrayList<>();
        if (atts != null) {
            for (int index : atts) {
                chosenSubset.add(currentChunk.get(0).attribute(index).name());
            }
        }
        while (chosenSubset.size()
                > dMaxOption.getValue()) {
            chosenSubset.remove(chosenSubset.size() - 1);
        }
        return chosenSubset;
    }
    
}
