package edu.uic.cs.nlp.findtask.da.maxent.nonbeam;

import com.google.common.base.Preconditions;
import edu.uic.cs.nlp.findtask.da.DaClassifier;
import edu.uic.cs.nlp.findtask.da.DaClassifierBase;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import opennlp.model.MaxentModel;

import java.util.Collection;

/**
 * The MaxEnt classifier without using beam search.
 *
 * @author Lin
 */
public abstract class MaxEntNonBeamDaClassifier extends DaClassifierBase implements DaClassifier {

    protected MaxentModel maxentModel;

    protected Collection<DialogTurnFeatureExtractor> featureExtractors;


    protected MaxEntNonBeamDaClassifier(MaxentModel maxentModel, Collection<DialogTurnFeatureExtractor> featureExtractors) {
        Preconditions.checkArgument(featureExtractors != null && featureExtractors.size() > 0, "Feature extractors cannot be empty");
        this.maxentModel = maxentModel;
        this.featureExtractors = featureExtractors;
    }

    public MaxEntNonBeamDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors) {
        this(null, featureExtractors);
    }


    @Override
    public Collection<DialogTurnFeatureExtractor> getFeatureExtractors() {
        return this.featureExtractors;
    }


}
