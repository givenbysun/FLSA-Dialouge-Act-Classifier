package edu.uic.cs.nlp.findtask.da.weka;

import java.util.Collection;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.findtask.da.DaClassifierBase;
import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import java_cup.action_part;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.findtask.da.DaClassifier;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInitiatingInferencer;
import edu.uic.cs.nlp.util.weka.WekaClassifier;
import edu.uic.cs.nlp.util.weka.WekaClassifierTrainer;

public abstract class WekaDaClassifier extends DaClassifierBase implements DaClassifier {

    Logger logger = LoggerFactory.getLogger(this.getClass());

    protected WekaClassifierTrainer wekaClassiferTrainer = new WekaClassifierTrainer();

    protected Collection<DialogTurnFeatureExtractor> featureExtractors;

    protected Class<? extends Classifier> classiferClass;

    protected WekaClassifier wekaClassifer;

    protected DialogGameInferencer dialogGameInferencer = new DialogGameInitiatingInferencer();

    public WekaDaClassifier(Class<? extends Classifier> classiferClass, Collection<DialogTurnFeatureExtractor> featureExtractors) {
        this.featureExtractors = featureExtractors;
        this.classiferClass = classiferClass;
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions) {
        this.trainClassifier(sessions, null);
    }


    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions, Actor actor) {
        this.wekaClassifer = this.wekaClassiferTrainer.trainWekaClassifier(this.classiferClass,
                DaTrainingInstanceCreator.createDaDmInstances(sessions, featureExtractors, dialogGameInferencer, actor),
                this.classiferClass.getName());
    }

    @Override
    public Collection<DialogTurnFeatureExtractor> getFeatureExtractors() {
        return this.featureExtractors;
    }

}
