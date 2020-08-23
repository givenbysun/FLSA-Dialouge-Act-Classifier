package edu.uic.cs.nlp.findtask.da.weka;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.FindTaskSessionActorDaFilter;
import edu.uic.cs.nlp.findtask.da.FindTaskSessionFilter;
import weka.classifiers.Classifier;

import java.util.Collection;

/**
 * @author Lin
 */
public class WekaOfflineActorDaClassifier extends WekaOfflineDaClassifier {

    private Actor actor;


    public WekaOfflineActorDaClassifier(Class<? extends Classifier> classifierClass, Collection<DialogTurnFeatureExtractor> featureExtractors, Actor actor) {
        super(classifierClass, featureExtractors);
        this.actor = actor;
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        return super.classifySession(session, actor);
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions) {
        super.trainClassifier(sessions, actor);
    }

    @Override
    public String[] extractLabels(FindTaskSession session) {
        return extractLabels(session, actor);
    }

    @Override
    public SessionTurnId[] extractSessionTurnIds(FindTaskSession session) {
        return super.extractSessionTurnIds(session, actor);
    }

    @Override
    public FindTaskSessionFilter getFindTaskSessionFilter() {
        return new FindTaskSessionActorDaFilter(actor);
    }
}
