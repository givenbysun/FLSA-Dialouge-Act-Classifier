package edu.uic.cs.nlp.findtask.da.maxent.nonbeam;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.FindTaskSessionActorDaFilter;
import edu.uic.cs.nlp.findtask.da.FindTaskSessionFilter;
import opennlp.model.MaxentModel;

import java.util.Collection;

/**
 * Classifier to classify DA by actors.
 *
 * @author Lin
 */
public class MaxEntNonBeamOfflineActorDaClassifier extends MaxEntNonBeamOfflineDaClassifier {
    private Actor actor;

    public MaxEntNonBeamOfflineActorDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors, Actor actor) {
        super(featureExtractors);
        this.actor = actor;
    }

    protected MaxEntNonBeamOfflineActorDaClassifier(MaxentModel model, Collection<DialogTurnFeatureExtractor> featureExtractors, Actor actor) {
        super(model, featureExtractors);
        this.actor = actor;
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions) {
        super.trainClassifier(sessions, actor);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        return super.classifySession(session, actor);
    }

    @Override
    public String[] extractLabels(FindTaskSession session) {
        return super.extractLabels(session, actor);
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
