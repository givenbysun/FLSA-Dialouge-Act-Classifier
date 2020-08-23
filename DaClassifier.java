package edu.uic.cs.nlp.findtask.da;

import java.util.Collection;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;

public interface DaClassifier extends HasFeatureExtractors {

    Collection<DialogTurnFeatureExtractor> getFeatureExtractors();

    String[] classifySession(FindTaskSession session);

    String[] extractLabels(FindTaskSession session);

    SessionTurnId[] extractSessionTurnIds(FindTaskSession session);

    void trainClassifier(Collection<FindTaskSession> sessions);


    /**
     * @return A FindTaskSessionFilter, by default it will return zero, which make sure all the DAs are used. For actor based DA classifiers, we return a DA based filter.
     */
    FindTaskSessionFilter getFindTaskSessionFilter();

    /**
     * Classify the utterance turns belonging to the actor only.
     *
     * @param session The session.
     * @param actor   The actor.
     * @return
     */
    String[] classifySession(FindTaskSession session, Actor actor);


    /**
     * Train the classifier with the turns belonging to the actor only.
     *
     * @param sessions The sessions.
     * @param actor    The actor.
     */
    void trainClassifier(Collection<FindTaskSession> sessions, Actor actor);
}
