package edu.uic.cs.nlp.findtask.da;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.ExperimentUtils;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;
import edu.uic.cs.nlp.dm.classifier.TurnLabelItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DaOneFoldExperimenter implements OneFoldExperimenter {

    private Logger logger = LoggerFactory.getLogger(DaOneFoldExperimenter.class);

    private DaClassifier daClassifier;

    public DaOneFoldExperimenter(DaClassifier daClassifier) {
        this.daClassifier = daClassifier;
    }

    @Override
    public String getName() {
        return this.daClassifier.getClass().getSimpleName();
    }

    @Override
    public void trainClassifier(List<FindTaskSession> sessions) {

        List<SessionTurnId> trainingIds = new ArrayList<SessionTurnId>();
        for (FindTaskSession session : sessions) {
            trainingIds.addAll(Arrays.asList(daClassifier.extractSessionTurnIds(session)));
        }

        if (trainingIds.isEmpty()) {
            logger.error("No training data is available");
            return;
        }

        this.daClassifier.trainClassifier(sessions);
    }

    @Override
    public List<TurnLabelItem> classifySessions(List<FindTaskSession> sessions) {


        List<TurnLabelItem> items = new ArrayList<TurnLabelItem>();
        for (FindTaskSession session : sessions) {
            items.addAll(this.classifySession(session));
        }
        return items;
    }

    public List<TurnLabelItem> classifySession(FindTaskSession session) {
        DaClassifier daClassifier = this.getDaClassifier();

        SessionTurnId[] sessionIds = daClassifier.extractSessionTurnIds(session);
        String[] trueLabels = daClassifier.extractLabels(session);
        String[] classifiedLabels = daClassifier.classifySession(session);

        if (trueLabels.length != classifiedLabels.length) {
            throw new IllegalStateException("Corrected and classified labels are NOT the same length");
        }

        List<TurnLabelItem> items = new ArrayList<TurnLabelItem>();
        for (int i = 0; i < sessionIds.length; i++) {
            items.add(new TurnLabelItem(sessionIds[i], trueLabels[i], classifiedLabels[i]));
        }
        return items;
    }


    public DaClassifier getDaClassifier() {
        return this.daClassifier;
    }

    @Override
    public String toString() {
        StringBuilder b = new StringBuilder();

        b.append("Experiment with Classisifer: " + this.daClassifier.getClass());
        b.append("The Features Used include: \n");
        for (DialogTurnFeatureExtractor daFeatureExtractor : this.daClassifier.getFeatureExtractors()) {
            b.append("    ").append(daFeatureExtractor.getClass().getSimpleName());
        }
        return b.toString();
    }

    @Override
    public String getFeatureNames() {
        return ExperimentUtils.genName(this.daClassifier.getFeatureExtractors());
    }

}
