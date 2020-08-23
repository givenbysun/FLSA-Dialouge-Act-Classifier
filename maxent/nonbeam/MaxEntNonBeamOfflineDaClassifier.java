package edu.uic.cs.nlp.findtask.da.maxent.nonbeam;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.DataMiningUtil;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DaClassifier;
import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import opennlp.model.MaxentModel;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * The MaxEnt classifier without using beam search for offline experiments.
 */
public class MaxEntNonBeamOfflineDaClassifier extends MaxEntNonBeamDaClassifier implements DaClassifier {


    public MaxEntNonBeamOfflineDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(featureExtractors);
    }

    protected MaxEntNonBeamOfflineDaClassifier(MaxentModel model, Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(model, featureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        return classifySession(session, null);
    }


    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {
        List<String> classified = new ArrayList<String>();
        List<DialogTurn> dTurns = session.getDialogTurns();
        for (int dTurnIndex = 0; dTurnIndex < dTurns.size(); dTurnIndex++) {
            DialogTurn dTurn = dTurns.get(dTurnIndex);
            if (dTurn.hasUtteranceTurn()) {

                if (actor != null && dTurn.getActor() != actor) {
                    continue;
                }

                Contexts contexts = DaTrainingInstanceCreator.extractDaDialogTurnContexts(session, dTurnIndex, null, featureExtractors);

                String label = this.maxentModel.getBestOutcome(this.maxentModel.eval(contexts.toFeatures()));
                classified.add(label);
            }
        }
        return classified.toArray(new String[0]);
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions, Actor actor) {
        try {
            this.maxentModel = DataMiningUtil.train(DaTrainingInstanceCreator.createDaTrainingEventStream(sessions, featureExtractors, null, actor), 200, 0);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions) {
        this.trainClassifier(sessions, null);
    }
}
