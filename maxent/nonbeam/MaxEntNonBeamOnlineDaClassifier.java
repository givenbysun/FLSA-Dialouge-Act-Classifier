package edu.uic.cs.nlp.findtask.da.maxent.nonbeam;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.DataMiningUtil;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DaClassifier;
import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInitiatingInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;
import opennlp.model.MaxentModel;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * The MaxEnt classifier without using beam search for online experiments.
 */
public class MaxEntNonBeamOnlineDaClassifier extends MaxEntNonBeamDaClassifier implements DaClassifier {

    private DialogGameInferencer dialogGameInferencer = new DialogGameInitiatingInferencer();


    public MaxEntNonBeamOnlineDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(featureExtractors);
    }


    protected MaxEntNonBeamOnlineDaClassifier(MaxentModel model, Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(model, featureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        String[] classified = new String[session.getUtTurnSize()];

        // online implementation
        List<DialogTurn> dTurns = session.getDialogTurns();
        int uTurnIndex = 0;
        DialogGameState dialogGames = new DialogGameState();
        for (int dTurnIndex = 0; dTurnIndex < dTurns.size(); dTurnIndex++) {
            DialogTurn dTurn = dTurns.get(dTurnIndex);
            if (dTurn.hasUtteranceTurn()) {

                Contexts contexts = DaTrainingInstanceCreator.extractDaDialogTurnContexts(session, dTurnIndex, new Object[]{dialogGames,
                        classified},
                        featureExtractors);
                String label = this.maxentModel.getBestOutcome(this.maxentModel.eval(contexts.toFeatures()));
                classified[uTurnIndex++] = label;

                // infer the DialogGames after current turn
                this.dialogGameInferencer.inferenceDialogGame(session, dTurnIndex, classified, dialogGames);
            }
        }
        return classified;
    }


    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions) {
        this.trainClassifier(sessions, null);
    }

    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {
        throw new IllegalStateException("Doesn't support actor for online experiments");
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions, Actor actor) {
        try {
            this.maxentModel = DataMiningUtil.train(DaTrainingInstanceCreator.createDaTrainingEventStream(sessions, featureExtractors, dialogGameInferencer, actor), 200, 2);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
