package edu.uic.cs.nlp.findtask.da.weka;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.DmInstance;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class WekaOfflineDaClassifier extends WekaDaClassifier {

    Logger logger = LoggerFactory.getLogger(this.getClass());

    public WekaOfflineDaClassifier(Class<? extends Classifier> classiferClass,
                                   Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(classiferClass, featureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        return this.classifySession(session, null);

    }

    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {

        List<String> classified = new ArrayList<String>();
        // off line implementation
        List<DialogTurn> dTurns = session.getDialogTurns();

        for (int dTurnIndex = 0; dTurnIndex < dTurns.size(); dTurnIndex++) {
            DialogTurn dTurn = dTurns.get(dTurnIndex);
            if (dTurn.hasUtteranceTurn()) {
                if (actor != null && dTurn.getActor() != actor) {
                    continue;
                }
                Contexts contexts = DaTrainingInstanceCreator.extractDaDialogTurnContexts(session, dTurnIndex, null, featureExtractors);
                DmInstance dmInstance = new DmInstance(contexts, dTurn.getUtteranceTurn().getDa());
                classified.add(this.wekaClassifer.classify(dmInstance));
            }
        }

        return classified.toArray(new String[0]);
    }


}
