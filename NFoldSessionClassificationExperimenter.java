package edu.uic.cs.nlp.findtask.da;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.ConfusionMatrix;
import edu.uic.cs.nlp.dm.ConfusionMatrixUtil;
import edu.uic.cs.nlp.dm.DataMiningUtil;
import edu.uic.cs.nlp.dm.Dataset;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;
import edu.uic.cs.nlp.dm.classifier.TurnClassificationResult;
import edu.uic.cs.nlp.dm.classifier.TurnLabelItem;

public class NFoldSessionClassificationExperimenter {

    Logger logger = LoggerFactory.getLogger(this.getClass());

    private OneFoldExperimenter oneFoldExperimenter;

    public NFoldSessionClassificationExperimenter(OneFoldExperimenter oneFoldExperimenter) {
        this.oneFoldExperimenter = oneFoldExperimenter;
    }

    public TurnClassificationResult nFoldTesting(List<FindTaskSession> corpus, FindTaskSessionFilter sessionFilter, int fold) {
        logger.info("Running {} fold testing using {} sessions", fold, corpus.size());

        List<FindTaskSession> sessions = null;

        if (sessionFilter == null) {
            sessions = corpus;
        } else {
            sessions = new ArrayList<FindTaskSession>();
            for (FindTaskSession session : corpus) {
                if (sessionFilter.qualify(session)) {
                    sessions.add(session);
                }
            }
        }

        Map<String, FindTaskSession> corpusMap = new HashMap<String, FindTaskSession>();
        for (FindTaskSession session : sessions) {
            corpusMap.put(session.getFindTaskSessionId(), session);
        }

        TurnClassificationResult experimentResult = new TurnClassificationResult(this.oneFoldExperimenter.getName(),
                this.oneFoldExperimenter.getFeatureNames());

        List<TurnLabelItem> daLabelResults = new ArrayList<TurnLabelItem>();

        List<Dataset<FindTaskSession>> datasets = DataMiningUtil.genNFoldDatasets(sessions, fold);
        for (int i = 0; i < datasets.size(); i++) {
            Dataset<FindTaskSession> dataset = datasets.get(i);

            logger.info("Running Fold {} testing with {} sessions as training", i + 1, dataset.getTraining().size());

            List<TurnLabelItem> foldResult = this.oneFoldTesting(dataset);

            ConfusionMatrix foldMatrix = this.calculateConfusionMatrix(foldResult);

            logger.info("Fold Confusion Matrix is: \n{}", foldMatrix.toString());

            logger.info("Fold Result, accuracy: {}", ConfusionMatrixUtil.calculateAccuracy(foldMatrix));

            experimentResult.addFoldMatrix(foldMatrix);

            daLabelResults.addAll(foldResult);
        }

		/* calculate matrix by actor */
        ConfusionMatrix eldMatrix = this.calculateActorConfusionMatrix(daLabelResults, corpusMap, Actor.ELD);
        ConfusionMatrix helMatrix = this.calculateActorConfusionMatrix(daLabelResults, corpusMap, Actor.HEL);
        experimentResult.setFinalMatrix(Actor.ELD.toString(), eldMatrix);
        experimentResult.setFinalMatrix(Actor.HEL.toString(), helMatrix);

		/* calculate the matrix for all utterances */
        ConfusionMatrix finalMatrix = this.calculateConfusionMatrix(daLabelResults);
        logger.info("Final Confusion Matrix is: \n{}", finalMatrix.toString());

        logger.info("Final Result, accuracy: {}", ConfusionMatrixUtil.calculateAccuracy(finalMatrix));

        experimentResult.setFinalMatrix(finalMatrix);

        experimentResult.setDaLabelResults(daLabelResults);

        return experimentResult;

    }

    List<TurnLabelItem> oneFoldTesting(Dataset<FindTaskSession> dataset) {
        this.oneFoldExperimenter.trainClassifier(dataset.getTraining());

        return this.oneFoldExperimenter.classifySessions(dataset.getTesting());
    }

    ConfusionMatrix calculateConfusionMatrix(List<TurnLabelItem> items) {
        String[] trueLabels = new String[items.size()];
        String[] classified = new String[items.size()];

        for (int i = 0; i < items.size(); i++) {
            TurnLabelItem item = items.get(i);
            trueLabels[i] = item.getLabel();
            classified[i] = item.getClassified();
        }

        return ConfusionMatrixUtil.calculateConfusionMatrix(trueLabels, classified);
    }

    /**
     * Calculate the confusion matrix by actor
     *
     * @param items
     * @param corpus
     * @param actor
     * @return
     */
    ConfusionMatrix calculateActorConfusionMatrix(List<TurnLabelItem> items, Map<String, FindTaskSession> corpus,
                                                  Actor actor) {
        List<String> trueLabels = new ArrayList<String>();
        List<String> classified = new ArrayList<String>();

        for (int i = 0; i < items.size(); i++) {
            TurnLabelItem item = items.get(i);

            SessionTurnId stId = item.getSessionTurnId();

            Actor itemActor = corpus.get(stId.getSessionId()).getUtTurn(stId.getTurnIndex()).getActor();

            if (itemActor == actor) {
                trueLabels.add(item.getLabel());
                classified.add(item.getClassified());

            }
        }

        return ConfusionMatrixUtil.calculateConfusionMatrix(trueLabels.toArray(new String[trueLabels.size()]),
                classified.toArray(new String[classified.size()]));
    }

}
