package edu.uic.cs.nlp.findtask.da.weka;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindCorpusReader;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.DmInstance;
import edu.uic.cs.nlp.dm.classifier.TurnClassificationResult;
import edu.uic.cs.nlp.findtask.da.DaExperimenter;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.CurrentHoActionExtractor;
import edu.uic.cs.nlp.findtask.da.feature.CurrentPointingExtractor;
import edu.uic.cs.nlp.findtask.da.feature.LocationFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.DialogHistoryFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.TaskMetaExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceChunkExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceDependencyExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceHeuristicExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceSynaticExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceWordPosExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;

public class WekaOnlineDaClassifier extends WekaDaClassifier {

	Logger logger = LoggerFactory.getLogger(this.getClass());

	public WekaOnlineDaClassifier(Class<? extends Classifier> classiferClass,
			Collection<DialogTurnFeatureExtractor> featureExtractors) {
		super(classiferClass, featureExtractors);
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
				DmInstance dmInstance = new DmInstance(contexts, dTurn.getUtteranceTurn().getDa());
				classified[uTurnIndex++] = this.wekaClassifer.classify(dmInstance);

				// infer the DialogGames after current turn
				this.dialogGameInferencer.inferenceDialogGame(session, dTurnIndex, classified, dialogGames);
			}
		}
		return classified;
	}

    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {
        throw new IllegalStateException("Doesn't support actor for online experiments");
    }

    public static void main(String[] args) {
		List<FindTaskSession> corpus = new FindCorpusReader("D:/Academic/Projects/RoboHelper/AnvilAnnotations")
				.readFindTaskCorpus();
		DaExperimenter experimenter = new DaExperimenter();
		DialogTurnFeatureExtractor[] featureExtractors = new DialogTurnFeatureExtractor[] {
				new UtteranceTurnFeatureExtractor(), // TU

				new UtteranceChunkExtractor(),
				new UtteranceDependencyExtractor(),
				new UtteranceHeuristicExtractor(),
				new UtteranceSynaticExtractor(),
				new UtteranceWordPosExtractor(), // TX

				new CurrentHoActionExtractor(),
				new CurrentPointingExtractor(),
				new LocationFeatureExtractor(), // MM

				new DialogHistoryFeatureExtractor(), // PT
				new TaskMetaExtractor(), // TM

		};

		TurnClassificationResult result = experimenter.runNFoldExperiments(corpus,
				new WekaOnlineDaClassifier(J48.class, Arrays.asList(featureExtractors)));
		experimenter.printDaLabelingResult(corpus, result.getDaLabelResults());
	}

}
