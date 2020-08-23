package edu.uic.cs.nlp.findtask.da.maxent;

import java.util.Collection;

import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import opennlp.tools.util.BeamSearchContextGenerator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.HasFeatureExtractors;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;

public class MaxEntDaContextGenerator implements BeamSearchContextGenerator<UtteranceTurn>, HasFeatureExtractors {

	Logger logger = LoggerFactory.getLogger(MaxEntDaContextGenerator.class);
	private Collection<DialogTurnFeatureExtractor> featureExtractors;
	private DialogGameInferencer dgInferencer;

	public MaxEntDaContextGenerator(Collection<DialogTurnFeatureExtractor> featureExtractors,
			DialogGameInferencer dgInferencer) {
		this.featureExtractors = featureExtractors;
		this.dgInferencer = dgInferencer;
	}

	@Override
	public String[] getContext(int utIndex, UtteranceTurn[] sequence, String[] priorDecisions,
			Object[] additionalContext) {
		FindTaskSession session = this.getFindTaskSession(additionalContext);

		int dTurnPosition = session.getDialogTurnPositionByUtterancePosition(utIndex);

		// check if classified labels are provided
		String[] classifiedLabels = this.getOnlineClassifiedLabels(additionalContext);

		// infer dialog games before current session
		DialogGameState games = this.inferDialogGames(session, dTurnPosition, classifiedLabels);

		return DaTrainingInstanceCreator.extractDaDialogTurnContexts(session, dTurnPosition, new Object[]{classifiedLabels, games},
                this.featureExtractors).toFeatures();

	}

	DialogGameState inferDialogGames(FindTaskSession session, int uptoDTurnPosition, String[] classifiedLabels) {
		DialogGameState games = new DialogGameState();

		// don't infer dialog game from current turn
		for (int dtIndex = 0; dtIndex < uptoDTurnPosition; dtIndex++) {
			this.dgInferencer.inferenceDialogGame(session, dtIndex, classifiedLabels, games);
		}

		return games;
	}

	FindTaskSession getFindTaskSession(Object[] additionalContexts) {
		if (additionalContexts != null) {
			for (Object o : additionalContexts) {
				if (o instanceof FindTaskSession) {
					return (FindTaskSession) o;
				}
			}
		}
		return null;
	}

	String[] getOnlineClassifiedLabels(Object[] additionalContexts) {
		if (additionalContexts != null) {
			for (Object o : additionalContexts) {
				if (o instanceof String[]) {
					return (String[]) o;
				}
			}
		}
		return null;
	}

	@Override
	public Collection<DialogTurnFeatureExtractor> getFeatureExtractors() {
		return this.featureExtractors;
	}

}
