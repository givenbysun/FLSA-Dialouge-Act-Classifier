package edu.uic.cs.nlp.findtask.da.mallet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelSequence;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.BooleanFeature;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.NumFeature;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInitiatingInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;
import edu.uic.cs.nlp.findtask.predictor.DialogTurnEventReader;

public class FeatureExtractorPipe extends Pipe {

	private static final long serialVersionUID = 1L;

	private Collection<DialogTurnFeatureExtractor> daFeatureExtractors;

	private DialogGameInferencer dialogGameInferencer = new DialogGameInitiatingInferencer();

	Logger logger = LoggerFactory.getLogger(this.getClass());

	public FeatureExtractorPipe(Collection<DialogTurnFeatureExtractor> daFeatureExtractors) {
		super(new Alphabet(), new LabelAlphabet());
		this.daFeatureExtractors = daFeatureExtractors;
	}

	public Collection<DialogTurnFeatureExtractor> getDaFeatureExtractors() {
		return this.daFeatureExtractors;
	}

	@Override
	public Instance pipe(Instance carrier) {
		FindTaskSession session = (FindTaskSession) carrier.getData();
		return this.pipeFindTaskSession(session, true);
	}

	/**
	 * Pipe a FindTaskSession for Training
	 * 
	 * @param session
	 * @param addIfNotPresent
	 * @return
	 */
	public Instance pipeFindTaskSession(FindTaskSession session, boolean addIfNotPresent) {
		List<DialogTurn> dTurns = session.getDialogTurns();

		int uTurnCount = DialogTurnEventReader.countUtteranceDialogTurns(dTurns);

		FeatureVector[] featureVectors = new FeatureVector[uTurnCount];
		Label[] labels = new Label[uTurnCount];

		int utIndex = 0;
		DialogGameState dialogGames = new DialogGameState();
		for (int dTurnIndex = 0; dTurnIndex < dTurns.size(); dTurnIndex++) {
			DialogTurn dTurn = dTurns.get(dTurnIndex);

			if (dTurn.hasUtteranceTurn()) {
				Contexts contexts = this.extractContexts(session, dTurn.getPosition(), dialogGames, null);
				featureVectors[utIndex] = this.convertContextsToFeatureVector(contexts, addIfNotPresent);
				// the label is DA
				String da = dTurn.getUtteranceTurn().getDa();
				labels[utIndex] = ((LabelAlphabet) getTargetAlphabet()).lookupLabel(da, addIfNotPresent);
				utIndex++;
			}

			// infer the DialogGames after current turn
			this.dialogGameInferencer.inferenceDialogGame(session, dTurnIndex, null, dialogGames);

		}

		return new Instance(new FeatureVectorSequence(featureVectors), new LabelSequence(labels), "", "");
	}

	/**
	 * Extract FeatureVectorSpace for offline classification
	 * 
	 * @param session
	 * @param addIfNotPresent
	 * @return
	 */
	public FeatureVectorSequence extractFeatureVectorSequence(FindTaskSession session, boolean addIfNotPresent) {
		List<DialogTurn> dTurns = session.getDialogTurns();

		int uTurnCount = DialogTurnEventReader.countUtteranceDialogTurns(dTurns);
		FeatureVector[] featureVectors = new FeatureVector[uTurnCount];

		int utIndex = 0;
		DialogGameState dialogGames = new DialogGameState();
		for (int dtIndex = 0; dtIndex < dTurns.size(); dtIndex++) {
			DialogTurn dTurn = dTurns.get(dtIndex);

			if (dTurn.hasUtteranceTurn()) {
				Contexts contexts = this.extractContexts(session, dTurn.getPosition(), dialogGames, null);
				featureVectors[utIndex++] = this.convertContextsToFeatureVector(contexts, addIfNotPresent);
			}

			// infer the DialogGames after current turn
			this.dialogGameInferencer.inferenceDialogGame(session, dtIndex, null, dialogGames);
		}
		return new FeatureVectorSequence(featureVectors);
	}

	/**
	 * Pipe the FindTaskSession online
	 * 
	 * @param session
	 *            the session
	 * @param uptoUttTurnIndex
	 * @param classifiedLabels
	 * @param addIfNotPresent
	 * @return
	 */
	public FeatureVectorSequence extractOnlineFeatureVectorSequence(FindTaskSession session, int uptoUttTurnIndex,
			String[] classifiedLabels) {

		FeatureVector[] featureVectors = new FeatureVector[uptoUttTurnIndex + 1];

		List<DialogTurn> dTurns = session.getDialogTurns();
		int utIndex = 0;
		DialogGameState dialogGames = new DialogGameState();
		for (int dtIndex = 0; dtIndex < dTurns.size(); dtIndex++) {
			DialogTurn dTurn = dTurns.get(dtIndex);

			if (dTurn.hasUtteranceTurn()) {
				Contexts contexts = this.extractContexts(session, dTurn.getPosition(), dialogGames, classifiedLabels);
				featureVectors[utIndex++] = this.convertContextsToFeatureVector(contexts, false);

				if (utIndex > uptoUttTurnIndex) {
					break;
				}
			}
			// infer the DialogGames after current turn
			this.dialogGameInferencer.inferenceDialogGame(session, dtIndex, classifiedLabels, dialogGames);
		}

		return new FeatureVectorSequence(featureVectors);
	}

	Contexts extractContexts(FindTaskSession session, int dTurnPosition, DialogGameState dialogGames,
			String[] classifiedLabels) {
		return DaTrainingInstanceCreator.extractDaDialogTurnContexts(session, dTurnPosition,
                new Object[]{dialogGames, classifiedLabels}, this.daFeatureExtractors);
	}

	FeatureVector convertContextsToFeatureVector(Contexts contexts, boolean addIfNotPresent) {
		// feature vector
		List<Integer> feaureIndices = new ArrayList<Integer>();
		List<Double> featureValues = new ArrayList<Double>();

		Set<BooleanFeature> booleanContexts = contexts.getBooleanContexts();

		for (BooleanFeature feature : booleanContexts) {
			int featureIndex = getDataAlphabet().lookupIndex(feature.getKey(), addIfNotPresent);
			if (featureIndex != -1) {
				double value = feature.getValue() ? 1.0 : 0.0;
				feaureIndices.add(featureIndex);
				featureValues.add(value);
			}
		}

		// convert num features
		Set<NumFeature> numContexts = contexts.getNumContexts();
		for (NumFeature feature : numContexts) {
			int featureIndex = getDataAlphabet().lookupIndex(feature.getKey(), addIfNotPresent);
			if (featureIndex != -1) {
				feaureIndices.add(featureIndex);
				featureValues.add(feature.getValue());
			}
		}

		// convert string features
		Set<StringFeature> stringContexts = contexts.getStringContexts();
		for (StringFeature feature : stringContexts) {

			String featureKey = feature.getKey() + "-" + feature.getValue();
			int featureIndex = getDataAlphabet().lookupIndex(featureKey, addIfNotPresent);
			if (featureIndex != -1) {
				feaureIndices.add(featureIndex);
				featureValues.add(1.0);
			}
		}

		int[] indicesArr = new int[feaureIndices.size()];
		double[] valuesArr = new double[featureValues.size()];
		for (int i = 0; i < indicesArr.length; i++) {
			indicesArr[i] = feaureIndices.get(i);
			valuesArr[i] = featureValues.get(i);
		}

		return new FeatureVector(getDataAlphabet(), indicesArr, valuesArr);
	}

}
