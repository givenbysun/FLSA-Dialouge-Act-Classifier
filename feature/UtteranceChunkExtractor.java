package edu.uic.cs.nlp.findtask.da.feature;

import java.util.ArrayList;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.BooleanFeature;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.feature.AnalyzedUtterance;
import edu.uic.cs.nlp.dm.classifier.feature.UtteranceTurnAnalyzer;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.model.Chunk;

public class UtteranceChunkExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		AnalyzedUtterance analyzedTurn = UtteranceTurnAnalyzer.analyzedUtterance(session, dTurn
				.getUtteranceTurn().getTurnIndex());

		contexts.addBooleanContexts(this.extractChunkFeatures(analyzedTurn));

		return contexts;
	}

	List<BooleanFeature> extractChunkFeatures(AnalyzedUtterance analyzedTurn) {
		List<BooleanFeature> features = new ArrayList<BooleanFeature>();

		List<Chunk> chunks = analyzedTurn.getChunks();

		for (Chunk chunk : chunks) {
			String chunkType = chunk.getType();

			features.add(new BooleanFeature(chunkType, true));
		}

		return features;
	}

	@Override
	public String getName() {
		return "CHUNK";
	}

}
