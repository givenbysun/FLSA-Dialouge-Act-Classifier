package edu.uic.cs.nlp.findtask.da.feature;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.NumFeature;
import edu.uic.cs.nlp.dm.classifier.feature.AnalyzedUtterance;
import edu.uic.cs.nlp.dm.classifier.feature.UtteranceTurnAnalyzer;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.model.PosToken;
import edu.uic.cs.nlp.util.LexiconUtil;
import edu.uic.cs.nlp.util.StanfordParserUtil;

public class UtteranceWordPosExtractor implements DialogTurnFeatureExtractor {

	Logger logger = LoggerFactory.getLogger(getClass());

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		AnalyzedUtterance analyzedTurn = UtteranceTurnAnalyzer.analyzedUtterance(session, dTurn
				.getUtteranceTurn().getTurnIndex());

		List<PosToken> posedTokens = analyzedTurn.getPosTokens();

		for (PosToken token : posedTokens) {
			String word = token.getWord();
			String baseWord = null;
			if (LexiconUtil.isWord(word)) {
				baseWord = StanfordParserUtil.getMorphology(word, token.getPos());
				contexts.add(this.genWordFeatureName(baseWord, 0), true);
				contexts.add(this.genPosFeatureName(token.getPos(), 0), true);
			}
		}

		// how many tokens in the utterance
		contexts.add(new NumFeature("#TERM_CNT", analyzedTurn.getPosTokens().size()));

		// how many sentence are there
		contexts.add(new NumFeature("#SEN_CNT", analyzedTurn.getSentences().size()));

		// how many words in the utterance
		contexts.add(new NumFeature("#WORD_CNT", this.countWords(analyzedTurn.getPosTokens())));

		return contexts;
	}

	int countWords(List<PosToken> tokens) {
		int count = 0;
		for (PosToken token : tokens) {
			if (LexiconUtil.isWord(token.getWord())) {
				count++;
			}
		}
		return count;
	}

	private String genWordFeatureName(String word, int utteranceOffset) {
		return "#WORD_" + word + (utteranceOffset == 0 ? "" : "_" + utteranceOffset);
	}

	private String genPosFeatureName(String pos, int utteranceOffset) {
		return "#POS_" + pos + (utteranceOffset == 0 ? "" : "_" + utteranceOffset);
	}

	@Override
	public String getName() {
		return "WORD";
	}

}
