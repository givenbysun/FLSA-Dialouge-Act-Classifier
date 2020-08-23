package edu.uic.cs.nlp.findtask.da.feature;

import java.util.Collection;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.dm.classifier.feature.AnalyzedUtterance;
import edu.uic.cs.nlp.dm.classifier.feature.UtteranceTurnAnalyzer;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.util.StanfordParserUtil;

public class UtteranceDependencyExtractor implements DialogTurnFeatureExtractor {

	public UtteranceDependencyExtractor() {
	}

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		AnalyzedUtterance analyzedTurn = UtteranceTurnAnalyzer.analyzedUtterance(session, dTurn
				.getUtteranceTurn().getTurnIndex());

		Tree lastSentenceTree = analyzedTurn.getParseTree();

		if (lastSentenceTree != null) {
			Collection<TypedDependency> tds = StanfordParserUtil.extractTypedDependencies(lastSentenceTree);
			if (tds != null && tds.size() > 0) {
				for (TypedDependency td : tds) {
					contexts.add(new StringFeature("#DP_" + td.gov().toString(), td.reln().toString()));
				}
			}
		}
		return contexts;
	}

	@Override
	public String getName() {
		return "DEP";
	}

}
