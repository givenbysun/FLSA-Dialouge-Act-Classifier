package edu.uic.cs.nlp.findtask.da.feature;

import edu.stanford.nlp.trees.Tree;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.dm.classifier.feature.AnalyzedUtterance;
import edu.uic.cs.nlp.dm.classifier.feature.UtteranceTurnAnalyzer;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class UtteranceSynaticExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		AnalyzedUtterance analyzedTurn = UtteranceTurnAnalyzer.analyzedUtterance(session, dTurn
				.getUtteranceTurn().getTurnIndex());

		Tree lastSentenceTree = analyzedTurn.getParseTree();

		if (lastSentenceTree != null) {
			Tree tree = null;
			if ("ROOT".equals(lastSentenceTree.label().toString()) && !lastSentenceTree.isLeaf()
					&& lastSentenceTree.children().length > 0) {
				tree = lastSentenceTree.children()[0];
			} else {
				tree = lastSentenceTree;
			}

			contexts.add(new StringFeature("#RootNode", tree.label().value()));

			if (!lastSentenceTree.isLeaf()) {
				Tree[] trees = tree.children();
				for (int i = 0; i < trees.length && i < 2; i++) {
					contexts.add(new StringFeature("#ChildNode" + i, trees[i].label().value()));
				}
			}
		}
		return contexts;
	}

	@Override
	public String getName() {
		return "SYNT";
	}

}
