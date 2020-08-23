package edu.uic.cs.nlp.findtask.da.feature;

import java.util.Arrays;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class UtteranceHeuristicExtractor implements DialogTurnFeatureExtractor {

	static List<String> ws;
	static List<String> yns;
	static List<String> poses;
	static List<String> neges;

	static {
		ws = Arrays.asList(new String[] { "who", "what", "whom", "where", "when", "how" });
		yns = Arrays.asList(new String[] { "do", "have", "did", "are", "is", "does", "can", "could" });
		poses = Arrays.asList(new String[] { "yes", "ok", "yeah", "okey", "uh-huh" });
		neges = Arrays.asList(new String[] { "no", "not", "nope", "don't" });
	}

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		UtteranceTurn turn = dTurn.getUtteranceTurn();
		String utterance = turn.getUtterance().toLowerCase();

		for (String cue : poses) {
			if (utterance.startsWith(cue)) {
				contexts.add("#YES", true);
				break;
			}
		}

		for (String cue : neges) {
			if (utterance.contains(cue)) {
				contexts.add("#NEG", true);
				break;
			}
		}

		if (utterance.endsWith("?")) {
			contexts.add("#Question", true);
		}

		for (String cue : yns) {
			if (utterance.startsWith(cue)) {
				contexts.add("#YN", true);
				break;
			}
		}

		for (String cue : ws) {
			if (utterance.startsWith(cue)) {
				contexts.add("#WH", true);
				break;
			}
		}

		if (utterance.endsWith("...")) {
			contexts.add("#Incomplete", true);
		}
		return contexts;
	}

	@Override
	public String getName() {
		return "HEUR";
	}
}
