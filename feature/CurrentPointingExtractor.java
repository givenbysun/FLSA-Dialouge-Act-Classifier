package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class CurrentPointingExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		if (dTurn.hasGestureTurn()) {
			contexts.add("#Pointing", true);
		}

		return contexts;
	}

	@Override
	public String getName() {
		return "GS";
	}

}
