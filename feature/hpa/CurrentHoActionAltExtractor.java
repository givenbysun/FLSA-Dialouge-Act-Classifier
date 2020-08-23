package edu.uic.cs.nlp.findtask.da.feature.hpa;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class CurrentHoActionAltExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		if (dTurn.hasHoTurn()) {

			contexts.add("#IsActing", HapticAlternativeUtil.getEncodedAction(dTurn.getHoActionTurn().getAction()));
		}

		return contexts;
	}

	@Override
	public String getName() {
		return "HOA";
	}

}
