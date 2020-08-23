package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.NumFeature;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class UtteranceTurnFeatureExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		UtteranceTurn uTurn = dTurn.getUtteranceTurn();

		// turn length in time
		contexts.add(new NumFeature("#TIME_LEN", uTurn.getDurations().doubleValue()));

		contexts.add(new StringFeature("#Actor", uTurn.getActor().toString()));

		return contexts;
	}

	@Override
	public String getName() {
		return "UT";
	}

}
