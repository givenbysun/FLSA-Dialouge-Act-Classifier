package edu.uic.cs.nlp.findtask.da.feature;

import java.math.BigDecimal;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.classifier.BooleanFeature;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.NumFeature;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class TaskMetaExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		contexts.add(new BooleanFeature("#IS_START", dTurnPosition == 0));
		contexts.add(new BooleanFeature("#IS_LAST", dTurnPosition == session.getDialogTurns().size() - 1));
		contexts.add(new NumFeature("#DIST_FROM_START", this.getDistanceFromStart(session, dTurnPosition)
				.doubleValue()));
		contexts.add(new NumFeature("#UTT_INDEX", dTurnPosition));

		return contexts;
	}

	BigDecimal getDistanceFromStart(FindTaskSession session, int dTurnPosition) {
		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		UtteranceTurn turn = dTurn.getUtteranceTurn();

		BigDecimal dist = turn.getStart().subtract(session.getTaskSpan().getStart());

		if (dist.compareTo(BigDecimal.ZERO) < 0) {
			dist = BigDecimal.ZERO;
		}
		return dist;
	}

	@Override
	public String getName() {
		return "TM";
	}
}
