package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.BooleanFeature;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

/**
 * Class <code>PreviousHoActionExtractor</code> extracts features of previous HO Action Turn
 * 
 * @author Lin Chen linchen04@gmail.com
 * @since Jan 14, 2013 12:03:23 PM
 * 
 */
public class PreviousHoActionExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		DialogTurn prevDTurn = session.getDialogTurnByPosition(dTurnPosition, -1);

		if (prevDTurn != null) {
			if (prevDTurn.getActorTurnSize() == 1 && prevDTurn.hasHoTurn()) {
				contexts.add(new StringFeature("#PHO_ACTOR", prevDTurn.getActor().toString()));
				contexts.add(new BooleanFeature("#PHO_SAME_ACTOR", dTurn.getActor() == prevDTurn.getActor()));
				contexts.add(new StringFeature("#PTURN_TYPE", "HOACTION"));
				contexts.add(new StringFeature("#PHO_ACT", prevDTurn.getHoActionTurn().getAction().toString()));
			}

		}
		return contexts;
	}

	@Override
	public String getName() {
		return "PHO";
	}

}
