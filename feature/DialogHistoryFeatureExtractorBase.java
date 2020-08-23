package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.BooleanFeature;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

/**
 * Class <code>DialogHistoryFeatureExtractor</code> extracts features of previous effective turn
 *
 * @author Lin Chen linchen04@gmail.com
 * @since Jan 14, 2013 12:03:23 PM
 */
public abstract class DialogHistoryFeatureExtractorBase implements DialogTurnFeatureExtractor {



    public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts, boolean includePreviousDA) {
        Contexts contexts = new Contexts();

        DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

        DialogTurn prevDTurn = session.getDialogTurnByPosition(dTurnPosition, -1);

        if (prevDTurn != null) {

            contexts.add(new StringFeature("#PTURN_ACTOR", prevDTurn.getActor().toString()));
            contexts.add(new BooleanFeature("#PT_SAME_ACTOR", dTurn.getActor() == prevDTurn.getActor()));

            if (prevDTurn.hasUtteranceTurn() && includePreviousDA) {
                contexts.add(new StringFeature("#PTURN_TYPE", "UTTERANCE"));

                // for online testing, the previously classified DAs will be put at extraInfo object, in a String[]
                String[] onlineClassifiedLabels = this.getOnlineClassifiedLabels(additionalContexts);

                int uTurnPosition = session.getUtterancePosition(prevDTurn.getPosition());

                String previousDa = onlineClassifiedLabels == null ? prevDTurn.getUtteranceTurn().getDa()
                        : onlineClassifiedLabels[uTurnPosition];

                contexts.add(new StringFeature("#PTURN_DA", previousDa));
            } else if (prevDTurn.getActorTurnSize() == 1 && prevDTurn.hasGestureTurn()) {
                contexts.add(new StringFeature("#PTURN_TYPE", "POINTING"));
            } else if (prevDTurn.getActorTurnSize() == 1 && prevDTurn.hasHoTurn()) {
                contexts.add(new StringFeature("#PTURN_TYPE", "HOACTION"));
                contexts.add(new StringFeature("#PTURN_ACT", prevDTurn.getHoActionTurn().getAction().toString()));
            }

        }
        return contexts;
    }

    String[] getOnlineClassifiedLabels(Object[] additionalContexts) {
        if (additionalContexts != null) {
            for (Object o : additionalContexts) {
                if (o instanceof String[]) {
                    if (o != null) {
                        return (String[]) o;
                    }
                }
            }
        }
        return null;
    }

    @Override
    public String getName() {
        return "DH";
    }

}
