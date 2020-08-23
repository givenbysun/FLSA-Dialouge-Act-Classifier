package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;

/**
 * Inferred dialogue game.
 */
public class DialogGameRuntimeFeatureExtractor implements DialogTurnFeatureExtractor {

    @Override
    public Contexts extractContexts(FindTaskSession session, int dTurnArrayIndex,
                                    Object[] additionalContext) {
        Contexts contexts = new Contexts();
        if (additionalContext != null) {
            DialogGameState dialogGames = this.getDialogGameContext(additionalContext);

            if (dialogGames != null && dialogGames.hasGame()) {
                contexts.add("#DialogGame", dialogGames.getCurrentGame());
            }
        }
        return contexts;
    }

    DialogGameState getDialogGameContext(Object[] additionalContexts) {
        if (additionalContexts != null) {
            for (Object context : additionalContexts) {
                if (context instanceof DialogGameState) {
                    return (DialogGameState) context;
                }
            }
        }
        return null;
    }

    @Override
    public String getName() {
        return "DG";
    }

}
