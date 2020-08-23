package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.DialogGameTurn;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.feature.AnalyzedUtterance;
import edu.uic.cs.nlp.dm.classifier.feature.UtteranceTurnAnalyzer;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Dialogue Game feature extractor from the Dialogue Game annotations.
 */
public class DialogGameAnnotationFeatureExtractor implements DialogTurnFeatureExtractor {

    @Override
    public Contexts extractContexts(FindTaskSession session, int dTurnArrayIndex, Object[] additionalContext) {

        DialogTurn dTurn = session.getDialogTurnByPosition(dTurnArrayIndex);
        UtteranceTurn utteranceTurn = dTurn.getUtteranceTurn();


        Contexts contexts = new Contexts();
        if (additionalContext != null) {
            DialogGameTurn dialogGame = this.getDialogGameContext(utteranceTurn, session.getDialogGameTurns());

            if (dialogGame != null) {
                contexts.add("#DialogGame", dialogGame.getGameName());
            }
        }
        return contexts;
    }

    /**
     * The the dialog game context.
     * The algorithms are:
     * 1) get all the dialogue game turns which intersect with the utterance turn, and starts before the current utterance turn.
     * 2) filter the dialogue game turns which starts with current utterance turn.
     * 3) if there are multiple dialog game turn, return the one with highest level(most nested).
     *
     * @param utteranceTurn
     * @param dialogGameTurns
     * @return
     */
    DialogGameTurn getDialogGameContext(UtteranceTurn utteranceTurn, List<DialogGameTurn> dialogGameTurns) {
        List<DialogGameTurn> intersectTurns = new ArrayList<DialogGameTurn>();
        for (DialogGameTurn dialogGameTurn : dialogGameTurns) {
            BigDecimal dgStart = dialogGameTurn.getStart();
            BigDecimal turnStart = utteranceTurn.getStart();
            if (dgStart.compareTo(turnStart) >= 0) {
                continue;
            }
            if (dialogGameTurn.intersects(utteranceTurn) && dialogGameTurn.getStart().intValue() != utteranceTurn.getTurnIndex()) {
                intersectTurns.add(dialogGameTurn);
            }
        }

        if (intersectTurns.size() == 0) {
            return null;
        }

        if (intersectTurns.size() == 1) {
            return intersectTurns.get(0);
        }

        Collections.sort(intersectTurns, dialogGameTurnComparator);

        return intersectTurns.get(0);
    }


    /**
     * A comparator for DialogGameTurn, it prefer higher level of the dialogue game, and newest dialog games.
     */
    private static Comparator<DialogGameTurn> dialogGameTurnComparator = new Comparator<DialogGameTurn>() {
        @Override
        public int compare(DialogGameTurn o1, DialogGameTurn o2) {
            //we compare level first
            if (o1.getLevel() != o2.getLevel()) {
                //we prefer most nested dg
                if (o1.getLevel().getLevel() > o2.getLevel().getLevel()) {
                    return -1;
                } else {
                    return 1;
                }
            } else {
                // same level, we prefer the one start later
                return o2.getStart().compareTo(o1.getStart());
            }
        }
    };


    @Override
    public String getName() {
        return "DG";
    }

}
