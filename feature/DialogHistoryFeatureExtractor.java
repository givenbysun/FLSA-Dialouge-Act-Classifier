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
public class DialogHistoryFeatureExtractor extends DialogHistoryFeatureExtractorBase implements DialogTurnFeatureExtractor {

    @Override
    public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
       return super.extractContexts(session, dTurnPosition, additionalContexts, true);
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
