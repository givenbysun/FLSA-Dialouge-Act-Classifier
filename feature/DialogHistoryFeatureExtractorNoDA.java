package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

/**
 * Class <code>DialogHistoryFeatureExtractor</code> extracts features of previous effective turn, but without include the previous DA.
 *
 * @author Lin Chen linchen04@gmail.com
 * @since Jan 14, 2013 12:03:23 PM
 */
public class DialogHistoryFeatureExtractorNoDA extends DialogHistoryFeatureExtractorBase implements DialogTurnFeatureExtractor {

    @Override
    public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
        return super.extractContexts(
                session, dTurnPosition, additionalContexts, false //false mean not to include the previous da.
        );
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
