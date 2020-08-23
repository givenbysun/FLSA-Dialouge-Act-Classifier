package edu.uic.cs.nlp.findtask.da;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;

/**
 * @author Lin
 */
public interface FindTaskSessionFilter {
    /**
     * Whether a session qualify for an experiment.
     *
     * @param session A session.
     * @return If the session qualifies, true; otherwise, false.
     */
    boolean qualify(FindTaskSession session);
}
