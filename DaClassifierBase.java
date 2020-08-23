package edu.uic.cs.nlp.findtask.da;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Lin
 */
public abstract class DaClassifierBase implements DaClassifier {

    @Override
    public String[] extractLabels(FindTaskSession session) {
        String[] tags = new String[session.getUtTurns().size()];
        List<UtteranceTurn> turns = session.getUtTurns();
        for (int i = 0; i < turns.size(); i++) {
            tags[i] = turns.get(i).getDa();
        }
        return tags;
    }

    @Override
    public SessionTurnId[] extractSessionTurnIds(FindTaskSession session) {
        SessionTurnId[] ids = new SessionTurnId[session.getUtTurns().size()];
        List<UtteranceTurn> turns = session.getUtTurns();
        for (int i = 0; i < turns.size(); i++) {
            ids[i] = new SessionTurnId(session.getFindTaskSessionId(), i);
        }
        return ids;
    }

    @Override
    public FindTaskSessionFilter getFindTaskSessionFilter() {
        return null;
    }

    /**
     * Extract the DAs by actor.
     *
     * @param session
     * @param actor
     * @return
     */
    protected String[] extractLabels(FindTaskSession session, Actor actor) {
        List<String> das = new ArrayList<String>();

        List<UtteranceTurn> turns = session.getUtTurns();
        for (int i = 0; i < turns.size(); i++) {
            if (turns.get(i).getActor() == actor) {
                das.add(turns.get(i).getDa());
            }
        }
        return das.toArray(new String[0]);
    }

    /**
     * Extract the sessionTurnIds by actor.
     *
     * @param session
     * @param actor
     * @return
     */
    protected SessionTurnId[] extractSessionTurnIds(FindTaskSession session, Actor actor) {
        List<SessionTurnId> ret = new ArrayList<SessionTurnId>();
        List<UtteranceTurn> turns = session.getUtTurns();
        for (int i = 0; i < turns.size(); i++) {
            if (turns.get(i).getActor() == actor) {
                ret.add(new SessionTurnId(session.getFindTaskSessionId(), i));
            }
        }
        return ret.toArray(new SessionTurnId[0]);
    }


}
