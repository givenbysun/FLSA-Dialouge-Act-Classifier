package edu.uic.cs.nlp.findtask.da;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * A filter to filter session based on DAs of actors.
 *
 * @author Lin
 */
public class FindTaskSessionActorDaFilter implements FindTaskSessionFilter {

    private Logger logger = LoggerFactory.getLogger(FindTaskSessionActorDaFilter.class);
    private Actor actor;

    public FindTaskSessionActorDaFilter(Actor actor) {
        this.actor = actor;
    }

    @Override
    public boolean qualify(FindTaskSession session) {
        List<UtteranceTurn> utteranceTurnList = session.getUtTurns();

        int actorDaCount = 0;
        for (UtteranceTurn utteranceTurn : utteranceTurnList) {
            if (utteranceTurn.getActor() == actor) {
                actorDaCount++;
            }
        }

        logger.info("{} has {} DAs for Actor {}", new Object[]{session.getFindTaskSessionId(), actorDaCount, actor});


        return actorDaCount > 0;
    }
}
