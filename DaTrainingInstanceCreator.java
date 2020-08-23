package edu.uic.cs.nlp.findtask.da;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.google.common.base.Preconditions;
import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.util.CollectionUtils;
import opennlp.model.Event;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.DmInstance;
import edu.uic.cs.nlp.dm.classifier.MaxEntEventStream;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameState;

/**
 *
 */
public class DaTrainingInstanceCreator {

    static Logger logger = LoggerFactory.getLogger(DaTrainingInstanceCreator.class);


    public static MaxEntEventStream createDaTrainingEventStream(Collection<FindTaskSession> sessions, Collection<DialogTurnFeatureExtractor> featureExtractors, DialogGameInferencer dgInferencer, Actor actor) {
        Collection<DmInstance> dmInstances = createDaDmInstances(sessions, featureExtractors, dgInferencer, actor);

        List<Event> events = new ArrayList<Event>();

        for (DmInstance dmInstance : dmInstances) {
            if (dmInstance.getContexts().toFeatures().length == 0) {
                continue;
            }

            events.add(new Event(dmInstance.getOutcome(), dmInstance.getContexts().toFeatures()));

        }
        return new MaxEntEventStream(events.toArray(new Event[0]));
    }

    /**
     * Create the DmInstance for online, with dynamic dialog game inference.
     *
     * @param sessions
     * @param featureExtractors
     * @param dgInferencer
     * @return
     */
    public static Collection<DmInstance> createDaDmInstances(Collection<FindTaskSession> sessions,
                                                             Collection<DialogTurnFeatureExtractor> featureExtractors, DialogGameInferencer dgInferencer, Actor actor) {
        logger.debug("Creating DmInstances using {} FindTaskSessions", sessions.size());

        Collection<DmInstance> dmInstances = new ArrayList<DmInstance>();

        for (FindTaskSession session : sessions) {
            List<DialogTurn> dTurns = session.getDialogTurns();

            DialogGameState dialogGames = (dgInferencer == null ? null : new DialogGameState()); // for every session, we have a new DialogGame

            for (int dtIndex = 0; dtIndex < dTurns.size(); dtIndex++) {
                DialogTurn dTurn = dTurns.get(dtIndex);

                if (dTurn.hasUtteranceTurn()) {

                    /**
                     * If an actor was specified, and current actor is not the actor, skip it.
                     */
                    if (actor != null && dTurn.getActor() != actor) {
                        continue;
                    }

                    Object[] additionalContexts = (dgInferencer == null ? new Object[0] : new Object[]{dialogGames});
                    Contexts contexts = extractDaDialogTurnContexts(session, dtIndex, additionalContexts, featureExtractors);
                    dmInstances.add(new DmInstance(contexts, dTurn.getUtteranceTurn().getDa()));
                }

                if (dgInferencer != null) {
                    // infer the DialogGames after current turn
                    dgInferencer.inferenceDialogGame(session, dtIndex, null, dialogGames);
                }

            }
        }
        return dmInstances;
    }


    /**
     * Extract the contexts from a dialog turn.
     *
     * @param session            The session where the dialog turn located.
     * @param dTurnPosition      The dialog turn's index.
     * @param additionalContexts Additional contexts of the turn. For run time, a DialogGame is put in additional contexts.
     * @param featureExtractors  The feature extractors.
     * @return The extracted contexts.
     */
    public static Contexts extractDaDialogTurnContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts,
                                                       Collection<DialogTurnFeatureExtractor> featureExtractors) {
        Preconditions.checkArgument(session != null, "Session cannot be null");
        Preconditions.checkArgument(dTurnPosition >= 0, "The dialog turn's index must be >= 0");
        Preconditions.checkArgument(CollectionUtils.isNotEmpty(featureExtractors), "The feature extractors cannot be empty");

        Contexts contexts = new Contexts();
        for (DialogTurnFeatureExtractor daFeatureExtractor : featureExtractors) {
            contexts.addContexts(daFeatureExtractor.extractContexts(session, dTurnPosition, additionalContexts));
        }
        return contexts;
    }

}
