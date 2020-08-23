package edu.uic.cs.nlp.findtask.da.maxent;

import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import opennlp.tools.util.Sequence;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

/**
 * @author Lin
 */
public class MaxEntOnlineDaClassifier extends MaxEntDaClassifier {

    public MaxEntOnlineDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(featureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {

        String[] classifiedLabels = new String[session.getUtTurnSize()];
        for (int i = 0; i < session.getUtTurns().size(); i++) {
            UtteranceTurn[] utArray = new UtteranceTurn[i + 1];
            for (int j = 0; j <= i; j++) {
                utArray[j] = session.getUtTurn(j);
            }

            Sequence bestSequence = beam.bestSequence(utArray, new Object[]{session, classifiedLabels});
            List<String> t = bestSequence.getOutcomes();
            String[] partialTags = (String[]) t.toArray(new String[t.size()]);
            classifiedLabels[i] = partialTags[i];
        }

        return classifiedLabels;
    }

    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {
        throw new IllegalStateException("Doesn't support actor for online experiments");
    }


}
