package edu.uic.cs.nlp.findtask.da.maxent;

import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import opennlp.tools.util.Sequence;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class MaxEntOfflineDaClassifier extends MaxEntDaClassifier {

    public MaxEntOfflineDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors) {
        super(featureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        List<UtteranceTurn> uts = session.getUtTurns();

        UtteranceTurn[] utArray = uts.toArray(new UtteranceTurn[uts.size()]);

        Sequence bestSequence = beam.bestSequence(utArray, new FindTaskSession[]{session});
        List<String> t = bestSequence.getOutcomes();
        return t.toArray(new String[t.size()]);
    }

    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {
        throw new IllegalStateException("This classifier is not intended to be run by speakers");
    }

}
