package edu.uic.cs.nlp.findtask.da.mallet;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.Sequence;
import edu.uic.cs.nlp.anvil.eah.FindCorpusReader;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.TurnClassificationResult;
import edu.uic.cs.nlp.findtask.da.DaExperimenter;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.CurrentHoActionExtractor;
import edu.uic.cs.nlp.findtask.da.feature.CurrentPointingExtractor;
import edu.uic.cs.nlp.findtask.da.feature.LocationFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.DialogHistoryFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.TaskMetaExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceChunkExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceDependencyExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceHeuristicExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceSynaticExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.UtteranceWordPosExtractor;

public class MalletCrfOfflineDaClassifier extends MalletCrfDaClassifier {

    Logger logger = LoggerFactory.getLogger(this.getClass());

    public MalletCrfOfflineDaClassifier(Collection<DialogTurnFeatureExtractor> daFeatureExtractors) {
        super(daFeatureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        if (this.crf == null) {
            throw new IllegalStateException("CRF has not been trained yet");
        }
        @SuppressWarnings("unchecked")
        Sequence<String> classifiedSeq = this.crf.transduce(this.featureExtractPipe
                .extractFeatureVectorSequence(session, false));
        return this.convertLabelSequenceToStrings(classifiedSeq);
    }


    @Override
    public String[] classifySession(FindTaskSession session, Actor actor) {
        throw new IllegalStateException("This classifier is not intended to be run by speakers");
    }


    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions, Actor actor) {
        throw new IllegalStateException("This classifier is not intended to be run by speakers");
    }

    public static void main(String[] args) {
        List<FindTaskSession> corpus = new FindCorpusReader("D:/Academic/Projects/RoboHelper/AnvilAnnotations")
                .readFindTaskCorpus();
        DaExperimenter experimenter = new DaExperimenter();
        DialogTurnFeatureExtractor[] featureExtractors = new DialogTurnFeatureExtractor[]{
                new UtteranceTurnFeatureExtractor(), // TU

                new UtteranceChunkExtractor(),
                new UtteranceDependencyExtractor(),
                new UtteranceHeuristicExtractor(),
                new UtteranceSynaticExtractor(),
                new UtteranceWordPosExtractor(), // TX

                new CurrentHoActionExtractor(),
                new CurrentPointingExtractor(),
                new LocationFeatureExtractor(), // MM

                new DialogHistoryFeatureExtractor(), // PT
                new TaskMetaExtractor(), // TM

        };

        TurnClassificationResult result = experimenter.runNFoldExperiments(corpus,
                new MalletCrfOfflineDaClassifier(Arrays.asList(featureExtractors)));
        experimenter.printDaLabelingResult(corpus, result.getDaLabelResults());
    }

}
