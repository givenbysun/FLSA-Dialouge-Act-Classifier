package edu.uic.cs.nlp.findtask.da.mallet;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.findtask.da.feature.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Sequence;
import edu.uic.cs.nlp.anvil.eah.FindCorpusReader;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.TurnClassificationResult;
import edu.uic.cs.nlp.findtask.da.DaExperimenter;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.DialogGameRuntimeFeatureExtractor;

public class MalletCrfOnlineDaClassifier extends MalletCrfDaClassifier {

    Logger logger = LoggerFactory.getLogger(this.getClass());

    public MalletCrfOnlineDaClassifier(Collection<DialogTurnFeatureExtractor> daFeatureExtractors) {
        super(daFeatureExtractors);
    }

    @Override
    public String[] classifySession(FindTaskSession session) {
        if (this.crf == null) {
            throw new IllegalStateException("CRF has not been trained yet");
        }
        String[] classifiedLabels = new String[session.getUtTurnSize()];

        for (int i = 0; i < session.getUtTurnSize(); i++) {

            FeatureVectorSequence fvs = this.featureExtractPipe.extractOnlineFeatureVectorSequence(session, i,
                    classifiedLabels);

            @SuppressWarnings("unchecked")
            Sequence<String> classifiedSeq = this.crf.transduce(fvs);

            String[] turnClassifiedLabels = this.convertLabelSequenceToStrings(classifiedSeq);

            classifiedLabels[i] = turnClassifiedLabels[i];


        }

        return classifiedLabels;
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
        List<FindTaskSession> corpus = new FindCorpusReader("FindTaskCorpus")
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

                new DialogGameRuntimeFeatureExtractor()

        };

        TurnClassificationResult result = experimenter.runNFoldExperiments(corpus,
                new MalletCrfOnlineDaClassifier(Arrays.asList(featureExtractors)));
        experimenter.printDaLabelingResult(corpus, result.getDaLabelResults());
    }

}
