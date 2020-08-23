package edu.uic.cs.nlp.findtask.da;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.FindCorpusReader;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.BaseExperimenter;
import edu.uic.cs.nlp.dm.classifier.SessionTurnId;
import edu.uic.cs.nlp.dm.classifier.TurnClassificationResult;
import edu.uic.cs.nlp.dm.classifier.TurnLabelItem;
import edu.uic.cs.nlp.findtask.da.DaClassifierFactory.DaClassifierType;
import edu.uic.cs.nlp.findtask.da.DaClassifierFactory.DaExperimentType;

/**
 * @author Lin
 */
public class DaExperimenter extends BaseExperimenter {

    static Logger logger = LoggerFactory.getLogger(DaExperimenter.class);


    static CommandLine parseOptions(String[] args) throws ParseException {
        Options options = new Options();

        options.addOption("s", "setting", true, "Classifier algorithm");

        CommandLineParser parser = new PosixParser();
        return parser.parse(options, args);
    }


    public void printDaLabelingResult(List<FindTaskSession> corpus, Collection<TurnLabelItem> labellingResult) {
        Map<String, FindTaskSession> corpusMap = new HashMap<String, FindTaskSession>();
        for (FindTaskSession session : corpus) {
            corpusMap.put(session.getFindTaskSessionId(), session);
        }

        for (TurnLabelItem result : labellingResult) {
            if (!result.getLabel().equals(result.getClassified())) {
                UtteranceTurn turn = this.getUtteranceTurn(corpusMap, result.getSessionTurnId());
                if (turn == null) {
                    throw new IllegalStateException(result.getSessionTurnId() + " returns null for UtteranceTurn");
                }

                logger.debug("Turn: [{}] with DA: [{}] was classified as: [{}]", new String[]{turn.getUtterance(),
                        turn.getDa(), result.getClassified()});
            }
        }

    }

    UtteranceTurn getUtteranceTurn(Map<String, FindTaskSession> corpus, SessionTurnId stId) {
        FindTaskSession session = corpus.get(stId.getSessionId());
        return session.getUtTurn(stId.getTurnIndex());
    }

    void runExperiments(DaExperimentSetting expSetting, List<FindTaskSession> corpus) {
        int classifierIndex = 0;
        int classifierSize = expSetting.getClassifiers().size();

        for (DaClassifierType classifierType : expSetting.getClassifiers()) {
            logger.info("Running {} out of {} classifier experiments", ++classifierIndex, classifierSize);
            runMultiFeatureSetExperiments(corpus, classifierType, expSetting.getExperiment(),
                    expSetting.getFeatureSets());
            logger.info("Running {} out of {} classifier experiments is done", ++classifierIndex, classifierSize);
        }

    }

    /**
     * Given the generated FeatureExtractor combinations, run a set of experiments
     *
     * @return
     */
    public List<TurnClassificationResult> runMultiFeatureSetExperiments(List<FindTaskSession> corpus,
                                                                        DaClassifierType classifierType,
                                                                        DaExperimentType experimentType,
                                                                        Collection<DialogTurnFeatureExtractor[]> dtfeSets) {
        List<TurnClassificationResult> experimentResults = new ArrayList<TurnClassificationResult>();

        logger.info("Running experiments with {} feature extractors using classifier: {} with {} setting",
                new Object[]{dtfeSets.size(), classifierType, experimentType});

        int featureIndex = 1;
        String prevResultFileName = null;
        int expCount = dtfeSets.size();
        for (DialogTurnFeatureExtractor[] dtfeSet : dtfeSets) {
            logger.info("Running {} out of {} sets experiments", featureIndex, expCount);

            experimentResults.add(this.runNFoldExperiments(corpus, classifierType, experimentType, dtfeSet));

            logger.info("Finished Running {} out of {} sets experiments", featureIndex, expCount);

            String fileName = classifierType + "_" + experimentType + "_" + new Date().getTime() + ".csv";
            saveResultsToCsv(experimentResults, Arrays.asList(new String[]{Actor.ELD.toString(), Actor.HEL.toString()}), fileName);

            // delete the previous file
            if (prevResultFileName != null) {
                FileUtils.deleteQuietly(new File(prevResultFileName));
            }

            prevResultFileName = fileName;
        }

        return experimentResults;
    }

    public TurnClassificationResult runNFoldExperiments(List<FindTaskSession> corpus, DaClassifierType classifierType,
                                                        DaExperimentType experimentType, DialogTurnFeatureExtractor[] dtfeSet) {

        return runNFoldExperiments(corpus, DaClassifierFactory.createDaClassifier(classifierType, experimentType, Arrays.asList(dtfeSet)));
    }

    public TurnClassificationResult runNFoldExperiments(List<FindTaskSession> corpus, DaClassifier daClassifier) {
        NFoldSessionClassificationExperimenter experimenter = new NFoldSessionClassificationExperimenter(
                new DaOneFoldExperimenter(daClassifier));

        return experimenter.nFoldTesting(corpus, daClassifier.getFindTaskSessionFilter(), 10);
    }

    static void runExperiments(String[] args) {
        CommandLine cmd = null;
        try {
            cmd = parseOptions(args);
        } catch (ParseException e) {
            e.printStackTrace();
            logger.error("Command line parsing error, system will exit now");
            System.exit(-1);
        }

        String experimentSettingPath = "Find-Dialoguer/da-experiments-sl-crf-offline.json";

        if (cmd.hasOption('s')) {
            String cmdOptionValue = cmd.getOptionValue('s');
            if (StringUtils.isBlank(cmdOptionValue)) {
                logger.error("experiment setting value is empty, using default: {}", experimentSettingPath);
            } else {
                experimentSettingPath = cmdOptionValue.trim();
            }
        }

        DaExperimentSetting experimentSetting = null;

        try {
            experimentSetting = DaExperimentSetting.readConfiguration(new File(experimentSettingPath));
        } catch (Exception e) {
            e.printStackTrace();
            logger.error("Experiment setting file {} reading error, system now exits", experimentSettingPath);
            System.exit(-1);
        }

        if (!DaExperimentSetting.validateExperimentSetting(experimentSetting)) {
            logger.error("Experiment setting is NOT valid, system now exits");
            System.exit(-1);
        }

        List<FindTaskSession> corpus = null;

        try {
            logger.info("Reading corpus");
            corpus = new FindCorpusReader(experimentSetting.getCorpus()).readFindTaskCorpus();
            logger.info("Reading corpus done, {} sessions have been read out", corpus.size());
        } catch (Exception e) {
            logger.error("Corpus reading error, system now exits");

            System.exit(-1);
        }

        new DaExperimenter().runExperiments(experimentSetting, corpus);

    }

    public static void main(String[] args) {
        runExperiments(args);
    }

}
