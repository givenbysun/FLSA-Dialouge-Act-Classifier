package edu.uic.cs.nlp.findtask.da;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringWriter;
import java.util.Collection;
import java.util.HashSet;

import javax.xml.bind.annotation.XmlTransient;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.uic.cs.nlp.findtask.da.DaClassifierFactory.DaClassifierType;
import edu.uic.cs.nlp.findtask.da.DaClassifierFactory.DaExperimentType;
import edu.uic.cs.nlp.util.JsonUtil;

public class DaExperimentSetting {
    static Logger logger = LoggerFactory.getLogger(DaExperimentSetting.class);

    private Collection<DaClassifierType> classifiers;
    private DaExperimentType experiment;
    private String corpus;

    private Collection<String[]> featureNameSets; // dummy variable, used to load from config file

    private Collection<DialogTurnFeatureExtractor[]> featureSets;

    public Collection<DaClassifierType> getClassifiers() {
        return this.classifiers;
    }

    public DaExperimentType getExperiment() {
        return this.experiment;
    }

    public String getCorpus() {
        return this.corpus;
    }

    @XmlTransient
    public Collection<DialogTurnFeatureExtractor[]> getFeatureSets() {
        return this.featureSets;
    }

    public Collection<String[]> getFeatureNameSets() {
        return this.featureNameSets;
    }

    void setFeatureNameSets(Collection<String[]> featureNameSets) {
        this.featureNameSets = featureNameSets;
    }

    void setClassifiers(Collection<DaClassifierType> classifiers) {
        this.classifiers = classifiers;
    }

    void setExperiment(DaExperimentType experiment) {
        this.experiment = experiment;
    }

    void setCorpus(String corpus) {
        this.corpus = corpus;
    }

    void setFeatureSets(Collection<DialogTurnFeatureExtractor[]> featureSets) {
        this.featureSets = featureSets;
    }

    public static boolean validateExperimentSetting(DaExperimentSetting expSetting) {
        return StringUtils.isNotBlank(expSetting.corpus) && expSetting.experiment != null &&
                expSetting.classifiers != null && expSetting.classifiers.size() > 0 &&
                expSetting.featureSets != null && expSetting.featureSets.size() > 0;

    }

    public static DaExperimentSetting readConfiguration(File configFile) throws IOException {

        return readConfiguration(FileUtils.readFileToString(configFile));
    }

    public static DaExperimentSetting readConfiguration(InputStream configFile) throws IOException {

        StringWriter writer = new StringWriter();
        IOUtils.copy(configFile, writer);
        String fileContent = writer.toString();

        return readConfiguration(fileContent);
    }

    static DaExperimentSetting readConfiguration(String configContent) throws IOException {
        DaExperimentSetting expSetting = JsonUtil.fromJson(configContent, DaExperimentSetting.class);

        Collection<DialogTurnFeatureExtractor[]> dtfeSets = new HashSet<DialogTurnFeatureExtractor[]>();
        for (String[] dtfeNames : expSetting.getFeatureNameSets()) {
            dtfeSets.add(DaFeatureExtractorFactory.getDialogTurnFeatureExtractors(dtfeNames));
        }

        expSetting.featureSets = dtfeSets;

        return expSetting;
    }
}
