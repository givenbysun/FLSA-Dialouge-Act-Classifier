package edu.uic.cs.nlp.findtask.da;

import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.findtask.da.maxent.nonbeam.MaxEntNonBeamOfflineActorDaClassifier;
import edu.uic.cs.nlp.findtask.da.maxent.nonbeam.MaxEntNonBeamOfflineDaClassifier;
import edu.uic.cs.nlp.findtask.da.maxent.nonbeam.MaxEntNonBeamOnlineDaClassifier;
import edu.uic.cs.nlp.findtask.da.weka.WekaOfflineActorDaClassifier;
import org.apache.commons.lang3.StringUtils;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import edu.uic.cs.nlp.findtask.da.mallet.MalletCrfOfflineDaClassifier;
import edu.uic.cs.nlp.findtask.da.mallet.MalletCrfOnlineDaClassifier;
import edu.uic.cs.nlp.findtask.da.maxent.MaxEntOfflineDaClassifier;
import edu.uic.cs.nlp.findtask.da.maxent.MaxEntOnlineDaClassifier;
import edu.uic.cs.nlp.findtask.da.weka.WekaOfflineDaClassifier;
import edu.uic.cs.nlp.findtask.da.weka.WekaOnlineDaClassifier;

public class DaClassifierFactory {
    public static enum DaClassifierType {
        CRF,
        MAXENT,
        DT,
        NB,
        SVM,
        MAXENT_NONBEAM,      //maxent without sequence labeling.
        MAXENT_HEL,
        MAXENT_ELD,
        DT_HEL,
        DT_ELD,
        NB_HEL,
        NB_ELD;

        static Map<String, DaClassifierType> mapping = new HashMap<String, DaClassifierType>();

        static {
            Set<DaClassifierType> allTypes = EnumSet.allOf(DaClassifierType.class);
            for (DaClassifierType daClassifierType : allTypes) {
                mapping.put(daClassifierType.toString(), daClassifierType);
            }
        }

        static DaClassifierType from(String typeName) {
            if (StringUtils.isBlank(typeName) || !mapping.containsKey(typeName.toUpperCase())) {
                return null;
            }

            return mapping.get(typeName.toUpperCase());
        }

    }

    /**
     * Online or offline.
     */
    public static enum DaExperimentType {
        ONLINE,
        OFFLINE;

        static Map<String, DaExperimentType> mapping = new HashMap<String, DaExperimentType>();

        static {
            Set<DaExperimentType> allTypes = EnumSet.allOf(DaExperimentType.class);
            for (DaExperimentType daExperimentType : allTypes) {
                mapping.put(daExperimentType.toString(), daExperimentType);
            }
        }

        static DaExperimentType from(String typeName) {
            if (StringUtils.isBlank(typeName) || !mapping.containsKey(typeName.toUpperCase())) {
                return null;
            }

            return mapping.get(typeName.toUpperCase());
        }
    }

    public static DaClassifier createDaClassifier(DaClassifierType classifierType, DaExperimentType expType,
                                                  Collection<DialogTurnFeatureExtractor> dtFeatureExtractors) {
        switch (expType) {
            case ONLINE:
                switch (classifierType) {
                    case CRF:
                        return new MalletCrfOnlineDaClassifier(dtFeatureExtractors);
                    case MAXENT:
                        return new MaxEntOnlineDaClassifier(dtFeatureExtractors);
                    case DT:
                        return new WekaOnlineDaClassifier(J48.class, dtFeatureExtractors);
                    case NB:
                        return new WekaOnlineDaClassifier(NaiveBayes.class, dtFeatureExtractors);
                    case SVM:
                        return new WekaOnlineDaClassifier(LibSVM.class, dtFeatureExtractors);
                    case MAXENT_NONBEAM:
                        return new MaxEntNonBeamOnlineDaClassifier(dtFeatureExtractors);
                }
                break;
            case OFFLINE:
                switch (classifierType) {
                    case CRF:
                        return new MalletCrfOfflineDaClassifier(dtFeatureExtractors);
                    case MAXENT:
                        return new MaxEntOfflineDaClassifier(dtFeatureExtractors);
                    case DT:
                        return new WekaOfflineDaClassifier(J48.class, dtFeatureExtractors);
                    case NB:
                        return new WekaOfflineDaClassifier(NaiveBayes.class, dtFeatureExtractors);
                    case SVM:
                        return new WekaOfflineDaClassifier(LibSVM.class, dtFeatureExtractors);
                    case MAXENT_NONBEAM:
                        return new MaxEntNonBeamOfflineDaClassifier(dtFeatureExtractors);
                    case MAXENT_ELD:
                        return new MaxEntNonBeamOfflineActorDaClassifier(dtFeatureExtractors, Actor.ELD);
                    case MAXENT_HEL:
                        return new MaxEntNonBeamOfflineActorDaClassifier(dtFeatureExtractors, Actor.HEL);
                    case DT_HEL:
                        return new WekaOfflineActorDaClassifier(J48.class, dtFeatureExtractors, Actor.HEL);
                    case DT_ELD:
                        return new WekaOfflineActorDaClassifier(J48.class, dtFeatureExtractors, Actor.ELD);
                    case NB_HEL:
                        return new WekaOfflineActorDaClassifier(NaiveBayes.class, dtFeatureExtractors, Actor.HEL);
                    case NB_ELD:
                        return new WekaOfflineActorDaClassifier(NaiveBayes.class, dtFeatureExtractors, Actor.ELD);
                }
                break;
            default:
                return null;
        }
        return null;
    }

}
