package edu.uic.cs.nlp.findtask.da;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.uic.cs.nlp.findtask.da.feature.*;
import org.apache.commons.lang3.StringUtils;

import edu.uic.cs.nlp.findtask.da.feature.DialogGameRuntimeFeatureExtractor;
import edu.uic.cs.nlp.findtask.da.feature.hpa.DialogHistoryAltFeatureExtractor;

public class DaFeatureExtractorFactory {

    public static final FeatureExtractorAggregator TX = new FeatureExtractorAggregator(
            "TX",
            new DialogTurnFeatureExtractor[]{
                    new UtteranceChunkExtractor(),
                    new UtteranceDependencyExtractor(),
                    new UtteranceHeuristicExtractor(),
                    new UtteranceSynaticExtractor(),
                    new UtteranceWordPosExtractor()
            });

    public static final FeatureExtractorAggregator GS = new FeatureExtractorAggregator(
            "GS",
            new DialogTurnFeatureExtractor[]{
                    new CurrentPointingExtractor()
            });

    public static final FeatureExtractorAggregator LO = new FeatureExtractorAggregator(
            "LO",
            new DialogTurnFeatureExtractor[]{
                    new LocationFeatureExtractor()
            });

    public  static final FeatureExtractorAggregator HO = new FeatureExtractorAggregator(
            "HO",
            new DialogTurnFeatureExtractor[]{
                    new CurrentHoActionExtractor()
            });

    /**
     * HO-Enhanced
     */
    public static final FeatureExtractorAggregator HOE = new FeatureExtractorAggregator(
            "HOE",
            new DialogTurnFeatureExtractor[]{
                    new CurrentHoActionExtractor(),
                    new PreviousHoActionExtractor()
            });

    public static final FeatureExtractorAggregator UT = new FeatureExtractorAggregator(
            "UT",
            new DialogTurnFeatureExtractor[]{
                    new TaskMetaExtractor(),
                    new UtteranceTurnFeatureExtractor(),

            });

    public static final FeatureExtractorAggregator DH = new FeatureExtractorAggregator(
            "DH",
            new DialogTurnFeatureExtractor[]{
                    new DialogHistoryFeatureExtractor()
            });

    /**
     * The DH feature without DA.
     */
    public static final FeatureExtractorAggregator DHO = new FeatureExtractorAggregator(
            "DHO",
            new DialogTurnFeatureExtractor[]{
                    new DialogHistoryFeatureExtractorNoDA()
            });

    public static final FeatureExtractorAggregator DGR = new FeatureExtractorAggregator("DGR",
            new DialogTurnFeatureExtractor[]{
                    new DialogGameRuntimeFeatureExtractor()
            });

    public static final FeatureExtractorAggregator DGA = new FeatureExtractorAggregator("DGA",
            new DialogTurnFeatureExtractor[]{
                    new DialogGameAnnotationFeatureExtractor()
            });




    /**
     * HO Enhanced feature:  Unrecognized HO actions are replaced with "NON-REC" tag
     */
    public static final FeatureExtractorAggregator HOEA = new FeatureExtractorAggregator(
            "HOEA",
            new DialogTurnFeatureExtractor[]{
                    new CurrentHoActionExtractor(),
                    new PreviousHoActionExtractor()
            });


    /**
     * Dialog History Feature: Unrecognized HO actions are replaced with "NON-REC" tag
     */
    public static final FeatureExtractorAggregator DHA = new FeatureExtractorAggregator(
            "DHA",
            new DialogTurnFeatureExtractor[]{
                    new DialogHistoryAltFeatureExtractor()
            });


    static final List<DialogTurnFeatureExtractor> allFeatureExtractors = Arrays
            .asList(new DialogTurnFeatureExtractor[]{
                    TX, GS, HO, UT, DH, DGA, DGR, LO, HOE, HOEA, DHA, DHO});

    static final Map<String, DialogTurnFeatureExtractor> dtfeMapping = new HashMap<String, DialogTurnFeatureExtractor>();

    static {
        for (DialogTurnFeatureExtractor dfe : allFeatureExtractors) {
            dtfeMapping.put(dfe.getName().toUpperCase(), dfe);
        }
    }

    public static Collection<DialogTurnFeatureExtractor> getAllFeatureExtractors() {
        return allFeatureExtractors;
    }

    /**
     * Get the FeatureExtractor by name
     *
     * @param dtfeName
     * @return
     */
    public static DialogTurnFeatureExtractor getDialogTurnFeatureExtractor(String dtfeName) {
        if (StringUtils.isBlank(dtfeName) || !dtfeMapping.containsKey(dtfeName.toUpperCase())) {
            return null;
        }

        return dtfeMapping.get(dtfeName.toUpperCase());
    }

    public static DialogTurnFeatureExtractor[] getDialogTurnFeatureExtractors(String[] dtfeNames) {
        return getDialogTurnFeatureExtractors(Arrays.asList(dtfeNames));
    }

    /**
     * Get the DTFEs by names
     *
     * @param dtfeNames
     * @return
     */
    public static DialogTurnFeatureExtractor[] getDialogTurnFeatureExtractors(Collection<String> dtfeNames) {

        if (dtfeNames == null || dtfeNames.size() == 0) {
            return new DialogTurnFeatureExtractor[0];
        }

        List<DialogTurnFeatureExtractor> dtfes = new ArrayList<DialogTurnFeatureExtractor>();

        for (String dtfeName : dtfeNames) {
            DialogTurnFeatureExtractor dtfe = getDialogTurnFeatureExtractor(dtfeName);
            if (dtfe != null) {
                dtfes.add(dtfe);
            } else {
                throw new IllegalStateException(dtfeName + " cannot be found");
            }
        }

        return dtfes.toArray(new DialogTurnFeatureExtractor[dtfes.size()]);
    }

}
