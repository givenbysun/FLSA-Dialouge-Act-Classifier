package edu.uic.cs.nlp.findtask.da.maxent;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.findtask.da.DaClassifierBase;
import edu.uic.cs.nlp.findtask.da.DaTrainingInstanceCreator;
import opennlp.model.MaxentModel;
import opennlp.tools.util.BeamSearch;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.UtteranceTurn;
import edu.uic.cs.nlp.dm.DataMiningUtil;
import edu.uic.cs.nlp.findtask.da.DaClassifier;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInferencer;
import edu.uic.cs.nlp.findtask.dialog.DialogGameInitiatingInferencer;

public abstract class MaxEntDaClassifier extends DaClassifierBase implements DaClassifier {

    private static final int DEFAULT_BEAM_SIZE = 5;

    protected MaxentModel model;

    protected MaxEntDaContextGenerator contextGen;

    protected int size;

    protected BeamSearch<UtteranceTurn> beam;

    private static DialogGameInferencer dialogGameInferencer = new DialogGameInitiatingInferencer();

    protected MaxEntDaClassifier(int beamSize, MaxentModel model, MaxEntDaContextGenerator contextGen) {
        this.size = beamSize;
        this.model = model;
        this.contextGen = contextGen;
        if (model != null) {
            this.beam = new DaBeamSearch(size, contextGen, model);
        }
    }

    protected MaxEntDaClassifier(MaxentModel model, MaxEntDaContextGenerator cg) {
        this(DEFAULT_BEAM_SIZE, model, cg);
    }

    public MaxEntDaClassifier(Collection<DialogTurnFeatureExtractor> featureExtractors) {
        this(DEFAULT_BEAM_SIZE, null, new MaxEntDaContextGenerator(featureExtractors, dialogGameInferencer));
    }

    public String[] extractTags(FindTaskSession session) {
        String[] tags = new String[session.getUtTurns().size()];
        List<UtteranceTurn> turns = session.getUtTurns();
        for (int i = 0; i < turns.size(); i++) {
            tags[i] = turns.get(i).getDa();
        }

        return tags;

    }

    @Override
    public Collection<DialogTurnFeatureExtractor> getFeatureExtractors() {
        return this.contextGen.getFeatureExtractors();
    }

    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions) {
        this.trainClassifier(sessions, null);
    }




    @Override
    public void trainClassifier(Collection<FindTaskSession> sessions, Actor actor) {
        try {
            this.model = DataMiningUtil.train(
                    DaTrainingInstanceCreator.createDaTrainingEventStream(sessions, this.contextGen.getFeatureExtractors(), dialogGameInferencer, actor), 200, 2);
            this.beam = new DaBeamSearch(size, contextGen, model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private class DaBeamSearch extends BeamSearch<UtteranceTurn> {
        DaBeamSearch(int size, MaxEntDaContextGenerator cg, MaxentModel model) {
            super(size, cg, model);
        }

        DaBeamSearch(int size, MaxEntDaContextGenerator cg, MaxentModel model, int cacheSize) {
            super(size, cg, model, cacheSize);
        }
    }

}
