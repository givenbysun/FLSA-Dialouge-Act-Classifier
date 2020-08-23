package edu.uic.cs.nlp.findtask.da.mallet;

import java.util.Collection;

import edu.uic.cs.nlp.findtask.da.DaClassifierBase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByLabelLikelihood;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Sequence;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.findtask.da.DaClassifier;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public abstract class MalletCrfDaClassifier extends DaClassifierBase implements DaClassifier {

	protected FeatureExtractorPipe featureExtractPipe;
	protected CRF crf;

	private Logger logger = LoggerFactory.getLogger(this.getClass());

	public MalletCrfDaClassifier(Collection<DialogTurnFeatureExtractor> daFeatureExtractors) {
		this.featureExtractPipe = new FeatureExtractorPipe(daFeatureExtractors);
		this.crf = null;
	}

	@Override
	public void trainClassifier(Collection<FindTaskSession> sessions) {
		logger.info("Training A Mallet CRF DA classifier");

		InstanceList instList = new InstanceList(this.featureExtractPipe);
		for (FindTaskSession session : sessions) {
			instList.addThruPipe(new Instance(session, "", "", ""));
		}

		logger.debug("InstanceList with {} Instances has been extracted:", instList.size());

		CRF crf = new CRF(instList.getDataAlphabet(),
				instList.getTargetAlphabet());

		crf.addFullyConnectedStatesForLabels();
		// initialize maxentModel's weights
		crf.setWeightsDimensionAsIn(instList, false);

		CRFTrainerByLabelLikelihood trainer = new CRFTrainerByLabelLikelihood(crf);
		trainer.setGaussianPriorVariance(10.0);

		trainer.train(instList);

		logger.info("A Mallet CRF DA classifier has been trained");

		this.crf = crf;

	}

	@Override
	public Collection<DialogTurnFeatureExtractor> getFeatureExtractors() {
		return this.featureExtractPipe.getDaFeatureExtractors();
	}

	protected String[] convertLabelSequenceToStrings(Sequence<String> labelSeq) {
		String[] labels = new String[labelSeq.size()];
		for (int i = 0; i < labelSeq.size(); i++) {
			labels[i] = labelSeq.get(i);
		}
		return labels;
	}

}
