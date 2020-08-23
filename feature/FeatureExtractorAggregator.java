package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class FeatureExtractorAggregator implements DialogTurnFeatureExtractor {

	private String name;
	private DialogTurnFeatureExtractor[] daFeatureExtractors;

	public FeatureExtractorAggregator(String name, DialogTurnFeatureExtractor[] daFeatureExtractors) {
		this.name = name;
		this.daFeatureExtractors = daFeatureExtractors;
	}

	public String getName() {
		return this.name;
	}

	@Override
	public Contexts extractContexts(FindTaskSession session, int utteranceTurnIndex, Object[] additionalContexts) {
		Contexts contexts = new Contexts();

		for (DialogTurnFeatureExtractor daFeatureExtractor : this.daFeatureExtractors) {
			contexts.addContexts(daFeatureExtractor.extractContexts(session, utteranceTurnIndex, additionalContexts));
		}

		return contexts;
	}

	@Override
	public String toString() {
		return this.getName();
	}

}
