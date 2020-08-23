package edu.uic.cs.nlp.findtask.da;

import java.util.List;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.TurnLabelItem;

public interface OneFoldExperimenter {
	
	String getName();
	
	String getFeatureNames();
	
	void trainClassifier(List<FindTaskSession> sessions);
	
	List<TurnLabelItem> classifySessions(List<FindTaskSession> sessions);

}
