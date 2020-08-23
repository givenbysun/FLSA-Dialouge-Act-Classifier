package edu.uic.cs.nlp.findtask.da;

import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.HasName;

public interface DialogTurnFeatureExtractor extends HasName {

	Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts);

}
