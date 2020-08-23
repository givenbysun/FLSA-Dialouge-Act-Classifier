package edu.uic.cs.nlp.findtask.da.feature;

import edu.uic.cs.nlp.anvil.eah.Actor;
import edu.uic.cs.nlp.anvil.eah.ActorSpan;
import edu.uic.cs.nlp.anvil.eah.ActorTurn;
import edu.uic.cs.nlp.anvil.eah.DialogTurn;
import edu.uic.cs.nlp.anvil.eah.FindTaskSession;
import edu.uic.cs.nlp.anvil.eah.LocationSpan;
import edu.uic.cs.nlp.dm.classifier.BooleanFeature;
import edu.uic.cs.nlp.dm.classifier.Contexts;
import edu.uic.cs.nlp.dm.classifier.StringFeature;
import edu.uic.cs.nlp.findtask.da.DialogTurnFeatureExtractor;

public class LocationFeatureExtractor implements DialogTurnFeatureExtractor {

	@Override
	public Contexts extractContexts(FindTaskSession session, int dTurnPosition, Object[] additionalContexts) {
		DialogTurn dTurn = session.getDialogTurnByPosition(dTurnPosition);

		return this.extractContexts(session, dTurn.getUtteranceTurn());
	}

	Contexts extractContexts(FindTaskSession session, ActorTurn turn) {
		Contexts contexts = new Contexts();

		String speakerLoc = this.getStartLocation(session, turn.getActor(), turn);

		Actor otherActor = turn.getActor() == Actor.ELD ? Actor.HEL : Actor.ELD;

		String otherLoc = this.getStartLocation(session, otherActor, turn);

		contexts.add(new StringFeature("#Loc", speakerLoc));
		contexts.add(new StringFeature("#OLoc", otherLoc));

		contexts.add(new BooleanFeature("#SameLoc", speakerLoc.equalsIgnoreCase(otherLoc)));

		contexts.add(new BooleanFeature("#ChangedLoc", this.isLocationChanged(session, turn)));

		return contexts;
	}

	protected String getStartLocation(FindTaskSession session, Actor actor, ActorSpan span) {
		for (LocationSpan loc : session.getLocations()) {
			if (loc.getActor() == actor || loc.getActor() == Actor.BOTH) {
				if (span.withinTimeSpan(loc)) {
					return loc.getLocation();
				} else if (span.startsWithin(loc)) {
					return loc.getLocation();
				}
			}
		}
		throw new IllegalStateException("No Location is availabe for " + session.getFindTaskSessionId());
	}

	protected String getEndLocation(FindTaskSession session, Actor actor, ActorSpan span) {
		for (LocationSpan loc : session.getLocations()) {
			if (loc.getActor() == actor || loc.getActor() == Actor.BOTH) {
				if (span.withinTimeSpan(loc)) {
					return loc.getLocation();
				} else if (span.endsWithin(loc)) {
					return loc.getLocation();
				}
			}
		}
		throw new IllegalStateException("No Location is availabe for " + session.getFindTaskSessionId()
				+ " actor span: " + span.getEnd());
	}

	protected boolean isLocationChanged(FindTaskSession session, ActorTurn turn) {

		String startLoc = this.getStartLocation(session, turn.getActor(), turn);
		String endLoc = this.getEndLocation(session, turn.getActor(), turn);

		return startLoc.equalsIgnoreCase(endLoc);
	}

	@Override
	public String getName() {
		return "LOC";
	}

}
