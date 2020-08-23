package edu.uic.cs.nlp.findtask.da.feature.hpa;

import java.util.HashSet;
import java.util.Set;

import edu.uic.cs.nlp.anvil.eah.HoActionTurn;

public class HapticAlternativeUtil {
	
	public static final String NON_RECOGNIZED_ENCODE = "NON-RECOG";
	public static Set<HoActionTurn.Action>  RECOGNIZED_HO_ACTIONS = new HashSet<HoActionTurn.Action>();
	static {
		RECOGNIZED_HO_ACTIONS.add(HoActionTurn.Action.OPEN);
		RECOGNIZED_HO_ACTIONS.add(HoActionTurn.Action.CLOSE);
		RECOGNIZED_HO_ACTIONS.add(HoActionTurn.Action.HOLD);
	}
	
	public static boolean canRecognized(HoActionTurn.Action action) {
		return RECOGNIZED_HO_ACTIONS.contains(action);
	}
	
	public static String getEncodedAction(HoActionTurn.Action action) {
		return canRecognized(action) ? action.toString() : NON_RECOGNIZED_ENCODE;
	}
	
	

}
