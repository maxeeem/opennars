/* 
 * The MIT License
 *
 * Copyright 2018 The OpenNARS authors.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.opennars.control;

import org.opennars.control.concept.ProcessAnticipation;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.entity.Task;
import org.opennars.entity.TermLink;
import org.opennars.inference.BudgetFunctions;
import org.opennars.inference.RuleTables;
import org.opennars.io.events.Events;
import org.opennars.language.Term;
import org.opennars.main.Nar;
import org.opennars.main.Parameters;
import org.opennars.storage.Memory;

/**
 * Concept reasoning context
 *
 * a concept is "fired" or activated by applying the reasoner
 *
 * @author Patrick Hammer
 *
 */
public class GeneralInferenceControl {
    
    public static void selectConceptForInference(final Memory mem, final Parameters narParameters, final Nar nar) {
        final Concept currentConcept;
        final Hypervector contextVec;
        final Term contextTerm;
        final Concept contextConcept;
        final boolean globalGoalOrQuestMode;
        synchronized (mem.concepts) { //modify concept bag
            if (VectorInference.isEnabled()) {
                contextVec = mem.lastContextVector;
                contextTerm = mem.lastContextTerm;
                Concept selected = mem.concepts.takeWithContext(contextVec);
                // If we keep re-selecting the context term itself, try once to pull a different
                // related concept, so analogy/synonym bridging can happen.
                if (selected != null && contextTerm != null && contextTerm.equals(selected.getTerm())) {
                    final Concept alternate = mem.concepts.takeWithContext(contextVec);
                    mem.concepts.putIn(selected);
                    selected = (alternate != null) ? alternate : selected;
                }
                currentConcept = selected;
                contextConcept = (contextTerm != null) ? mem.concept(contextTerm) : null;
            } else {
                contextVec = null;
                contextTerm = null;
                contextConcept = null;
                currentConcept = mem.concepts.takeOut();
            }
            if (currentConcept==null) {
                return;
            }

            boolean goalMode = false;
            // If ANY goal/quest is currently pending anywhere, suppress bridge injection
            // to keep procedural goal confidence from being diluted.
            for (final Concept c : mem.concepts) {
                if (c == null) {
                    continue;
                }
                if ((c.desires != null && !c.desires.isEmpty()) || (c.quests != null && !c.quests.isEmpty())) {
                    goalMode = true;
                    break;
                }
            }
            // currentConcept is taken out of the bag above; include it in the check too.
            if (!goalMode
                    && ((currentConcept.desires != null && !currentConcept.desires.isEmpty())
                    || (currentConcept.quests != null && !currentConcept.quests.isEmpty()))) {
                goalMode = true;
            }
            globalGoalOrQuestMode = goalMode;
        }

        final DerivationContext nal = new DerivationContext(mem, narParameters, nar);
        boolean putBackConcept = false;
        float forgetCycles = 0.0f;
        synchronized(currentConcept) { //use current concept (current concept is the resource)  
            // VectorNARS: inject bridge associations at most once per concept-cycle.
            // Suppress them entirely when any goal/quest is pending (goal-focused mode).
            if (!globalGoalOrQuestMode) {
                VectorInference.processBridge(mem, narParameters, nar, currentConcept, contextVec, contextTerm, contextConcept);
            }
            ProcessAnticipation.maintainDisappointedAnticipations(narParameters, currentConcept, nar);
            if(currentConcept.taskLinks.size() == 0) { //remove concepts without tasklinks and without termlinks
                mem.concepts.pickOut(currentConcept.getTerm());
                mem.conceptRemoved(currentConcept);
                return;
            }
            if(currentConcept.termLinks.size() == 0) {  //remove concepts without tasklinks and without termlinks
                mem.concepts.pickOut(currentConcept.getTerm());
                mem.conceptRemoved(currentConcept);
                return;
            }

            VectorInference.updateContext(mem, currentConcept);

            nal.setCurrentConcept(currentConcept);
            putBackConcept = fireConcept(nal, 1, contextVec, contextTerm, contextConcept, nar, globalGoalOrQuestMode);
            if(putBackConcept) {
                forgetCycles = nal.memory.cycles(nal.memory.narParameters.CONCEPT_FORGET_DURATIONS);
                if(nal.memory.emotion != null) {
                    nal.currentConcept.setQuality(BudgetFunctions.or(nal.currentConcept.getQuality(),nal.memory.emotion.happy()));
                }
            }
        }
        if(putBackConcept) { // put back into bag (bag is the resource)
            synchronized (nal.memory.concepts) {
                nal.memory.concepts.putBack(nal.currentConcept, forgetCycles, nal.memory);
            }
        }
    }

    // /return true if concept must be put back
    public static boolean fireConcept(
            final DerivationContext nal,
            final int numTaskLinks,
            final Hypervector contextVec,
            final Term contextTerm,
            final Concept contextConcept,
            final Nar nar,
            final boolean globalGoalOrQuestMode) {
        for (int i = 0; i < numTaskLinks; i++) {
            if (nal.currentConcept.taskLinks.size() == 0) {
                return false;
            }
            nal.currentTaskLink = nal.currentConcept.taskLinks.takeOut();                    
            if (nal.currentTaskLink == null) {
                return false;
            }
            if (nal.currentTaskLink.budget.aboveThreshold()) {
                fireTaskLink(nal, nal.memory.narParameters.TERMLINK_MAX_REASONED, contextVec, contextTerm, contextConcept, nar, globalGoalOrQuestMode);
            }
            nal.currentConcept.taskLinks.putBack(nal.currentTaskLink, nal.memory.cycles(nal.memory.narParameters.TASKLINK_FORGET_DURATIONS), nal.memory);
        }
        return true;
    }
    
    protected static void fireTaskLink(
            final DerivationContext nal,
            int termLinks,
            final Hypervector contextVec,
            final Term contextTerm,
            final Concept contextConcept,
            final Nar nar,
            final boolean globalGoalOrQuestMode) {
        @SuppressWarnings("rawtypes")
        final Task task = nal.currentTaskLink.getTarget();
        nal.setCurrentTerm(nal.currentConcept.term);
        nal.setCurrentTaskLink(nal.currentTaskLink);
        nal.setCurrentBeliefLink(null);
        nal.setCurrentTask(task); // one of the two places where this variable is set

        if(nal.memory.emotion != null) {
            nal.memory.emotion.adjustBusy(nal.currentTaskLink.getPriority(),nal.currentTaskLink.getDurability(),nal);
        }
        if (nal.currentTaskLink.type == TermLink.TRANSFORM) {
            nal.setCurrentBelief(null);
            //TermLink tasklink_as_termlink = new TermLink(nal.currentTaskLink.getTerm(), TermLink.TRANSFORM, nal.getCurrentTaskLink().index);
            //if(nal.currentTaskLink.novel(tasklink_as_termlink, nal.memory.time(), true)) { //then record yourself, but also here novelty counts
                RuleTables.transformTask(nal.currentTaskLink, nal); // to turn this into structural inference as below?
            //}
        } else {            
            while (termLinks > 0) {
                final TermLink termLink = nal.currentConcept.selectTermLink(nal.currentTaskLink, nal.time.time(), nal.narParameters);
                if (termLink == null) {
                    break;
                }
                fireTermlink(termLink, nal);
                nal.currentConcept.returnTermLink(termLink);
                termLinks--;
            }
        }
                
        nal.memory.emit(Events.ConceptFire.class, nal);
        //memory.logic.TASKLINK_FIRE.commit(currentTaskLink.budget.getPriority());
    }

    public static boolean fireTermlink(final TermLink termLink, final DerivationContext nal) {
        nal.setCurrentBeliefLink(termLink);
        RuleTables.reason(nal.currentTaskLink, termLink, nal);
        nal.memory.emit(Events.TermLinkSelect.class, termLink, nal.currentConcept, nal);                  
        return true;
    }
}
