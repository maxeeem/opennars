package org.opennars.control;

import org.opennars.entity.BudgetValue;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.entity.Sentence;
import org.opennars.entity.Stamp;
import org.opennars.entity.Task;
import org.opennars.entity.TruthValue;
import org.opennars.io.Symbols;
import org.opennars.language.Similarity;
import org.opennars.language.Tense;
import org.opennars.language.Term;
import org.opennars.main.Nar;
import org.opennars.main.Parameters;
import org.opennars.storage.Memory;

public final class VectorInference {

    private static final double VECTOR_BRIDGE_SIMILARITY_THRESHOLD = 0.8;

    private VectorInference() {
    }

    public static boolean isEnabled() {
        return Boolean.getBoolean("opennars.vector") || Boolean.getBoolean("opennars.vectorContext");
    }

    public static boolean isSelectionEnabled() {
        if (Boolean.getBoolean("opennars.vectorConceptSelection")) {
            return true;
        }
        return Boolean.getBoolean("opennars.vector");
    }

    private static boolean isBridgeEnabled() {
        if (Boolean.getBoolean("opennars.vectorBridgeInjection")) {
            return true;
        }
        return Boolean.getBoolean("opennars.vector");
    }

    public static void updateContext(final Memory mem, final Concept current) {
        // Context steering should only happen when explicitly enabled.
        if (!Boolean.getBoolean("opennars.vectorContext") || !isEnabled()) {
            return;
        }

        if (mem == null || current == null) {
            return;
        }

        // Real-context guard: do not let deterministic random placeholder vectors steer attention.
        // Only concepts with embeddings loaded from the embedding map are allowed to update context.
        if (!current.hasUserVector || current.vector == null) {
            return;
        }

        // Only nudge when the previous context also came from a real embedding.
        if (mem.lastContextTerm != null) {
            final Concept previousContextConcept = mem.concept(mem.lastContextTerm);
            if (previousContextConcept == null || !previousContextConcept.hasUserVector) {
                // Treat the previous context as invalid/non-semantic.
                mem.lastContextVector = null;
                mem.lastContextTerm = null;
            }
        }

        if (mem.lastContextVector != null && current.vector != null) {
            current.vector.nudge(mem.lastContextVector, 0.05);
        }

        mem.lastContextVector = current.vector;
        mem.lastContextTerm = current.getTerm();
    }

    public static void processBridge(
            final Memory mem,
            final Parameters narParameters,
            final Nar nar,
            final Concept current,
            final Hypervector contextVec,
            final Term contextTerm,
            final Concept contextConcept) {

        if (!isEnabled() || !isBridgeEnabled()) {
            return;
        }

        // No-nonsense bridge guard:
        // When a concept is actively involved in Goal/Quest processing (desires/quests pending),
        // do not inject fuzzy similarity associations that can dilute procedural confidence.
        try {
            if (current != null
                    && ((current.desires != null && !current.desires.isEmpty())
                    || (current.quests != null && !current.quests.isEmpty()))) {
                return;
            }
        } catch (Exception ignored) {
        }

        if (contextVec == null
                || contextTerm == null
                || contextConcept == null
                || current == null
                || current.getTerm() == null
                || current.getTerm().equals(contextTerm)
                || current.vector == null
                || contextConcept.vector == null) {
            return;
        }

        final double sim = current.vector.similarity(contextConcept.vector);
        if (sim <= VECTOR_BRIDGE_SIMILARITY_THRESHOLD) {
            return;
        }

        try {
            final Term similarityTerm = Similarity.make(current.getTerm(), contextTerm);
            if (similarityTerm == null) {
                return;
            }

            // Integrity fix: don't re-inject static vector associations as new evidence.
            // Similarity statements are stored as beliefs under the *concept of the statement term itself*.
            if (alreadyBelieves(mem, current, similarityTerm)) {
                return;
            }

            final BudgetValue budget = new BudgetValue(1.0f, 0.9f, 1.0f, nar.narParameters);
            final TruthValue truth = new TruthValue(1.0f, sim * 0.9, nar.narParameters);
            final Stamp stamp = new Stamp(nar, mem, Tense.Present);

            final Sentence<Term> bridge = new Sentence<>(similarityTerm, Symbols.JUDGMENT_MARK, truth, stamp);
            final Task<Term> bridgeTask = new Task<>(bridge, budget, Task.EnumType.INPUT);
            mem.localInference(bridgeTask, narParameters, nar);
        } catch (Exception ignored) {
            // Term construction / localInference failures should not interrupt the main loop.
        }
    }

    private static boolean alreadyBelieves(final Memory mem, final Concept current, final Term similarityTerm) {
        if (similarityTerm == null) {
            return false;
        }

        // Primary check: has the similarity statement already been accepted as a belief?
        // (It will live in the concept for the similarity statement itself.)
        if (mem != null) {
            final Concept similarityConcept = mem.concept(similarityTerm);
            if (similarityConcept != null && similarityConcept.beliefs != null) {
                try {
                    for (final Task<?> beliefTask : similarityConcept.beliefs) {
                        if (beliefTask == null || beliefTask.sentence == null || beliefTask.sentence.term == null) {
                            continue;
                        }
                        if (similarityTerm.equals(beliefTask.sentence.term)) {
                            return true;
                        }
                    }
                } catch (Exception ignored) {
                }
            }
        }

        // Fallback: if something stored it directly on the current concept, detect that too.
        if (current != null && current.beliefs != null) {
            try {
                for (final Task<?> beliefTask : current.beliefs) {
                    if (beliefTask == null || beliefTask.sentence == null || beliefTask.sentence.term == null) {
                        continue;
                    }
                    if (similarityTerm.equals(beliefTask.sentence.term)) {
                        return true;
                    }
                }
            } catch (Exception ignored) {
            }
        }
        return false;
    }
}
