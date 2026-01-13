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

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public final class VectorInference {

    private static final AtomicInteger injectedCount = new AtomicInteger(0);
    private static final AtomicInteger skippedCount = new AtomicInteger(0);

    private VectorInference() {
    }

    public static boolean isEnabled() {
        return Boolean.getBoolean("opennars.vector");
    }

    public static void updateContext(final Memory mem, final Concept current) {
        // Vector features are gated behind a single flag.
        if (!isEnabled()) {
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
            final Concept contextConcept,
            final boolean anyGoalOrQuestExists) {

        if (!isEnabled()) {
            return;
        }

        if (narParameters == null || !narParameters.VECTOR_BRIDGE_ENABLED) {
            return;
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

        if (narParameters.VECTOR_BRIDGE_SKIP_SELF) {
            try {
                if (Term.isSelf(current.getTerm()) || Term.isSelf(contextTerm)) {
                    skippedCount.incrementAndGet();
                    if (narParameters.VECTOR_BRIDGE_LOG) {
                        System.out.println("[VectorBridge] skip reason=self term=" + current.getTerm() + " ctx=" + contextTerm);
                    }
                    return;
                }
            } catch (Exception ignored) {
            }
        }

        // Real-vector guard: avoid random placeholder vectors causing spammy bridges.
        if (narParameters.VECTOR_BRIDGE_REQUIRE_USER_VECTORS) {
            if (!current.hasUserVector || !contextConcept.hasUserVector) {
                skippedCount.incrementAndGet();
                if (narParameters.VECTOR_BRIDGE_LOG) {
                    System.out.println("[VectorBridge] skip reason=noUserVector cur=" + current.getTerm() + " ctx=" + contextTerm);
                }
                return;
            }
        }

        final double sim = current.vector.similarity(contextConcept.vector);
        if (sim <= narParameters.VECTOR_BRIDGE_SIMILARITY_THRESHOLD) {
            skippedCount.incrementAndGet();
            if (narParameters.VECTOR_BRIDGE_LOG) {
                System.out.println("[VectorBridge] skip reason=belowThreshold sim=" + sim + " thr=" + narParameters.VECTOR_BRIDGE_SIMILARITY_THRESHOLD);
            }
            return;
        }

        final String key = bridgeKey(current.getTerm(), contextTerm);
        final long now = (nar != null) ? nar.time() : System.currentTimeMillis();
        if (isRecentlyInjected(mem, key, now, narParameters.VECTOR_BRIDGE_COOLDOWN)) {
            skippedCount.incrementAndGet();
            if (narParameters.VECTOR_BRIDGE_LOG) {
                System.out.println("[VectorBridge] skip reason=cooldown key=" + key);
            }
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
                skippedCount.incrementAndGet();
                if (narParameters.VECTOR_BRIDGE_LOG) {
                    System.out.println("[VectorBridge] skip reason=alreadyBelieves term=" + similarityTerm);
                }
                return;
            }

            float prio = 1.0f;
            float dura = 0.9f;
            float qual = 1.0f;
            if (anyGoalOrQuestExists) {
                final float factor = narParameters.VECTOR_BRIDGE_THROTTLE_FACTOR_WHEN_GOAL_OR_QUEST;
                prio = clamp01(prio * factor);
                dura = clamp01(dura * factor);
                if (narParameters.VECTOR_BRIDGE_LOG) {
                    System.out.println("[VectorBridge] throttle factor=" + factor + " (goal/quest active)");
                }
            }

            final BudgetValue budget = new BudgetValue(prio, dura, qual, nar.narParameters);
            final TruthValue truth = new TruthValue(1.0f, sim * 0.9, nar.narParameters);
            final Stamp stamp = new Stamp(nar, mem, Tense.Present);

            final Sentence<Term> bridge = new Sentence<>(similarityTerm, Symbols.JUDGMENT_MARK, truth, stamp);
            final Task<Term> bridgeTask = new Task<>(bridge, budget, Task.EnumType.INPUT);
            mem.localInference(bridgeTask, narParameters, nar);

            int n = injectedCount.incrementAndGet();
            if (narParameters.VECTOR_BRIDGE_LOG || n % 100 == 0) {
                System.out.println("[VectorBridgeSummary] injected=" + n + " skipped=" + skippedCount.get() + " (at inject)");
            }
            
            markInjected(mem, key, now, narParameters.VECTOR_BRIDGE_RECENT_MAX);
            if (narParameters.VECTOR_BRIDGE_LOG) {
                System.out.println("[VectorBridge] inject sim=" + sim + " task=" + similarityTerm + " prio=" + prio + " dura=" + dura);
            }
        } catch (Exception e) {
            // Term construction / localInference failures should not interrupt the main loop.
            if (narParameters.VECTOR_BRIDGE_LOG) {
                System.out.println("[VectorBridge] skip reason=exception type=" + e.getClass().getSimpleName() + " msg=" + e.getMessage());
            }
        }
    }

    private static float clamp01(final float v) {
        if (v < 0.0f) {
            return 0.0f;
        }
        if (v > 1.0f) {
            return 1.0f;
        }
        return v;
    }

    private static String bridgeKey(final Term a, final Term b) {
        final String sa = (a != null) ? a.toString() : "";
        final String sb = (b != null) ? b.toString() : "";
        if (sa.compareTo(sb) <= 0) {
            return sa + "||" + sb;
        }
        return sb + "||" + sa;
    }

    private static boolean isRecentlyInjected(final Memory mem, final String key, final long now, final long cooldown) {
        if (mem == null || key == null || key.isEmpty() || cooldown <= 0) {
            return false;
        }
        try {
            final Long last = mem.vectorBridgeLastInjected.get(key);
            return last != null && (now - last) < cooldown;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static void markInjected(final Memory mem, final String key, final long now, final int maxEntries) {
        if (mem == null || key == null || key.isEmpty()) {
            return;
        }
        try {
            // Keep eviction roughly LRU-by-injection (LinkedHashMap is insertion-order by default).
            // Updating an existing key doesn't move it to the end, so we remove+put.
            if (mem.vectorBridgeLastInjected.containsKey(key)) {
                mem.vectorBridgeLastInjected.remove(key);
            }
            mem.vectorBridgeLastInjected.put(key, now);
            final int max = (maxEntries > 0) ? maxEntries : 0;
            if (max > 0 && mem.vectorBridgeLastInjected.size() > max) {
                final Iterator<Map.Entry<String, Long>> it = mem.vectorBridgeLastInjected.entrySet().iterator();
                while (mem.vectorBridgeLastInjected.size() > max && it.hasNext()) {
                    it.next();
                    it.remove();
                }
            }
        } catch (Exception ignored) {
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
