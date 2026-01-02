package org.opennars.core;

import org.junit.Test;

import static org.junit.Assert.*;

import java.lang.reflect.Field;

import org.opennars.entity.BudgetValue;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.language.Term;
import org.opennars.main.Nar;
import org.opennars.storage.Bag;

public class VectorNarsTest {

    private static long[] bitsOf(final Hypervector v) {
        try {
            final Field f = Hypervector.class.getDeclaredField("bits");
            f.setAccessible(true);
            return (long[]) f.get(v);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("Failed to reflect Hypervector.bits", e);
        }
    }

    private static Hypervector complementOf(final Hypervector src) {
        final Hypervector dst = new Hypervector();
        final long[] srcBits = bitsOf(src);
        final long[] dstBits = bitsOf(dst);
        assertEquals("Hypervector word length must match", srcBits.length, dstBits.length);
        for (int i = 0; i < srcBits.length; i++) {
            dstBits[i] = ~srcBits[i];
        }
        return dst;
    }

    @Test
    public void testVectorOverridesPriority() throws Exception {
        System.setProperty("opennars.vectorContext", "true");

        final Nar nar = new Nar();

        // Use a deterministic tiny bag so baseline selection is not probabilistic.
        // Levels=2 -> with priorities 0.6 and 0.4 they land in different levels.
        // Capacity=3 (odd) -> initial levelIndex = 1, so the first takeOut() picks the high level.
        final Bag<Concept, Term> bag = new Bag<>(2, 3, 0);

        // Context vector (simulated focus)
        final Hypervector contextVec = Hypervector.random(12345);

        // Contender A: higher priority, but maximally irrelevant (complement => similarity ~ 0.0)
        final Concept conceptA = new Concept(new BudgetValue(0.6f, 0.5f, 0.5f, nar.memory.narParameters), new Term("A"), nar.memory);
        conceptA.setPriority(0.6f);
        conceptA.vector = complementOf(contextVec);

        // Contender B: lower priority, but perfectly relevant (identical => similarity 1.0)
        final Concept conceptB = new Concept(new BudgetValue(0.4f, 0.5f, 0.5f, nar.memory.narParameters), new Term("B"), nar.memory);
        conceptB.setPriority(0.4f);
        conceptB.vector = contextVec;

        bag.putIn(conceptA);
        bag.putIn(conceptB);

        // Sanity check similarities (proof preconditions)
        assertEquals(0.0, conceptA.vector.similarity(contextVec), 1e-9);
        assertEquals(1.0, conceptB.vector.similarity(contextVec), 1e-9);

        // Predicted scores:
        // A: 0.6 * (0.3 + 0.7 * 0.0) = 0.18
        // B: 0.4 * (0.3 + 0.7 * 1.0) = 0.40
        final Concept selected = bag.takeWithContext(contextVec);

        assertNotNull("Should select a concept", selected);
        assertSame("Should select Concept B (Target) due to vector similarity", conceptB, selected);
    }

    @Test
    public void testBaselinePriorityWinsWithoutContext() throws Exception {
        System.clearProperty("opennars.vectorContext");

        final Nar nar = new Nar();
        final Bag<Concept, Term> bag = new Bag<>(2, 3, 0);

        final Concept conceptA = new Concept(new BudgetValue(0.6f, 0.5f, 0.5f, nar.memory.narParameters), new Term("A"), nar.memory);
        conceptA.setPriority(0.6f);
        conceptA.vector = Hypervector.random(1);

        final Concept conceptB = new Concept(new BudgetValue(0.4f, 0.5f, 0.5f, nar.memory.narParameters), new Term("B"), nar.memory);
        conceptB.setPriority(0.4f);
        conceptB.vector = Hypervector.random(2);

        bag.putIn(conceptA);
        bag.putIn(conceptB);

        // With null context, Bag.takeWithContext delegates to Bag.takeOut.
        // With this deterministic 2-level bag setup, the first takeOut picks the high level.
        final Concept selected = bag.takeWithContext(null);

        assertNotNull("Should select a concept", selected);
        assertSame("Should select Concept A (High Priority) when no context provided", conceptA, selected);
    }
}
