package org.opennars.storage;

import org.junit.Test;
import org.opennars.entity.BudgetValue;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.io.Narsese;
import org.opennars.main.Nar;
import org.opennars.language.Term;

import java.io.File;

import static org.junit.Assert.*;

public class PersistenceTest {

    @Test
    public void testVectorPersistence() throws Exception {
        final String filename = "test_vector_persistence.nar";
        final File file = new File(filename);

        // 1. Create and Train
        final Nar nar1 = new Nar();
        final Term term = new Narsese(nar1).parseTerm("TestTerm");
        final BudgetValue budget = new BudgetValue(1.0f, 0.5f, 0.5f, nar1.narParameters);

        final Concept c1 = nar1.memory.conceptualize(budget, term);
        assertNotNull("Concept should be created", c1);

        // Set a specific deterministic vector (Seed 12345)
        final Hypervector vecOriginal = Hypervector.random(12345);
        c1.vector = vecOriginal;
        c1.hasUserVector = true; // Mark as "Real" data

        // 2. Save
        nar1.SaveToFile(filename);

        // 3. Load
        final Nar nar2 = Nar.LoadFromFile(filename);
        final Term term2 = new Narsese(nar2).parseTerm("TestTerm");
        final Concept c2 = nar2.memory.concept(term2);

        // 4. Verify
        assertNotNull("Loaded concept should exist", c2);
        assertNotNull("Vector should be persisted", c2.vector);

        final double similarity = vecOriginal.similarity(c2.vector);
        assertEquals("Persisted vector should match original perfectly", 1.0, similarity, 1e-12);
        assertTrue("Should preserve the 'Real Data' flag", c2.hasUserVector);

        // Cleanup
        //noinspection ResultOfMethodCallIgnored
        file.delete();
    }
}
