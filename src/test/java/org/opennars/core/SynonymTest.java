package org.opennars.core;

import org.junit.Test;

import static org.junit.Assert.*;

import org.opennars.entity.BudgetValue;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.entity.Sentence;
import org.opennars.io.events.AnswerHandler;
import org.opennars.language.Term;
import org.opennars.main.Nar;

/**
 * Proves that VectorNARS can solve the Synonym Problem via a dynamic similarity bridge.
 *
 * Logic:
 * 1. Teach: <cat --> furry>.
 * 2. Query: <kitty --> furry>?
 * 3. Fact: cat and kitty share the same semantic hypervector.
 * 4. Expect: system answers kitty is furry (via injected <cat <-> kitty>).
 */
public class SynonymTest {

    @Test
    public void testSynonymInference() throws Exception {
        final String prev = System.getProperty("opennars.vectorContext");
        System.setProperty("opennars.vectorContext", "true");
        try {
            final Nar nar = new Nar();

            // Ensure concepts exist and assign a shared semantic vector.
            final BudgetValue activation = new BudgetValue(1.0f, 0.9f, 1.0f, nar.narParameters);
            final Term termCat = Term.get("cat");
            final Term termKitty = Term.get("kitty");

            final Concept cat = nar.memory.conceptualize(activation, termCat);
            final Concept kitty = nar.memory.conceptualize(activation, termKitty);
            assertNotNull(cat);
            assertNotNull(kitty);

            final Hypervector shared = Hypervector.random(100);
            cat.vector = shared;
            kitty.vector = shared;

            // Teach knowledge about cat.
            nar.addInput("<cat --> furry>.");

            // Let the knowledge settle a bit.
            for (int i = 0; i < 25; i++) {
                nar.cycle();
            }

            final boolean[] success = {false};
            nar.ask("<kitty --> furry>", new AnswerHandler() {
                @Override
                @SuppressWarnings("rawtypes")
                public void onSolution(Sentence belief) {
                    final String s = String.valueOf(belief);
                    if (s.contains("kitty") && s.contains("furry") && s.contains("-->")) {
                        success[0] = true;
                        off();
                    }
                }
            });

            for (int i = 0; i < 500; i++) {
                nar.cycle();
                if (success[0]) break;
            }

            assertTrue("System should infer <kitty --> furry> via vector analogy", success[0]);
        } finally {
            if (prev == null) {
                System.clearProperty("opennars.vectorContext");
            } else {
                System.setProperty("opennars.vectorContext", prev);
            }
        }
    }
}
