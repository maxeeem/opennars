package org.opennars.core;

import org.junit.Test;
import org.opennars.io.events.AnswerHandler;
import org.opennars.main.Nar;
import org.opennars.storage.GloVeLoader;

import java.io.File;

import static org.junit.Assert.assertTrue;

/**
 * Regression/diagnostic test for VectorNARS + real embeddings:
 *
 * Script equivalent:
 * <<$x --> angry> ==> <$x --> irrational>>.
 * <hulk --> furious>.
 * <hulk --> irrational>?
 * 5000
 */
public class HulkIrrationalGloveTest {

    @Test
    public void testPhase21HulkIrrationalWithLocalGlove() throws Exception {
        final String prevVector = System.getProperty("opennars.vector");
        final String prevVectorContext = System.getProperty("opennars.vectorContext");
        final String prevVectorBridgeInjection = System.getProperty("opennars.vectorBridgeInjection");

        System.setProperty("opennars.vector", "true");
        System.setProperty("opennars.vectorContext", "true");
        System.setProperty("opennars.vectorBridgeInjection", "true");

        try {
            final File gloveFile = new File("glove-embeddings/glove.txt");
            assertTrue(
                    "Missing local embeddings file: " + gloveFile.getAbsolutePath() + "\n" +
                        "Expected it at glove-embeddings/glove.txt (copied from ~/hybrid_nars/hybrid_nars_rust/assets/glove.txt)",
                    gloveFile.exists() && gloveFile.isFile());

            final Nar nar = new Nar();

            // Load all vectors (limit<=0 means no limit). A .bin cache will be generated next to glove.txt.
            final int loaded = GloVeLoader.loadAndCount(nar, gloveFile, 0);
            assertTrue("Failed to load any embeddings from: " + gloveFile.getAbsolutePath(), loaded > 0);

            nar.addInput("<<$x --> angry> ==> <$x --> irrational>>.");
            nar.addInput("<hulk --> furious>.");

            final boolean[] success = {false};
            nar.ask("<hulk --> irrational>", new AnswerHandler() {
                @Override
                @SuppressWarnings("rawtypes")
                public void onSolution(final org.opennars.entity.Sentence belief) {
                    final String s = String.valueOf(belief);
                    if (s.contains("hulk") && s.contains("irrational") && s.contains("-->")) {
                        success[0] = true;
                        off();
                    }
                }
            });

            for (int i = 0; i < 5000; i++) {
                nar.cycle();
                if (success[0]) {
                    break;
                }
            }

            assertTrue("Expected an answer for <hulk --> irrational> within 5000 cycles", success[0]);
        } finally {
            restoreProperty("opennars.vector", prevVector);
            restoreProperty("opennars.vectorContext", prevVectorContext);
            restoreProperty("opennars.vectorBridgeInjection", prevVectorBridgeInjection);
        }
    }

    private static void restoreProperty(final String key, final String prev) {
        if (prev == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, prev);
        }
    }
}
