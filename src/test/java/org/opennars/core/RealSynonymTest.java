package org.opennars.core;

import org.junit.Test;

import static org.junit.Assert.*;

import org.opennars.io.events.AnswerHandler;
import org.opennars.language.Term;
import org.opennars.main.Nar;
import org.opennars.storage.GloVeLoader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Loads a tiny (synthetic) mini-GloVe file and proves that vector grounding enables
 * synonym-style property transfer without manual Hypervector assignment.
 *
 * Teach: <cat --> furry>.
 * Ask:   <kitten --> furry>?
 * Expect: Answer derived via injected <cat <-> kitten>.
 */
public class RealSynonymTest {

    @Test
    public void testRealWorldGroundingMiniGlove() throws Exception {
        final String prev = System.getProperty("opennars.vector");
        System.setProperty("opennars.vector", "true");
        try {
            final Nar nar = new Nar();

            // Build a tiny 300d "GloVe" file.
            // cat and kitten are identical here to guarantee strong similarity after projection.
            final File tmp = File.createTempFile("opennars-mini-glove", ".txt");
            tmp.deleteOnExit();

            try (BufferedWriter w = new BufferedWriter(new FileWriter(tmp))) {
                w.write(buildLine("cat", buildCatVector()));
                w.newLine();
                w.write(buildLine("kitten", buildCatVector()));
                w.newLine();
                w.write(buildLine("dog", buildDogVector()));
                w.newLine();
            }

            GloVeLoader.load(nar, tmp, 3);

            // Teach knowledge about cat.
            nar.addInput("<cat --> furry>.");
            for (int i = 0; i < 25; i++) {
                nar.cycle();
            }

            final boolean[] success = {false};
            nar.ask("<kitten --> furry>", new AnswerHandler() {
                @Override
                @SuppressWarnings("rawtypes")
                public void onSolution(org.opennars.entity.Sentence belief) {
                    final String s = String.valueOf(belief);
                    if (s.contains("kitten") && s.contains("furry") && s.contains("-->")) {
                        success[0] = true;
                        off();
                    }
                }
            });

            for (int i = 0; i < 800; i++) {
                nar.cycle();
                if (success[0]) break;
            }

            assertTrue("System should infer <kitten --> furry> via projected GloVe vectors", success[0]);
        } finally {
            if (prev == null) {
                System.clearProperty("opennars.vector");
            } else {
                System.setProperty("opennars.vector", prev);
            }
        }
    }

    private static float[] buildCatVector() {
        final float[] v = new float[300];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) Math.sin(i * 0.01);
        }
        return v;
    }

    private static float[] buildDogVector() {
        final float[] v = new float[300];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) Math.cos(i * 0.02) * 3.0f;
        }
        return v;
    }

    private static String buildLine(final String word, final float[] vec) {
        final StringBuilder sb = new StringBuilder();
        sb.append(Term.get(word));
        for (final float f : vec) {
            sb.append(' ');
            sb.append(Float.toString(f));
        }
        return sb.toString();
    }
}
