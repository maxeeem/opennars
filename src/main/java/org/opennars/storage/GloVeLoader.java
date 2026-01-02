package org.opennars.storage;

import org.opennars.entity.BudgetValue;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.entity.ProjectionMatrix;
import org.opennars.language.Term;
import org.opennars.main.Nar;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Loads GloVe-style embeddings from a text file and projects them into Hypervectors.
 *
 * Expected format per line:
 *   word v1 v2 ... vN
 */
public final class GloVeLoader {

    private GloVeLoader() {
    }

    public static void load(final Nar nar, final File gloveFile, final int inputDim, final int limit, final long seed) throws IOException {
        if (nar == null) {
            throw new IllegalArgumentException("nar is null");
        }
        if (gloveFile == null) {
            throw new IllegalArgumentException("gloveFile is null");
        }
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive");
        }

        final ProjectionMatrix proj = new ProjectionMatrix(inputDim, seed);
        final BudgetValue activation = new BudgetValue(1.0f, 0.9f, 1.0f, nar.narParameters);

        int loaded = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(gloveFile))) {
            String line;
            while ((line = br.readLine()) != null && (limit <= 0 || loaded < limit)) {
                line = line.trim();
                if (line.isEmpty()) continue;

                final String[] parts = line.split("\\s+");
                if (parts.length != inputDim + 1) {
                    continue;
                }

                final String word = parts[0];
                final float[] vec = new float[inputDim];
                try {
                    for (int i = 0; i < inputDim; i++) {
                        vec[i] = Float.parseFloat(parts[i + 1]);
                    }

                    final Hypervector hv = proj.project(vec);

                    final Term term = Term.get(word);
                    final Concept c = nar.memory.conceptualize(activation, term);
                    if (c != null) {
                        c.vector = hv;
                    }

                    loaded++;
                } catch (Exception ignored) {
                    // Skip malformed lines.
                }
            }
        }
    }

    public static void load(final Nar nar, final File gloveFile, final int limit) throws IOException {
        load(nar, gloveFile, 300, limit, 42L);
    }
}
