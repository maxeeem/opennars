package org.opennars.entity;

import java.util.Random;

/**
 * Projects dense float vectors (e.g., GloVe 300d) into Binary Hypervectors (1024 bits).
 * Uses Sparse Ternary Projections (Achlioptas distribution) for speed.
 */
public class ProjectionMatrix {
    private final int inputDim;
    private final int outputDim;
    private final byte[][] matrix; // values in {-1, 0, +1}

    public ProjectionMatrix(final int inputDim, final long seed) {
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive");
        }
        this.inputDim = inputDim;
        this.outputDim = 1024;
        this.matrix = new byte[outputDim][inputDim];

        final Random rand = new Random(seed);
        for (int r = 0; r < outputDim; r++) {
            for (int c = 0; c < inputDim; c++) {
                // Achlioptas Distribution:
                // 1/6 probability of +1
                // 1/6 probability of -1
                // 2/3 probability of 0
                final float val = rand.nextFloat();
                if (val < 0.1666f) {
                    this.matrix[r][c] = 1;
                } else if (val < 0.3333f) {
                    this.matrix[r][c] = -1;
                } else {
                    this.matrix[r][c] = 0;
                }
            }
        }
    }

    public Hypervector project(final float[] input) {
        if (input == null || input.length != inputDim) {
            throw new IllegalArgumentException("Input dim mismatch");
        }

        final long[] bits = new long[outputDim / 64];
        for (int i = 0; i < outputDim; i++) {
            float sum = 0.0f;
            final byte[] row = matrix[i];
            for (int j = 0; j < inputDim; j++) {
                final byte w = row[j];
                if (w == 1) {
                    sum += input[j];
                } else if (w == -1) {
                    sum -= input[j];
                }
            }

            if (sum > 0.0f) {
                final int longIdx = i / 64;
                final int bitIdx = i % 64;
                bits[longIdx] |= (1L << bitIdx);
            }
        }

        return new Hypervector(bits);
    }
}
