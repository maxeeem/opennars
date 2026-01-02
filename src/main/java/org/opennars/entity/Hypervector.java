package org.opennars.entity;

import java.io.Serializable;
import java.util.Random;

/**
 * Binary Spatter Code implementation for VectorNARS.
 * Uses 1024-bit binary vectors packed into longs.
 */
public class Hypervector implements Serializable {
    // 1024 bits / 64 bits-per-long = 16 longs
    private static final int LONGS = 16; 
    private final long[] bits;

    public Hypervector() {
        this.bits = new long[LONGS];
    }
    
    private Hypervector(long[] bits) {
        this.bits = bits;
    }

    /** * Initialize random vector (Orthogonal Mode). 
     * Used for concepts without pre-trained embeddings.
     */
    public static Hypervector random(long seed) {
        Random rand = new Random(seed);
        long[] newBits = new long[LONGS];
        for (int i = 0; i < LONGS; i++) {
            newBits[i] = rand.nextLong();
        }
        return new Hypervector(newBits);
    }

    /**
     * Normalized Hamming Distance.
     * @return 0.0 (Different) to 1.0 (Identical).
     */
    public double similarity(Hypervector other) {
        if (other == null) return 0.5; // Neutral
        int diff = 0;
        for (int i = 0; i < LONGS; i++) {
            diff += Long.bitCount(this.bits[i] ^ other.bits[i]);
        }
        return 1.0 - ((double)diff / (LONGS * 64.0));
    }

    /**
     * Hebbian Learning (Nudge).
     * Moves this vector slightly closer to the target vector.
     * @param target The vector to learn from.
     * @param rate Probability of flipping a bit to match target (0.0 - 1.0).
     */
    public void nudge(Hypervector target, double rate) {
        if (target == null) return;
        Random rand = new Random();
        for (int i = 0; i < LONGS; i++) {
            long diffMask = this.bits[i] ^ target.bits[i]; // 1 where bits differ
            if (diffMask == 0) continue;
            
            // Iterate bits in the long (simplified for performance)
            // Ideally: iterate all 64 bits. Here we do a stochastic update per long for speed
            // or we can just replace chunks.
            // Accurate bit-flip implementation:
            for (int b = 0; b < 64; b++) {
                long mask = 1L << b;
                if ((diffMask & mask) != 0) { // If bits differ
                    if (rand.nextDouble() < rate) {
                        this.bits[i] ^= mask; // Flip to match
                    }
                }
            }
        }
    }
}
