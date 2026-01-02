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

    Hypervector(long[] bits, boolean trustedNoCopy) {
        if (bits == null || bits.length != LONGS) {
            throw new IllegalArgumentException("Expected " + LONGS + " longs (1024 bits)");
        }
        this.bits = trustedNoCopy ? bits : bits.clone();
    }

    /**
     * Construct a hypervector from raw bit-packing.
     * The input array is defensively copied.
     */
    public Hypervector(final long[] bits) {
        if (bits == null || bits.length != LONGS) {
            throw new IllegalArgumentException("Expected " + LONGS + " longs (1024 bits)");
        }
        this.bits = bits.clone();
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
        return new Hypervector(newBits, true);
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
        Random rand = new Random(deterministicSeed(this.bits, target.bits));
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

    private static long deterministicSeed(long[] a, long[] b) {
        // FNV-1a 64-bit hash over both vectors (stable across runs)
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < a.length; i++) {
            h ^= a[i];
            h *= 0x100000001b3L;
        }
        for (int i = 0; i < b.length; i++) {
            h ^= b[i];
            h *= 0x100000001b3L;
        }
        return h;
    }

    public long[] getBits() {
        return this.bits;
    }
}
