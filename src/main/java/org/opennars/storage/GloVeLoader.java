package org.opennars.storage;

import org.opennars.entity.BudgetValue;
import org.opennars.entity.Concept;
import org.opennars.entity.Hypervector;
import org.opennars.entity.ProjectionMatrix;
import org.opennars.language.Term;
import org.opennars.main.Nar;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * High-Performance GloVe Loader.
 * Features:
 * 1. Parallel Parsing (utilizes all CPU cores for projection work).
 * 2. Binary Caching (saves projected vectors to .bin for fast reload).
 * 3. Progress Tracking (periodic console updates).
 */
public final class GloVeLoader {

    private static final int CACHE_VERSION = 1;
    private static final int DEFAULT_BATCH_LINES = 50_000;

    private GloVeLoader() {
    }

    public static void load(final Nar nar, final File gloveFile, final int limit) throws IOException {
        load(nar, gloveFile, 300, limit, 42L);
    }

    public static void load(final Nar nar, final File gloveFile, final int inputDim, final int limit, final long seed) throws IOException {
        if (nar == null) {
            throw new IllegalArgumentException("nar is null");
        }
        if (gloveFile == null) {
            throw new IllegalArgumentException("gloveFile is null");
        }
        if (!gloveFile.exists()) {
            throw new FileNotFoundException(gloveFile.getAbsolutePath());
        }
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive");
        }

        final File cacheFile = new File(gloveFile.getAbsolutePath() + ".bin");
        if (cacheFile.exists() && cacheFile.length() > 0) {
            System.out.println("   [Cache Detected] Loading binary vectors...");
            loadFromBinary(nar, cacheFile, inputDim, limit);
            return;
        }

        System.out.println("   [Parsing Text] Generating hypervectors (Multi-threaded)...");
        parseAndCache(nar, gloveFile, cacheFile, inputDim, limit, seed);
    }

    private static void loadFromBinary(final Nar nar, final File cacheFile, final int expectedInputDim, final int limit) throws IOException {
        final BudgetValue activation = new BudgetValue(1.0f, 0.9f, 1.0f, nar.narParameters);
        final long startTime = System.currentTimeMillis();

        int loaded = 0;
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(cacheFile)))) {
            final int version = dis.readInt();
            if (version != CACHE_VERSION) {
                throw new IOException("Unsupported cache version: " + version);
            }
            final int inputDim = dis.readInt();
            final int longsPerVector = dis.readInt();
            if (inputDim != expectedInputDim) {
                throw new IOException("Cache inputDim mismatch: expected " + expectedInputDim + " but got " + inputDim);
            }

            while (limit <= 0 || loaded < limit) {
                try {
                    final String word = dis.readUTF();
                    final long[] bits = new long[longsPerVector];
                    for (int i = 0; i < longsPerVector; i++) {
                        bits[i] = dis.readLong();
                    }

                    final Term term = Term.get(word);
                    final Concept c = nar.memory.conceptualize(activation, term);
                    if (c != null) {
                        c.vector = new Hypervector(bits);
                    }

                    loaded++;
                    if (loaded % 5000 == 0) {
                        printProgress(loaded, startTime);
                    }
                } catch (EOFException eof) {
                    break;
                }
            }
        }

        System.out.println("\n   Loaded " + loaded + " vectors from cache.");
    }

    private static void parseAndCache(
            final Nar nar,
            final File txtFile,
            final File binFile,
            final int inputDim,
            final int limit,
            final long seed) throws IOException {

        final ProjectionMatrix proj = new ProjectionMatrix(inputDim, seed);
        final BudgetValue activation = new BudgetValue(1.0f, 0.9f, 1.0f, nar.narParameters);
        final long startTime = System.currentTimeMillis();

        final int longsPerVector = 16; // 1024 bits packed into 16 longs
        final AtomicInteger totalLoaded = new AtomicInteger(0);

        boolean wroteAny = false;
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(binFile)));
             BufferedReader br = new BufferedReader(new FileReader(txtFile))) {

            dos.writeInt(CACHE_VERSION);
            dos.writeInt(inputDim);
            dos.writeInt(longsPerVector);

            while (limit <= 0 || totalLoaded.get() < limit) {
                final int remaining = (limit <= 0) ? Integer.MAX_VALUE : (limit - totalLoaded.get());
                final int batchSize = Math.min(DEFAULT_BATCH_LINES, remaining);
                final List<String> lines = readBatch(br, batchSize);
                if (lines.isEmpty()) {
                    break;
                }

                final Entry[] entries = new Entry[lines.size()];
                IntStream.range(0, lines.size()).parallel().forEach(i -> {
                    final Entry e = parseAndProject(lines.get(i), inputDim, proj);
                    entries[i] = e;
                });

                for (final Entry entry : entries) {
                    if (entry == null) {
                        continue;
                    }
                    if (limit > 0 && totalLoaded.get() >= limit) {
                        break;
                    }

                    dos.writeUTF(entry.word);
                    final long[] bits = entry.hv.getBits();
                    if (bits.length != longsPerVector) {
                        continue;
                    }
                    for (int i = 0; i < longsPerVector; i++) {
                        dos.writeLong(bits[i]);
                    }

                    final Term term = Term.get(entry.word);
                    final Concept c = nar.memory.conceptualize(activation, term);
                    if (c != null) {
                        c.vector = entry.hv;
                    }

                    wroteAny = true;
                    final int current = totalLoaded.incrementAndGet();
                    if (current % 1000 == 0) {
                        printProgress(current, startTime);
                    }
                }
            }
        } catch (Exception e) {
            if (!wroteAny && binFile.exists()) {
                // best-effort cleanup of an empty/partial cache
                //noinspection ResultOfMethodCallIgnored
                binFile.delete();
            }
            if (e instanceof IOException) {
                throw (IOException) e;
            }
            throw new IOException(e);
        }

        System.out.println("\n   Parsed & Cached " + totalLoaded.get() + " vectors.");
    }

    private static List<String> readBatch(final BufferedReader br, final int maxLines) throws IOException {
        final List<String> lines = new ArrayList<>(Math.min(maxLines, 8192));
        for (int i = 0; i < maxLines; i++) {
            final String line = br.readLine();
            if (line == null) {
                break;
            }
            final String trimmed = line.trim();
            if (!trimmed.isEmpty()) {
                lines.add(trimmed);
            }
        }
        return lines;
    }

    private static Entry parseAndProject(final String line, final int inputDim, final ProjectionMatrix proj) {
        if (line == null || line.isEmpty()) {
            return null;
        }

        try {
            final StringTokenizer st = new StringTokenizer(line);
            if (!st.hasMoreTokens()) {
                return null;
            }
            final String word = st.nextToken();
            final float[] vec = new float[inputDim];

            for (int i = 0; i < inputDim; i++) {
                if (!st.hasMoreTokens()) {
                    return null;
                }
                vec[i] = Float.parseFloat(st.nextToken());
            }

            final Hypervector hv = proj.project(vec);
            return new Entry(word, hv);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void printProgress(final int count, final long startTime) {
        final long elapsed = System.currentTimeMillis() - startTime;
        final double rate = count / (Math.max(1, elapsed) / 1000.0);
        System.out.print(String.format("\r   Loading... %d vectors processed [%.0f vecs/sec]  ", count, rate));
    }

    private static final class Entry {
        final String word;
        final Hypervector hv;

        Entry(final String word, final Hypervector hv) {
            this.word = word;
            this.hv = hv;
        }
    }
}
