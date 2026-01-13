package org.opennars.storage;

import org.opennars.entity.Hypervector;
import org.opennars.entity.ProjectionMatrix;
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
import java.util.Map;
import java.util.StringTokenizer;
import java.util.concurrent.ConcurrentHashMap;
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

    private static final int CACHE_VERSION = 2;
    private static final int DEFAULT_BATCH_LINES = 50_000;
    private static final long DEFAULT_SEED = 42L;
    private static final long MIN_CACHE_BYTES_V1 = 12L; // 3 ints
    private static final long MIN_CACHE_BYTES_V2 = 20L; // 3 ints + 1 long

    private GloVeLoader() {
    }

    public static void load(final Nar nar, final File gloveFile, final int limit) throws IOException {
        loadAndCount(nar, gloveFile, limit);
    }

    /**
     * Loads vectors and returns how many were loaded. Infers input dimension from the text file.
     */
    public static int loadAndCount(final Nar nar, final File gloveFile, final int limit) throws IOException {
        final int inferredDim = inferInputDim(gloveFile);
        return loadAndCount(nar, gloveFile, inferredDim, limit, DEFAULT_SEED);
    }

    public static void load(final Nar nar, final File gloveFile, final int inputDim, final int limit, final long seed) throws IOException {
        loadAndCount(nar, gloveFile, inputDim, limit, seed);
    }

    /**
     * Loads vectors and returns how many were loaded.
     */
    public static int loadAndCount(final Nar nar, final File gloveFile, final int inputDim, final int limit, final long seed) throws IOException {
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
            // Guard against previously-created empty header-only caches.
            if (cacheFile.length() <= MIN_CACHE_BYTES_V1) {
                //noinspection ResultOfMethodCallIgnored
                cacheFile.delete();
            } else {
            System.out.println("   [Cache Detected] Loading binary vectors...");
                final int loaded = loadFromBinary(nar, cacheFile, inputDim, seed, limit);
                if (loaded > 0) {
                    return loaded;
                }
                // Fallback: invalid/stale cache (or contained no vectors)
                System.out.println("   [Cache Ignored] Regenerating from text...");
            }
        }

        System.out.println("   [Parsing Text] Generating hypervectors (Multi-threaded)...");
        return parseAndCache(nar, gloveFile, cacheFile, inputDim, limit, seed);
    }

    private static int loadFromBinary(final Nar nar, final File cacheFile, final int expectedInputDim, final long expectedSeed, final int limit) throws IOException {
        final long startTime = System.currentTimeMillis();
        final Map<String, Hypervector> map = ensureEmbeddingMap(nar);

        int loaded = 0;
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(cacheFile)))) {
            final int version = dis.readInt();
            if (version != CACHE_VERSION) {
                //noinspection ResultOfMethodCallIgnored
                cacheFile.delete();
                return 0;
            }
            final int inputDim = dis.readInt();
            final int longsPerVector = dis.readInt();
            final long seed = dis.readLong();
            if (cacheFile.length() <= MIN_CACHE_BYTES_V2) {
                //noinspection ResultOfMethodCallIgnored
                cacheFile.delete();
                return 0;
            }
            if (inputDim != expectedInputDim || seed != expectedSeed) {
                //noinspection ResultOfMethodCallIgnored
                cacheFile.delete();
                return 0;
            }

            while (limit <= 0 || loaded < limit) {
                try {
                    final String word = dis.readUTF();
                    final long[] bits = new long[longsPerVector];
                    for (int i = 0; i < longsPerVector; i++) {
                        bits[i] = dis.readLong();
                    }

                    map.put(word, new Hypervector(bits));

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
        if (loaded == 0) {
            //noinspection ResultOfMethodCallIgnored
            cacheFile.delete();
        }
        return loaded;
    }

    private static int parseAndCache(
            final Nar nar,
            final File txtFile,
            final File binFile,
            final int inputDim,
            final int limit,
            final long seed) throws IOException {

        final ProjectionMatrix proj = new ProjectionMatrix(inputDim, seed);
        final long startTime = System.currentTimeMillis();
        final Map<String, Hypervector> map = ensureEmbeddingMap(nar);

        final int longsPerVector = 16; // 1024 bits packed into 16 longs
        final AtomicInteger totalLoaded = new AtomicInteger(0);

        boolean wroteAny = false;
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(binFile)));
             BufferedReader br = new BufferedReader(new FileReader(txtFile))) {

            dos.writeInt(CACHE_VERSION);
            dos.writeInt(inputDim);
            dos.writeInt(longsPerVector);
            dos.writeLong(seed);

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

                    map.put(entry.word, entry.hv);

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

        final int loaded = totalLoaded.get();
        if (loaded == 0) {
            // Don't leave behind a misleading empty cache.
            //noinspection ResultOfMethodCallIgnored
            binFile.delete();
            throw new IOException("No vectors were loaded. Likely embedding dimension mismatch (expected " + inputDim + ") or malformed file: " + txtFile.getAbsolutePath());
        }

        System.out.println("\n   Parsed & Cached " + loaded + " vectors.");
        return loaded;
    }

    private static int inferInputDim(final File gloveFile) throws IOException {
        if (gloveFile == null) {
            throw new IllegalArgumentException("gloveFile is null");
        }
        if (!gloveFile.exists()) {
            throw new FileNotFoundException(gloveFile.getAbsolutePath());
        }

        try (BufferedReader br = new BufferedReader(new FileReader(gloveFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                final String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                final StringTokenizer st = new StringTokenizer(trimmed);
                final int tokenCount = st.countTokens();

                // Handle word2vec-style header: "<vocabSize> <dim>"
                if (tokenCount == 2) {
                    final String a = st.nextToken();
                    final String b = st.nextToken();
                    if (isInteger(a) && isInteger(b)) {
                        continue;
                    }
                }

                if (tokenCount < 3) {
                    continue;
                }
                final int dim = tokenCount - 1;
                if (dim <= 0) {
                    continue;
                }
                return dim;
            }
        }

        throw new IOException("Could not infer embedding dimension from file: " + gloveFile.getAbsolutePath());
    }

    private static boolean isInteger(final String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        int i = 0;
        final int len = s.length();
        if (s.charAt(0) == '-') {
            if (len == 1) {
                return false;
            }
            i = 1;
        }
        for (; i < len; i++) {
            final char c = s.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
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

    private static Map<String, Hypervector> ensureEmbeddingMap(final Nar nar) {
        if (nar.memory.gloveVectors == null) {
            nar.memory.gloveVectors = new ConcurrentHashMap<>(64 * 1024);
        }
        return nar.memory.gloveVectors;
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
