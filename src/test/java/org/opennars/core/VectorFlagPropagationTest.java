package org.opennars.core;

import org.junit.Test;
import org.opennars.control.VectorInference;

import static org.junit.Assert.assertTrue;

public class VectorFlagPropagationTest {

    @Test
    public void testVectorFlagReachesTestJVM() {
        final String raw = System.getProperty("opennars.vector");
        assertTrue(
                "Expected -Dopennars.vector=true in test JVM, but got opennars.vector=" + raw,
                Boolean.getBoolean("opennars.vector"));

        assertTrue(
                "VectorInference.isEnabled() should be true when opennars.vector=true",
                VectorInference.isEnabled());
    }
}
