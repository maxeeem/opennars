package org.opennars.entity;

import org.junit.Test;

import static org.junit.Assert.*;

public class LearningTest {

    @Test
    public void testAttractionNudgeDecreasesDistance() {
        final Hypervector v1 = Hypervector.random(1001);
        final Hypervector v2 = Hypervector.random(2002);

        final double dist0 = 1.0 - v1.similarity(v2);

        v1.nudge(v2, 0.2f);

        final double dist1 = 1.0 - v1.similarity(v2);

        assertTrue("Attraction should decrease distance", dist1 < dist0);
        assertTrue("Distance should end up below ~0.5 for random vectors", dist1 < 0.5);
    }

    @Test
    public void testRepulsionNudgeIncreasesDistance() {
        final Hypervector v1 = Hypervector.random(3003);
        final Hypervector v2 = Hypervector.random(4004);

        final double dist0 = 1.0 - v1.similarity(v2);

        v1.nudge(v2.invert(), 0.2f);

        final double dist1 = 1.0 - v1.similarity(v2);

        assertTrue("Repulsion should increase distance", dist1 > dist0);
        assertTrue("Distance should end up above ~0.5 for random vectors", dist1 > 0.5);
    }
}
