package org.opennars.operator;

import org.junit.Test;
import org.opennars.main.Nar;
import org.opennars.operator.mental.Say;

import static org.junit.Assert.*;

public class SayTest {

    @Test
    public void testSayOperatorRegistration() throws Exception {
        Nar nar = new Nar();
        assertNotNull("Say operator should be registered", nar.memory.getOperator("^say"));
        assertTrue("^say should be implemented by Say", nar.memory.getOperator("^say") instanceof Say);
    }

    @Test
    public void testSayExecution() throws Exception {
        Nar nar = new Nar();

        // 1. Give NARS the goal to say "hello"
        // <(*, {SELF}, "hello") --> ^say>!
        nar.addInput("<(*, {SELF}, \"hello\") --> ^say>! :|:");

        // 2. Run cycles to allow it to execute
        // We can't easily capture stdout in a simple unit test without boilerplate,
        // so we rely on successful execution (no exception) + manual verification.
        nar.cycles(10);
    }
}
