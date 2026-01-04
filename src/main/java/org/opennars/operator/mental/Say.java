/* 
 * The MIT License
 *
 * Copyright 2018 The OpenNARS authors.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.opennars.operator.mental;

import org.opennars.entity.Task;
import org.opennars.interfaces.Timable;
import org.opennars.language.Term;
import org.opennars.operator.Operation;
import org.opennars.operator.Operator;
import org.opennars.storage.Memory;

import java.util.List;

/**
 * The Vocal Cord of NARS.
 * Usage: <(*, {SELF}, "content") --> ^say>
 * Effect: Prints "[OUTPUT] content" to stdout.
 */
public class Say extends Operator {

    public Say() {
        super("^say");
    }

    @Override
    protected List<Task> execute(final Operation operation, final Term[] args, final Memory memory, final Timable time) {
        if (args == null || args.length == 0) {
            return null;
        }

        // Convention in this codebase: args[0] is usually {SELF}, args[1] is the payload.
        final Term content = (args.length >= 2) ? args[1] : args[0];
        if (content == null) {
            return null;
        }

        final String raw = String.valueOf(content.name());
        System.out.println("[OUTPUT] " + normalizeContent(raw));
        return null;
    }

    private static String normalizeContent(final String raw) {
        if (raw == null) {
            return "";
        }
        if (raw.length() >= 2 && raw.charAt(0) == '"' && raw.charAt(raw.length() - 1) == '"') {
            return raw.substring(1, raw.length() - 1);
        }
        return raw;
    }
}
