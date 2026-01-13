#!/usr/bin/env python3
"""
Quick test of minimal motor grounding (reduced episodes for fast validation).
"""

import subprocess
import sys

def main():
    print("\n" + "="*60)
    print("QUICK TEST: Minimal Motor Grounding")
    print("="*60 + "\n")
    
    # Test with just 1 rep, fewer episodes to verify the system works
    cmd = [
        "python3", "broca.py",
        "--domain", "micro_gscan",
        "--condition", "bridge_on",
        "--reps", "1",
        "--seed", "42",
    ]
    
    print("Running: " + " ".join(cmd))
    print("(This will run 1 rep with default episodes)\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Test completed successfully!")
        print("Check runs/ directory for output files")
    else:
        print(f"\n✗ Test failed with code {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
