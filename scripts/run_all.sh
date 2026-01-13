#!/bin/bash
set -e

STAMP=$(date +"%Y%m%d_%H%M%S")
echo "Run Timestamp: $STAMP"

echo "Building OpenNARS..."
mvn -Dmaven.javadoc.skip=true package -DskipTests

echo "Running Baseline (Domains: all, Stamp: $STAMP)..."
# Using 50 reps (100 trials total per condition) to keep run time manageable (~10 mins)
# Full paper version would use >400 reps.
python broca.py --domain all --run baseline --reps 5 --seed 1 --stamp "$STAMP"

echo "Running Ablation (Domains: all, Stamp: $STAMP)..."
python broca.py --domain all --run ablate_bridge --reps 5 --seed 1 --stamp "$STAMP"

echo "Done. Results in runs/ folder starting with $STAMP."
