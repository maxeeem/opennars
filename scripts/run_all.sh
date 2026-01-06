#!/bin/bash
set -e

echo "Building OpenNARS..."
mvn -Dmaven.javadoc.skip=true package -DskipTests

echo "Running Baseline (Domains: all)..."
python broca.py --domain all --run baseline --reps 50 --seed 1

echo "Running Ablation (Domains: all)..."
python broca.py --domain all --run ablate_bridge --reps 50 --seed 1

echo "Done."
