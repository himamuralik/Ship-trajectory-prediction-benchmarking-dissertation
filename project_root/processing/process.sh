#!/usr/bin/env bash

DATASET="new_york"
LOG_LEVEL=4
SEED=47033218

# Run pipeline with error checking
run_step() {
    echo "Running: $*"
    if ! python "$@"; then
        echo "Error in step: $*"
        exit 1
    fi
}

run_step downloader.py "$DATASET" -l "$LOG_LEVEL" -s
run_step cleaner.py "$DATASET" -l "$LOG_LEVEL" -s --seed "$SEED" --memory conserve
run_step interpolator.py "$DATASET" -l "$LOG_LEVEL" -s
run_step sliding_window.py "$DATASET" -l "$LOG_LEVEL" -s
run_step formatter.py "$DATASET" -l "$LOG_LEVEL" -s

echo "Pipeline completed successfully"
