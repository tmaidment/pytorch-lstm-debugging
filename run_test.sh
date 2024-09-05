#!/bin/bash

# Run tests for different PyTorch versions
run_test() {
    echo "Running test for PyTorch $1"
    conda activate pytorch_$1
    python test.py
    conda deactivate
}

run_test "1.13.0"
run_test "2.0.0"