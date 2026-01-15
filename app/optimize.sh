#!/bin/bash

# Knowledge Extraction Optimization Wrapper Script
# Usage: ./optimize.sh --projects PROJECT1 [PROJECT2 ...] [--model MODEL] [--output-dir DIR]

cd /workspace/app
python src/optimize_workflow.py "$@"
