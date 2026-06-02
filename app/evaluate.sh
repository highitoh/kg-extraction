#!/bin/bash

# Knowledge Evaluation Wrapper Script
# Usage: ./evaluate.sh [--project PROJECT] [--model MODEL]

cd /workspace/app
uv run python src/evaluate_workflow.py "$@"
