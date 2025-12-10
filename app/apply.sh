#!/bin/bash

# Action Applicator Wrapper Script
# Usage: ./apply.sh --project PROJECT [--no-backup] [--no-progress]

cd /workspace/app
python src/optimize/action_applicator.py "$@"
