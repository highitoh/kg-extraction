#!/bin/bash

# Action Applicator Wrapper Script
# Usage: ./apply.sh [--no-backup] [--no-progress]

cd /workspace/app
python src/optimize/action_applicator.py "$@"
