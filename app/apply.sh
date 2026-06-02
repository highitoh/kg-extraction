#!/bin/bash

# Action Applicator Wrapper Script
# Usage: ./apply.sh [--no-backup] [--no-progress]

cd /workspace/app
uv run python src/optimize/action_applicator.py "$@"
