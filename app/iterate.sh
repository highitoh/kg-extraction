#!/bin/bash

# Iterative Knowledge Extraction Optimization Wrapper Script
# Usage: ./iterate.sh --batch <batch_file1> [batch_file2 ...] [OPTIONS]
#
# Arguments:
#   --batch FILE [FILE ...]  One or more batch files (project derived from filename: batch-{project}.txt)
#
# Options:
#   --max-iterations N       Maximum iterations (default: 5)
#   --quality-threshold F    Stop when combined score >= threshold (default: 2.0)
#   --model MODEL            LLM model name (default: gemini-3.5-flash-lite
#   --no-apply               Dry-run optimization without applying changes
#   --resume RUN_ID          Resume from a previous run
#   --output-dir DIR         Base output directory (default: /workspace/app/output)
#
# Examples:
#   # Single project iteration
#   ./iterate.sh --batch ../doc/batch-cocoa.txt --model gemini-3.7-flash
#
#   # Multiple projects iteration
#   ./iterate.sh --batch ../doc/batch-cocoa.txt ../doc/batch-partner.txt --model gemini-3.7-flash
#
#   # Resume a previous run
#   ./iterate.sh --resume 20260130120000

cd /workspace/app
uv run python src/iteration_workflow.py "$@"
