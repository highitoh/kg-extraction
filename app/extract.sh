#!/bin/bash

# Knowledge Extraction Wrapper Script
# Usage: ./extract.sh <pdf_path> [--project PROJECT] [--model MODEL] [--output-dir DIR]

cd /workspace/app
python src/extract_workflow.py "$@"
