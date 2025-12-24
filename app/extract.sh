#!/bin/bash

# Knowledge Extraction Wrapper Script
# Usage:
#   Single file: ./extract.sh <pdf_path> [OPTIONS]
#   Batch mode:  ./extract.sh --batch <batch_file> [OPTIONS]
# Options: [--project PROJECT] [--model MODEL] [--output-dir DIR] [--max-retries N]

cd /workspace/app

# Default values
BATCH_MODE=false
BATCH_FILE=""
MAX_RETRIES=1
OTHER_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch)
      BATCH_MODE=true
      BATCH_FILE="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

# Single file mode
if [ "$BATCH_MODE" = false ]; then
  python src/extract_workflow.py "${OTHER_ARGS[@]}"
  exit $?
fi

# Batch mode
if [ ! -f "$BATCH_FILE" ]; then
  echo "Error: Batch file not found: $BATCH_FILE"
  exit 1
fi

echo "=== Batch Knowledge Extraction ==="
echo "Batch file: $BATCH_FILE"
echo "Max retries: $MAX_RETRIES"
echo ""

# Get batch file directory for relative path resolution
# Convert to absolute path first
BATCH_FILE_ABS=$(cd "$(dirname "$BATCH_FILE")" && pwd)/$(basename "$BATCH_FILE")
BATCH_DIR=$(dirname "$BATCH_FILE_ABS")

# Count total files (excluding comments and blank lines)
TOTAL_FILES=$(grep -v "^#" "$BATCH_FILE" | grep -v "^[[:space:]]*$" | wc -l)
CURRENT=0
SUCCESS_COUNT=0
FAILED_COUNT=0

# Start time
START_TIME=$(date +%s)

# Process each line in batch file
while IFS= read -r line || [[ -n "$line" ]]; do
  # Remove carriage return (Windows line endings)
  line="${line%$'\r'}"

  # Skip comments and blank lines
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" =~ ^[[:space:]]*$ ]] && continue

  CURRENT=$((CURRENT + 1))

  # Resolve path (relative to batch file directory)
  if [[ "$line" = /* ]]; then
    # Absolute path
    PDF_PATH="$line"
  else
    # Relative path
    PDF_PATH="$BATCH_DIR/$line"
  fi

  echo "[$CURRENT/$TOTAL_FILES] Processing: $line"

  # Check if file exists
  if [ ! -f "$PDF_PATH" ]; then
    echo "  ✗ File not found - skipping"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    echo ""
    continue
  fi

  # Retry logic
  ATTEMPT=0
  SUCCESS=false

  while [ $ATTEMPT -lt $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
    ATTEMPT=$((ATTEMPT + 1))

    if [ $ATTEMPT -gt 1 ]; then
      echo "  ⟳ Retrying (attempt $ATTEMPT/$MAX_RETRIES)..."
      sleep 5
    fi

    # Run extraction
    FILE_START=$(date +%s)
    if python src/extract_workflow.py "$PDF_PATH" "${OTHER_ARGS[@]}"; then
      FILE_END=$(date +%s)
      FILE_TIME=$((FILE_END - FILE_START))
      echo "  ✓ Extraction complete (${FILE_TIME}s)"
      SUCCESS=true
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      if [ $ATTEMPT -eq $MAX_RETRIES ]; then
        echo "  ✗ Extraction failed after $MAX_RETRIES attempts"
        FAILED_COUNT=$((FAILED_COUNT + 1))
      fi
    fi
  done

  echo ""
done < "$BATCH_FILE"

# End time and summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=== Batch Extraction Complete ==="
echo "Total files: $TOTAL_FILES"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Total time: ${TOTAL_TIME}s"
echo ""

# Exit with error if any files failed
[ $FAILED_COUNT -eq 0 ] && exit 0 || exit 1
