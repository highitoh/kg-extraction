#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <project-name> [project-name ...]"
    exit 1
fi

OUTPUT_DIR="/workspace/app/output"
DELETE_COUNT=0
ERROR_COUNT=0

for PROJECT_NAME in "$@"; do
    PROJECT_DIR="$OUTPUT_DIR/$PROJECT_NAME"

    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Error: Project '$PROJECT_NAME' does not exist"
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
    fi

    rm -rf "$PROJECT_DIR"
    echo "Deleted project: $PROJECT_NAME"
    DELETE_COUNT=$((DELETE_COUNT + 1))
done

if [ "$DELETE_COUNT" -gt 1 ]; then
    echo "Deleted $DELETE_COUNT projects."
fi

if [ "$ERROR_COUNT" -gt 0 ]; then
    exit 1
fi
