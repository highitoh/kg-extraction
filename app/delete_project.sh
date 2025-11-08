#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <project-name>"
    exit 1
fi

PROJECT_NAME=$1
PROJECT_DIR="/workspace/app/output/$PROJECT_NAME"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project '$PROJECT_NAME' does not exist"
    exit 1
fi

rm -rf "$PROJECT_DIR"
echo "Deleted project: $PROJECT_NAME"
