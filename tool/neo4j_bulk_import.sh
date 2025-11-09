#!/bin/bash
#
# RUN THIS SHELL FROM WSL TERMINAL!
#
# Copy nodes.csv and relationships.csv from specified project to neo4j/import directory
#
# Usage:
#   ./neo4j_bulk_import.sh [PROJECT_NAME]
#
# Arguments:
#   PROJECT_NAME    Project name (default: default)

# Change to script directory
cd "$(dirname "$0")"

# Get project name from first argument, default to "default"
PROJECT=${1:-default}
PROJECT_DIR="../app/output/${PROJECT}"

# Validate project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Error: Project directory '${PROJECT_DIR}' does not exist"
    echo ""
    echo "Available projects:"
    ls -1 ../app/output/ 2>/dev/null || echo "  (no projects found)"
    exit 1
fi

# Validate CSV files exist
if [ ! -f "${PROJECT_DIR}/nodes.csv" ]; then
    echo "Error: nodes.csv not found in ${PROJECT_DIR}"
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/relationships.csv" ]; then
    echo "Error: relationships.csv not found in ${PROJECT_DIR}"
    exit 1
fi

# Stop Neo4j container
echo "Stopping Neo4j container..."
docker stop neo4j_container

# Clean up existing database
echo "Removing existing neo4j database..."
rm -rf ../neo4j/data/databases/neo4j
rm -rf ../neo4j/data/transactions/neo4j

# Copy CSV files to import directory
echo "Copying CSV files to Neo4j import directory..."
cp -p "${PROJECT_DIR}/nodes.csv" ../neo4j/import/
cp -p "${PROJECT_DIR}/relationships.csv" ../neo4j/import/

# Import data to Neo4j
echo "Importing data to Neo4j..."
docker run --rm \
    -v "$(pwd)/../neo4j/data:/data" \
    -v "$(pwd)/../neo4j/import:/import" \
    neo4j:5 \
    neo4j-admin database import full \
    --overwrite-destination=true \
    --multiline-fields=true \
    --nodes=/import/nodes.csv \
    --relationships=/import/relationships.csv

# Start Neo4j container
echo "Starting Neo4j container..."
docker start neo4j_container

echo "Import completed successfully!"
