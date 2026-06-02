#!/bin/bash
#
# Changeset Management Script
# Manage changesets between knowledge graph snapshots
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
PROJECT="default"
OUTPUT_DIR="/workspace/app/output"
MODEL=""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
Usage: ./changeset.sh <command> [options]

Commands:
    generate    Generate changeset between two snapshots
    list        List all changesets for a project
    show        Show details of a specific changeset

Options:
    --project PROJECT       Project name (default: default)
    --before SNAPSHOT_ID    Before snapshot ID (required for generate)
    --after SNAPSHOT_ID     After snapshot ID (required for generate)
    --changeset ID          Changeset ID (required for show)
    --model MODEL           LLM model for driver extraction (default: use env default)
    --output-dir DIR        Output directory (default: /workspace/app/output)
    --help                  Show this help message

Examples:
    # Generate changeset between two snapshots
    ./changeset.sh generate --project myproject --before 20260501110000 --after 20260501120000

    # Generate with specific model
    ./changeset.sh generate --project myproject --before 20260501110000 --after 20260501120000 --model gemini-3.0-flash-lite

    # List all changesets
    ./changeset.sh list --project myproject

    # Show specific changeset
    ./changeset.sh show --project myproject --changeset cs_20260501_120000
EOF
}

# Parse arguments
COMMAND=""
BEFORE_SNAPSHOT=""
AFTER_SNAPSHOT=""
CHANGESET_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        generate|list|show)
            COMMAND=$1
            shift
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --before)
            BEFORE_SNAPSHOT="$2"
            shift 2
            ;;
        --after)
            AFTER_SNAPSHOT="$2"
            shift 2
            ;;
        --changeset)
            CHANGESET_ID="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate command
if [ -z "$COMMAND" ]; then
    echo -e "${RED}Error: No command specified${NC}"
    show_help
    exit 1
fi

# Execute command
case $COMMAND in
    generate)
        if [ -z "$BEFORE_SNAPSHOT" ] || [ -z "$AFTER_SNAPSHOT" ]; then
            echo -e "${RED}Error: --before and --after are required for generate command${NC}"
            exit 1
        fi

        echo -e "${GREEN}Generating changeset: $BEFORE_SNAPSHOT → $AFTER_SNAPSHOT${NC}"
        echo "Project: $PROJECT"
        if [ -n "$MODEL" ]; then
            echo "Model: $MODEL"
        fi
        echo ""

        # Build Python command
        PYTHON_CMD="import sys; sys.path.insert(0, '.'); \
from src.utils.changeset_generator import generate_changeset; \
changeset_id = generate_changeset('$BEFORE_SNAPSHOT', '$AFTER_SNAPSHOT', '$PROJECT', '$OUTPUT_DIR'"

        if [ -n "$MODEL" ]; then
            PYTHON_CMD="${PYTHON_CMD}, model='$MODEL'"
        fi

        PYTHON_CMD="${PYTHON_CMD}); \
print(f'\\n✓ Changeset created: {changeset_id}')"

        uv run python -c "$PYTHON_CMD"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Changeset generation completed successfully${NC}"
        else
            echo -e "${RED}Changeset generation failed${NC}"
            exit 1
        fi
        ;;

    list)
        echo -e "${GREEN}Listing changesets for project: $PROJECT${NC}"
        echo ""

        METADATA_FILE="$OUTPUT_DIR/$PROJECT/changesets/metadata.json"

        if [ ! -f "$METADATA_FILE" ]; then
            echo -e "${YELLOW}No changesets found for project '$PROJECT'${NC}"
            exit 0
        fi

        uv run python -c "
import json
import sys
sys.path.insert(0, '.')

with open('$METADATA_FILE', 'r') as f:
    metadata = json.load(f)

changesets = metadata.get('changesets', [])

if not changesets:
    print('${YELLOW}No changesets found${NC}')
    sys.exit(0)

print(f'Found {len(changesets)} changeset(s):\n')
print(f'{'ID':<25} {'Before':<16} {'After':<16} {'Changes':<10} {'Drivers':<10}')
print('-' * 90)

for cs in changesets:
    cs_id = cs.get('changeSetId', 'N/A')
    before_id = cs.get('beforeSnapshotId', 'N/A')
    after_id = cs.get('afterSnapshotId', 'N/A')
    total_changes = cs.get('total_changes', 0)
    drivers_count = cs.get('drivers_count', 0)

    print(f'{cs_id:<25} {before_id:<16} {after_id:<16} {total_changes:<10} {drivers_count:<10}')
"
        ;;

    show)
        if [ -z "$CHANGESET_ID" ]; then
            echo -e "${RED}Error: --changeset is required for show command${NC}"
            exit 1
        fi

        CHANGESET_FILE="$OUTPUT_DIR/$PROJECT/changesets/$CHANGESET_ID/changeset.json"

        if [ ! -f "$CHANGESET_FILE" ]; then
            echo -e "${RED}Error: Changeset not found: $CHANGESET_ID${NC}"
            exit 1
        fi

        echo -e "${GREEN}Changeset: $CHANGESET_ID${NC}"
        echo ""

        uv run python -c "
import json
import sys

with open('$CHANGESET_FILE', 'r') as f:
    changeset = json.load(f)

print(f'Before Snapshot: {changeset.get(\"beforeSnapshotId\")}')
print(f'After Snapshot:  {changeset.get(\"afterSnapshotId\")}')
print(f'Timestamp:       {changeset.get(\"timestamp\")}')
print()

summary = changeset.get('summary', {})
print('Summary:')
print(f'  Total changes:     {summary.get(\"total_changes\", 0)}')
print(f'  Added classes:     {summary.get(\"added_classes\", 0)}')
print(f'  Updated classes:   {summary.get(\"updated_classes\", 0)}')
print(f'  Deleted classes:   {summary.get(\"deleted_classes\", 0)}')
print(f'  Added properties:  {summary.get(\"added_properties\", 0)}')
print(f'  Deleted properties:{summary.get(\"deleted_properties\", 0)}')
print()

drivers = changeset.get('drivers', [])
if drivers:
    print(f'Drivers ({len(drivers)}):')
    for driver in drivers:
        driver_type = driver.get('type', 'Unknown')
        category = driver.get('category', 'other')
        description = driver.get('description', '')
        print(f'  - {driver_type} ({category})')
        if description:
            print(f'    {description}')
    print()

changes = changeset.get('changes', [])
print(f'Changes ({len(changes)}):')
for i, change in enumerate(changes[:20], 1):  # Show first 20
    change_type = change.get('changeType')
    obj_type = change.get('objectType')

    if obj_type == 'NODE':
        node = change.get('afterNode') or change.get('beforeNode')
        label = node.get('label', 'Unknown') if node else 'Unknown'
        print(f'  {i}. {change_type} {obj_type}: {label}')
    else:
        rel = change.get('afterRelationship') or change.get('beforeRelationship')
        src_label = rel.get('src_label', 'Unknown') if rel else 'Unknown'
        dest_label = rel.get('dest_label', 'Unknown') if rel else 'Unknown'
        print(f'  {i}. {change_type} {obj_type}: {src_label} → {dest_label}')

if len(changes) > 20:
    print(f'  ... and {len(changes) - 20} more changes')
"
        ;;

    *)
        echo -e "${RED}Error: Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
