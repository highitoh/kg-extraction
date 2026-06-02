#!/bin/bash

# Snapshot Management Wrapper Script
# Usage: ./snapshot.sh <command> [OPTIONS]
#
# Commands:
#   list --project PROJECT              List all snapshots
#   show --project PROJECT --snapshot ID   Show snapshot details
#   latest --project PROJECT            Show latest snapshot
#   rollback --project PROJECT --snapshot ID   Rollback to snapshot
#
# Examples:
#   ./snapshot.sh list --project cocoa
#   ./snapshot.sh show --project cocoa --snapshot 20260425120000
#   ./snapshot.sh latest --project cocoa
#   ./snapshot.sh rollback --project cocoa --snapshot 20260425120000 --backup

cd /workspace/app
uv run python src/snapshot.py "$@"
