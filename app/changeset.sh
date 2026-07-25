#!/bin/bash

# Changeset Management Wrapper Script
# Usage: ./changeset.sh <command> [OPTIONS]
#
# Each extraction records its own changeset, so there is no generate command.
#
# Commands:
#   list --project PROJECT
#       List all changesets
#   show --project PROJECT --changeset ID
#       Show changeset details
#   range --project PROJECT --before ID --after ID
#       Show the changes recorded between two snapshots (aggregated)
#   validate --project PROJECT --changeset ID
#       Validate a changeset against the JSON schema
#
# Examples:
#   ./changeset.sh list --project myproject
#   ./changeset.sh show --project myproject --changeset cs_20260501_120000
#   ./changeset.sh range --project myproject --before 20260501110000 --after 20260501120000

cd /workspace/app
uv run python src/changeset.py "$@"
