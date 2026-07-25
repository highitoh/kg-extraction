"""
Changeset Management CLI

Command-line interface for reviewing changesets. A changeset is recorded
automatically by each extraction; this CLI lists them, shows one, and
aggregates the ones spanning a range of snapshots. Changesets are
metamodel-driven instance graphs (see src/utils/changeset_generator.py).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the app directory to sys.path to enable src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

APP_DIR = Path(__file__).parent.parent
OUTPUT_SCHEMA_PATH = APP_DIR / "schemas" / "changeset-generator" / "output.schema.json"


def _local_name(iri: str) -> str:
    """Extract local name from an IRI."""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]


def _load_changeset(args) -> dict:
    """Load a changeset.json for the CLI commands that read one."""
    changeset_file = Path(args.output_dir) / args.project / "changesets" / args.changeset / "changeset.json"
    if not changeset_file.exists():
        print(f"Error: changeset not found at {changeset_file}")
        sys.exit(1)
    with open(changeset_file, "r", encoding="utf-8") as f:
        return json.load(f)


def range_command(args):
    """Show the combined changes recorded between two snapshots."""
    from src.utils.changeset_generator import aggregate_changesets

    try:
        changeset = aggregate_changesets(
            before_snapshot_id=args.before,
            after_snapshot_id=args.after,
            project=args.project,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not changeset["aggregatedFrom"]:
        print(f"No changesets recorded between {args.before} and {args.after}")
        return

    print(f"\nAggregated from {len(changeset['aggregatedFrom'])} changesets: {', '.join(changeset['aggregatedFrom'])}")
    _print_changeset(changeset)


def list_command(args):
    """List all changesets for a project."""
    metadata_file = Path(args.output_dir) / args.project / "changesets" / "metadata.json"

    if not metadata_file.exists():
        print(f"No changesets found for project: {args.project}")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    changesets = metadata.get("changesets", [])
    if not changesets:
        print(f"No changesets found for project: {args.project}")
        return

    print(f"\nProject: {args.project}")
    print(f"Total changesets: {len(changesets)}\n")
    print(f"{'Changeset ID':<22} {'Before':<16} {'After':<16} {'Changes':<8} {'Drivers':<8}")
    print("-" * 76)

    for entry in changesets:
        print(
            f"{entry.get('changeSetId', ''):<22} "
            f"{entry.get('beforeSnapshotId', ''):<16} "
            f"{entry.get('afterSnapshotId', ''):<16} "
            f"{entry.get('total_changes', 0):<8} "
            f"{entry.get('drivers_count', 0):<8}"
        )


def show_command(args):
    """Show details of a specific changeset."""
    _print_changeset(_load_changeset(args))


def _print_changeset(changeset: dict):
    """Render a changeset (single or aggregated) for the terminal."""
    instances = changeset.get("classes", [])
    links = changeset.get("properties", [])
    instance_by_id = {inst.get("id"): inst for inst in instances}
    links_by_src = {}
    for link in links:
        links_by_src.setdefault(link.get("src_id"), []).append(link)

    print("\n=== Changeset Details ===")
    print(f"Changeset ID: {changeset.get('changeSetId', '')}")
    print(f"Before snapshot: {changeset.get('beforeSnapshotId', '')}")
    print(f"After snapshot:  {changeset.get('afterSnapshotId', '')}")
    print(f"Timestamp: {changeset.get('timestamp', '')}")

    print("\nSummary:")
    for key, value in changeset.get("summary", {}).items():
        print(f"  {key}: {value}")

    print("\nInstances by type:")
    counts = {}
    for inst in instances:
        counts[_local_name(inst.get("class_iri", ""))] = counts.get(_local_name(inst.get("class_iri", "")), 0) + 1
    for name in sorted(counts):
        print(f"  {name}: {counts[name]}")

    # Change instances carry change_type in their attributes
    changes = [inst for inst in instances if (inst.get("attributes") or {}).get("change_type")]
    if changes:
        print("\nChanges:")
        for change in changes:
            print(f"  - {change.get('label', '')}")
            for link in links_by_src.get(change.get("id"), []):
                dest = instance_by_id.get(link.get("dest_id"), {})
                print(f"      {_local_name(link.get('property_iri', ''))}: {dest.get('label', link.get('dest_id'))}")

    # LLM-extracted instances carry a review judgment
    extracted = [inst for inst in instances if inst.get("judgment")]
    if extracted:
        print("\nExtracted instances (LLM):")
        for inst in extracted:
            print(f"  - [{_local_name(inst.get('class_iri', ''))}] {inst.get('label', '')} ({inst.get('judgment')})")
            for source in inst.get("sources", [])[:2]:
                print(f"      source: {source[:100]}")


def validate_command(args):
    """Validate a changeset against the JSON schema and check link integrity."""
    import jsonschema

    changeset = _load_changeset(args)

    with open(OUTPUT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    errors = []
    validator = jsonschema.Draft202012Validator(schema)
    for error in validator.iter_errors(changeset):
        path = "/".join(str(p) for p in error.absolute_path)
        errors.append(f"schema: {path}: {error.message}")

    # Referential integrity: every link endpoint must be a known instance
    known_ids = {inst.get("id") for inst in changeset.get("classes", [])}
    for link in changeset.get("properties", []):
        for endpoint in ("src_id", "dest_id"):
            if link.get(endpoint) not in known_ids:
                errors.append(f"integrity: link {link.get('id', '')}: {endpoint} '{link.get(endpoint)}' not found")

    if errors:
        print(f"Validation FAILED ({len(errors)} errors):")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    print(f"Validation OK: {len(changeset.get('classes', []))} instances, {len(changeset.get('properties', []))} links")


def main():
    """Main entry point for changeset CLI."""
    parser = argparse.ArgumentParser(
        description="Manage changesets between knowledge graph snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/app/output",
        help="Base output directory (default: /workspace/app/output)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    range_parser = subparsers.add_parser("range", help="Show the changes recorded between two snapshots")
    range_parser.add_argument("--project", required=True, help="Project name")
    range_parser.add_argument("--before", required=True, help="Snapshot the range starts from (exclusive)")
    range_parser.add_argument("--after", required=True, help="Snapshot the range ends at (inclusive)")

    list_parser = subparsers.add_parser("list", help="List all changesets for a project")
    list_parser.add_argument("--project", required=True, help="Project name")

    show_parser = subparsers.add_parser("show", help="Show details of a specific changeset")
    show_parser.add_argument("--project", required=True, help="Project name")
    show_parser.add_argument("--changeset", required=True, help="Changeset ID")

    validate_parser = subparsers.add_parser("validate", help="Validate a changeset against the JSON schema")
    validate_parser.add_argument("--project", required=True, help="Project name")
    validate_parser.add_argument("--changeset", required=True, help="Changeset ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "range": range_command,
        "list": list_command,
        "show": show_command,
        "validate": validate_command,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
