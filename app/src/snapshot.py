"""
Snapshot Management CLI

Command-line interface for managing knowledge graph snapshots.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the app directory to sys.path to enable src imports
app_dir = os.path.dirname(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.utils.snapshot_repository import SnapshotRepository


def list_command(args):
    """List all snapshots for a project."""
    repo = SnapshotRepository(output_dir=args.output_dir)

    try:
        snapshots = repo.list_snapshots(args.project)

        if not snapshots:
            print(f"No snapshots found for project: {args.project}")
            return

        print(f"\nProject: {args.project}")
        print(f"Total snapshots: {len(snapshots)}\n")
        print(f"{'Snapshot ID':<16} {'Timestamp':<16} {'Trigger':<12} {'Classes':<8} {'Properties':<10}")
        print("-" * 70)

        for snapshot in snapshots:
            snapshot_id = snapshot["snapshot_id"]
            timestamp = snapshot["timestamp"]
            trigger = snapshot.get("trigger_type", "unknown")
            classes = snapshot["counts"]["total_classes"]
            properties = snapshot["counts"]["total_properties"]

            print(f"{snapshot_id:<16} {timestamp:<16} {trigger:<12} {classes:<8} {properties:<10}")

        latest_id = snapshots[0]["snapshot_id"]
        latest_classes = snapshots[0]["counts"]["total_classes"]
        latest_properties = snapshots[0]["counts"]["total_properties"]
        print(f"\nLatest: {latest_id} ({latest_classes} classes, {latest_properties} properties)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def show_command(args):
    """Show details of a specific snapshot."""
    repo = SnapshotRepository(output_dir=args.output_dir)

    try:
        snapshot = repo.get_snapshot(args.project, args.snapshot)
        metadata = snapshot["metadata"]
        knowledge = snapshot["knowledge"]

        print(f"\n=== Snapshot Details ===")
        print(f"Snapshot ID: {metadata['snapshot_id']}")
        print(f"Project: {metadata['project']}")
        print(f"Created: {metadata['created_at']}")

        print(f"\nTrigger:")
        trigger = metadata.get("trigger", {})
        # Handle both 'type' and 'triggerType' (camelCase)
        trigger_type = trigger.get("type") or trigger.get("triggerType", "unknown")
        print(f"  Type: {trigger_type}")

        # Handle different source field names
        if trigger.get("source_pdf"):
            print(f"  Source PDF: {trigger['source_pdf']}")
        elif trigger.get("sourceIdentifier"):
            print(f"  Source: {trigger['sourceIdentifier']}")

        if trigger.get("sourceName"):
            print(f"  Source Name: {trigger['sourceName']}")

        if trigger.get("extraction_result_file") or trigger.get("extractionResultFile"):
            result_file = trigger.get("extraction_result_file") or trigger.get("extractionResultFile")
            print(f"  Extraction result: {result_file}")

        print(f"\nCounts:")
        counts = metadata.get("counts", {})
        print(f"  Total classes: {counts.get('total_classes', 0)}")
        print(f"  Total properties: {counts.get('total_properties', 0)}")

        if counts.get("classes_by_type"):
            print(f"\n  Classes by type:")
            for class_type, count in counts["classes_by_type"].items():
                print(f"    {class_type}: {count}")

        if metadata.get("notes"):
            print(f"\nNotes: {metadata['notes']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def latest_command(args):
    """Show the latest snapshot."""
    repo = SnapshotRepository(output_dir=args.output_dir)

    try:
        snapshot = repo.get_latest_snapshot(args.project)
        metadata = snapshot["metadata"]

        print(f"\n=== Latest Snapshot ===")
        print(f"Snapshot ID: {metadata['snapshot_id']}")
        print(f"Project: {metadata['project']}")
        print(f"Created: {metadata['created_at']}")

        print(f"\nTrigger:")
        trigger = metadata.get("trigger", {})
        # Handle both 'type' and 'triggerType' (camelCase)
        trigger_type = trigger.get("type") or trigger.get("triggerType", "unknown")
        print(f"  Type: {trigger_type}")

        # Handle different source field names
        if trigger.get("source_pdf"):
            print(f"  Source PDF: {trigger['source_pdf']}")
        elif trigger.get("sourceIdentifier"):
            print(f"  Source: {trigger['sourceIdentifier']}")

        if trigger.get("sourceName"):
            print(f"  Source Name: {trigger['sourceName']}")

        print(f"\nCounts:")
        counts = metadata.get("counts", {})
        print(f"  Total classes: {counts.get('total_classes', 0)}")
        print(f"  Total properties: {counts.get('total_properties', 0)}")

        if counts.get("classes_by_type"):
            print(f"\n  Classes by type:")
            for class_type, count in sorted(counts["classes_by_type"].items()):
                print(f"    {class_type}: {count}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def rollback_command(args):
    """Rollback to a specific snapshot."""
    repo = SnapshotRepository(output_dir=args.output_dir)

    try:
        # Confirm rollback unless --yes flag is set
        if not args.yes:
            response = input(
                f"\nAre you sure you want to rollback project '{args.project}' to snapshot '{args.snapshot}'? (yes/no): "
            )
            if response.lower() not in ["yes", "y"]:
                print("Rollback cancelled.")
                return

        # Perform rollback
        result = repo.rollback_to_snapshot(args.project, args.snapshot, backup=args.backup)

        print(f"\n=== Rollback Complete ===")
        print(f"Project: {result['project']}")
        print(f"Snapshot: {result['snapshot_id']}")
        print(f"Timestamp: {result['rollback_timestamp']}")

        if result["backup_created"]:
            print(f"\nBackup created: {result['backup_file']}")

        print(f"\nRestored file: {result['restored_file']}")
        print("\nSuccessfully rolled back to snapshot.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during rollback: {e}")
        sys.exit(1)


def main():
    """Main entry point for snapshot CLI."""
    parser = argparse.ArgumentParser(
        description="Manage knowledge graph snapshots", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/app/output",
        help="Base output directory (default: /workspace/app/output)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all snapshots for a project")
    list_parser.add_argument("--project", required=True, help="Project name")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of a specific snapshot")
    show_parser.add_argument("--project", required=True, help="Project name")
    show_parser.add_argument("--snapshot", required=True, help="Snapshot ID")

    # Latest command
    latest_parser = subparsers.add_parser("latest", help="Show the latest snapshot")
    latest_parser.add_argument("--project", required=True, help="Project name")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a specific snapshot")
    rollback_parser.add_argument("--project", required=True, help="Project name")
    rollback_parser.add_argument("--snapshot", required=True, help="Snapshot ID to rollback to")
    rollback_parser.add_argument(
        "--backup", action="store_true", default=True, help="Create backup before rollback (default: True)"
    )
    rollback_parser.add_argument("--no-backup", dest="backup", action="store_false", help="Skip backup creation")
    rollback_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "list":
        list_command(args)
    elif args.command == "show":
        show_command(args)
    elif args.command == "latest":
        latest_command(args)
    elif args.command == "rollback":
        rollback_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
