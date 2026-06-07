"""Generate changesets between knowledge graph snapshots."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.utils import change_diff_engine, driver_extractor, source_change_detector

load_dotenv()


def generate_changeset(
    before_snapshot_id: str,
    after_snapshot_id: str,
    project: str,
    output_dir: str = "/workspace/app/output",
    model: Optional[str] = None,
) -> str:
    """Generate changeset with automatic driver extraction.

    Args:
        before_snapshot_id: Earlier snapshot ID
        after_snapshot_id: Later snapshot ID
        project: Project name
        output_dir: Base output directory
        model: Optional LLM model for driver extraction (recommended: gemini-3.0-flash-lite)

    Returns:
        Changeset ID

    Example:
        >>> changeset_id = generate_changeset("20260501110000", "20260501120000", "myproject")
        >>> print(changeset_id)
        "cs_20260501_120000"
    """
    # 1. Load snapshots and compute KG diff
    before_kg = change_diff_engine.load_knowledge(project, before_snapshot_id, output_dir)
    after_kg = change_diff_engine.load_knowledge(project, after_snapshot_id, output_dir)
    kg_diff = change_diff_engine.compute_diff(before_kg, after_kg)

    # 2. Detect source changes (Miro items, PDF changes, etc.)
    source_changes = source_change_detector.detect_source_changes(
        before_snapshot_id, after_snapshot_id, project, output_dir
    )

    # 3. Extract drivers via LLM clustering
    drivers = driver_extractor.cluster_and_extract_drivers(source_changes, kg_diff, model=model)

    # 4. Map drivers to KG changes and create change objects
    changes = _create_change_objects(kg_diff, drivers)

    # 5. Create changeset
    changeset_id = _generate_changeset_id(after_snapshot_id)
    changeset = {
        "changeSetId": changeset_id,
        "beforeSnapshotId": before_snapshot_id,
        "afterSnapshotId": after_snapshot_id,
        "timestamp": datetime.now().isoformat(),
        "changes": changes,
        "drivers": drivers,  # Store extracted drivers
        "summary": kg_diff["summary"],
    }

    # 6. Save changeset
    _save_changeset(project, changeset_id, changeset, output_dir)

    return changeset_id


def _generate_changeset_id(snapshot_id: str) -> str:
    """Generate changeset ID from snapshot ID.

    Args:
        snapshot_id: Snapshot ID (YYYYMMDDHHMMSS)

    Returns:
        Changeset ID (cs_YYYYMMDD_HHMMSS)
    """
    # Extract date and time parts
    if len(snapshot_id) >= 14:
        date_part = snapshot_id[:8]  # YYYYMMDD
        time_part = snapshot_id[8:14]  # HHMMSS
        return f"cs_{date_part}_{time_part}"
    else:
        # Fallback to UUID if snapshot_id is malformed
        return f"cs_{uuid.uuid4().hex[:16]}"


def _create_change_objects(kg_diff: dict, drivers: List[dict]) -> List[dict]:
    """Create change objects from KG diff and map drivers.

    Args:
        kg_diff: KG diff result from change_diff_engine
        drivers: Extracted drivers from LLM

    Returns:
        List of change objects
    """
    changes = []

    # Assign default driver if none extracted
    default_driver = (
        {"driverId": "default", "type": "Unknown", "description": "No driver identified", "category": "other"}
        if not drivers
        else None
    )

    # Create changes for added classes
    for cls in kg_diff.get("added_classes", []):
        change = {
            "changeId": str(uuid.uuid4()),
            "changeType": "ADD",
            "objectType": "NODE",
            "beforeNode": None,
            "afterNode": cls,
            "diff": {},
            "driver": _assign_driver_to_change(cls, drivers, default_driver),
        }
        changes.append(change)

    # Create changes for updated classes
    for update in kg_diff.get("updated_classes", []):
        change = {
            "changeId": str(uuid.uuid4()),
            "changeType": "UPDATE",
            "objectType": "NODE",
            "beforeNode": update["before"],
            "afterNode": update["after"],
            "diff": update["diff"],
            "driver": _assign_driver_to_change(update["after"], drivers, default_driver),
        }
        changes.append(change)

    # Create changes for deleted classes
    for cls in kg_diff.get("deleted_classes", []):
        change = {
            "changeId": str(uuid.uuid4()),
            "changeType": "DELETE",
            "objectType": "NODE",
            "beforeNode": cls,
            "afterNode": None,
            "diff": {},
            "driver": _assign_driver_to_change(cls, drivers, default_driver),
        }
        changes.append(change)

    # Create changes for added properties
    for prop in kg_diff.get("added_properties", []):
        change = {
            "changeId": str(uuid.uuid4()),
            "changeType": "ADD",
            "objectType": "RELATIONSHIP",
            "beforeRelationship": None,
            "afterRelationship": prop,
            "diff": {},
            "driver": _assign_driver_to_change(prop, drivers, default_driver),
        }
        changes.append(change)

    # Create changes for deleted properties
    for prop in kg_diff.get("deleted_properties", []):
        change = {
            "changeId": str(uuid.uuid4()),
            "changeType": "DELETE",
            "objectType": "RELATIONSHIP",
            "beforeRelationship": prop,
            "afterRelationship": None,
            "diff": {},
            "driver": _assign_driver_to_change(prop, drivers, default_driver),
        }
        changes.append(change)

    return changes


def _assign_driver_to_change(entity: dict, drivers: List[dict], default_driver: Optional[dict]) -> dict:
    """Assign driver to a change based on entity and available drivers.

    Simple strategy: If only 1 driver, assign to all changes.
    If multiple drivers, use first driver (LLM ordered by importance).

    Args:
        entity: KG entity (class or property)
        drivers: List of extracted drivers
        default_driver: Fallback driver if no drivers available

    Returns:
        Driver dict
    """
    if not drivers:
        return default_driver

    if len(drivers) == 1:
        return drivers[0]

    # For multiple drivers: assign first (most important)
    # Future enhancement: match by entity label, sources, etc.
    return drivers[0]


def _save_changeset(project: str, changeset_id: str, changeset: dict, output_dir: str):
    """Save changeset to disk.

    Args:
        project: Project name
        changeset_id: Changeset ID
        changeset: Changeset dict
        output_dir: Base output directory
    """
    # Create changesets directory
    changesets_dir = Path(output_dir) / project / "changesets"
    changesets_dir.mkdir(parents=True, exist_ok=True)

    # Create changeset subdirectory
    changeset_dir = changesets_dir / changeset_id
    changeset_dir.mkdir(parents=True, exist_ok=True)

    # Save changeset.json
    changeset_file = changeset_dir / "changeset.json"
    with open(changeset_file, "w", encoding="utf-8") as f:
        json.dump(changeset, f, ensure_ascii=False, indent=2)

    # Update changesets metadata index
    _update_changesets_metadata(project, changeset, output_dir)


def _update_changesets_metadata(project: str, changeset: dict, output_dir: str):
    """Update changesets/metadata.json index.

    Args:
        project: Project name
        changeset: Changeset dict
        output_dir: Base output directory
    """
    changesets_dir = Path(output_dir) / project / "changesets"
    metadata_file = changesets_dir / "metadata.json"

    # Load existing metadata or create new
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {"changesets": []}

    # Add new changeset entry
    changeset_entry = {
        "changeSetId": changeset["changeSetId"],
        "beforeSnapshotId": changeset["beforeSnapshotId"],
        "afterSnapshotId": changeset["afterSnapshotId"],
        "timestamp": changeset["timestamp"],
        "total_changes": changeset["summary"]["total_changes"],
        "drivers_count": len(changeset.get("drivers", [])),
    }

    metadata["changesets"].append(changeset_entry)

    # Save updated metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
