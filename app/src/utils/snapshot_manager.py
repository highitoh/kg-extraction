"""
Snapshot Manager for Knowledge Graph Snapshots

Creates and manages snapshots of knowledge graphs after each extraction.
Provides version history and enables rollback capabilities.
"""

import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SnapshotManager:
    """Manages creation and storage of knowledge graph snapshots."""

    def __init__(self, output_dir: str = "/workspace/app/output", timestamp: Optional[str] = None):
        """
        Initialize SnapshotManager.

        Args:
            output_dir: Base output directory for all projects
            timestamp: Optional manual timestamp in YYYYMMDD format (converted to YYYYMMDD000000)
        """
        self.output_dir = Path(output_dir)
        self.manual_timestamp = self._normalize_timestamp(timestamp) if timestamp else None

    def create_snapshot(
        self,
        project: str,
        trigger_type: str = "extraction",
        source_pdf: Optional[str] = None,
        extraction_result_file: Optional[str] = None,
        notes: Optional[str] = None,
        raw_source_data: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Create a new snapshot of the current knowledge graph.

        Args:
            project: Project name
            trigger_type: Type of trigger ("extraction", "optimization", "manual")
            source_pdf: Path to PDF that triggered this snapshot
            extraction_result_file: Path to extraction result file
            notes: Optional notes about this snapshot
            raw_source_data: Optional raw source data (e.g., Miro board JSON) to store

        Returns:
            Snapshot metadata dict with snapshot_id, timestamp, counts, etc.

        Raises:
            FileNotFoundError: If knowledge.json doesn't exist
            IOError: If snapshot creation fails
        """
        # Verify knowledge.json exists
        project_dir = self.output_dir / project
        knowledge_file = project_dir / "knowledge.json"

        if not knowledge_file.exists():
            raise FileNotFoundError(f"No knowledge.json found for project '{project}' at {knowledge_file}")

        # Load current knowledge graph
        with open(knowledge_file, "r", encoding="utf-8") as f:
            knowledge = json.load(f)

        # Get latest snapshot timestamp for validation
        snapshots_dir = project_dir / "snapshots"
        latest_snapshot = self._get_latest_snapshot_timestamp(snapshots_dir)

        # Generate snapshot ID
        snapshot_id = self._generate_snapshot_id()

        # Validate time-ordering constraint
        if latest_snapshot and snapshot_id <= latest_snapshot:
            raise ValueError(
                f"Timestamp {snapshot_id} must be later than latest snapshot {latest_snapshot}. "
                f"Time-ordered extraction is required to maintain snapshot consistency."
            )

        # Handle potential collision (same-second extraction)
        snapshot_dir = snapshots_dir / snapshot_id

        if snapshot_dir.exists():
            # Append microseconds to avoid collision
            snapshot_id = f"{snapshot_id}_{datetime.now().strftime('%f')}"
            snapshot_dir = snapshots_dir / snapshot_id

        # Create snapshot directory
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy knowledge.json to snapshot directory
        snapshot_knowledge_file = snapshot_dir / "knowledge.json"
        shutil.copy2(knowledge_file, snapshot_knowledge_file)

        # Store raw source data if provided (e.g., Miro board JSON)
        if raw_source_data:
            raw_data_file = snapshot_dir / "source_raw.json"
            with open(raw_data_file, "w", encoding="utf-8") as f:
                json.dump(raw_source_data, f, ensure_ascii=False, indent=2)

        # Compute counts and statistics
        counts = self._compute_counts(knowledge)

        # Build extraction trigger metadata (what caused this snapshot to be created)
        # NOTE: "trigger" here means "extraction trigger" (technical), NOT "change driver" (business)
        extraction_trigger = {"triggerType": trigger_type}  # "extraction" or "manual"

        # Handle PDF source
        if source_pdf:
            extraction_trigger["sourceIdentifier"] = f"pdf:{source_pdf}"
            extraction_trigger["sourceName"] = os.path.basename(source_pdf)
            extraction_trigger["sourceMetadata"] = {}
            if os.path.exists(source_pdf):
                extraction_trigger["sourceMetadata"]["hash"] = self._compute_file_hash(source_pdf)

        # Handle Miro board source
        if raw_source_data and "id" in raw_source_data:  # Detect Miro board structure
            extraction_trigger["sourceIdentifier"] = f"miro:{raw_source_data.get('id')}"
            extraction_trigger["sourceName"] = raw_source_data.get("name", "Unnamed Board")
            extraction_trigger["sourceMetadata"] = {
                "itemCount": len(raw_source_data.get("items", [])),
                "hash": hashlib.sha256(json.dumps(raw_source_data, sort_keys=True).encode()).hexdigest()[:16],
            }

        # Handle GitHub repository source
        if raw_source_data and "owner" in raw_source_data and "repo" in raw_source_data:  # Detect GitHub structure
            owner = raw_source_data.get("owner")
            repo = raw_source_data.get("repo")
            extraction_trigger["sourceIdentifier"] = f"github:{owner}:{repo}"
            extraction_trigger["sourceName"] = f"{owner}/{repo}"
            extraction_trigger["sourceMetadata"] = {
                "owner": owner,
                "repo": repo,
                "ref": raw_source_data.get("ref", "main"),
                "commit_sha": raw_source_data.get("commit_sha", ""),
                "tag": raw_source_data.get("tag"),
                "branch": raw_source_data.get("branch"),
                "profile": raw_source_data.get("profile", "documentation"),
                "url": raw_source_data.get("url", f"https://github.com/{owner}/{repo}"),
                "file_count": raw_source_data.get("file_count", 0),
                "hash": hashlib.sha256(json.dumps(raw_source_data, sort_keys=True).encode()).hexdigest()[:16],
            }

        if extraction_result_file:
            extraction_trigger["extractionResultFile"] = extraction_result_file

        # Create snapshot metadata
        snapshot_metadata = {
            "snapshot_id": snapshot_id,
            "project": project,
            "timestamp": snapshot_id.split("_")[0],  # Remove microseconds if present
            "created_at": datetime.now().isoformat(),
            "trigger": extraction_trigger,  # Keep existing field name for compatibility
            "counts": counts,
        }

        if notes:
            snapshot_metadata["notes"] = notes

        # Save snapshot metadata
        snapshot_metadata_file = snapshot_dir / "snapshot.json"
        with open(snapshot_metadata_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_metadata, f, ensure_ascii=False, indent=2)

        # Update index metadata
        self._update_metadata_index(project, snapshot_metadata)

        return snapshot_metadata

    def _normalize_timestamp(self, timestamp: str) -> str:
        """
        Normalize timestamp from YYYYMMDD to YYYYMMDD000000.

        Args:
            timestamp: Timestamp in YYYYMMDD format

        Returns:
            Normalized timestamp in YYYYMMDD000000 format

        Raises:
            ValueError: If timestamp format is invalid
        """
        if len(timestamp) != 8:
            raise ValueError(f"Timestamp must be in YYYYMMDD format (8 digits), got: {timestamp}")

        try:
            # Validate it's a valid date
            datetime.strptime(timestamp, "%Y%m%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {timestamp}. Expected YYYYMMDD.") from e

        return f"{timestamp}000000"

    def _generate_snapshot_id(self) -> str:
        """
        Generate timestamp-based snapshot ID.

        Returns:
            Snapshot ID in format YYYYMMDDHHMMSS
        """
        if self.manual_timestamp:
            return self.manual_timestamp
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def _compute_counts(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count classes and properties by type.

        Args:
            knowledge: Knowledge graph dict with 'classes' and 'properties'

        Returns:
            Dict with total counts and breakdowns by type
        """
        classes = knowledge.get("classes", [])
        properties = knowledge.get("properties", [])

        # Count classes by type
        classes_by_type = defaultdict(int)
        for cls in classes:
            # Extract type from class_iri (e.g., "#stakeholder" -> "stakeholder")
            class_iri = cls.get("class_iri", "")
            if "#" in class_iri:
                class_type = class_iri.split("#")[-1]
                classes_by_type[class_type] += 1

        return {
            "total_classes": len(classes),
            "total_properties": len(properties),
            "classes_by_type": dict(classes_by_type),
        }

    def _get_latest_snapshot_timestamp(self, snapshots_dir: Path) -> Optional[str]:
        """
        Get the timestamp of the latest snapshot.

        Args:
            snapshots_dir: Path to snapshots directory

        Returns:
            Latest snapshot timestamp (YYYYMMDDHHMMSS) or None if no snapshots exist
        """
        if not snapshots_dir.exists():
            return None

        # Get all snapshot directories (excluding metadata.json)
        snapshot_dirs = [d.name for d in snapshots_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]

        if not snapshot_dirs:
            return None

        # Return the maximum timestamp (lexicographic ordering works for YYYYMMDDHHMMSS)
        # Handle timestamps with microseconds (format: YYYYMMDDHHMMSS_microseconds)
        timestamps = [s.split("_")[0] for s in snapshot_dirs]
        return max(timestamps)

    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA256 hash (first 16 characters)
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]

    def _update_metadata_index(self, project: str, snapshot_metadata: Dict[str, Any]) -> None:
        """
        Update the central metadata index file.

        Args:
            project: Project name
            snapshot_metadata: Metadata for the new snapshot
        """
        project_dir = self.output_dir / project
        snapshots_dir = project_dir / "snapshots"
        metadata_file = snapshots_dir / "metadata.json"

        # Load or initialize metadata
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "project": project,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_snapshots": 0,
                "snapshots": [],
            }

        # Add new snapshot to index
        trigger = snapshot_metadata["trigger"]
        snapshot_entry = {
            "snapshot_id": snapshot_metadata["snapshot_id"],
            "timestamp": snapshot_metadata["timestamp"],
            "trigger_type": trigger.get("triggerType", trigger.get("type", "unknown")),
            "counts": snapshot_metadata["counts"],
        }

        # Add source information based on available fields
        if "sourceIdentifier" in trigger:
            snapshot_entry["source"] = trigger["sourceIdentifier"]
            snapshot_entry["source_name"] = trigger.get("sourceName", "")
        elif "source_pdf" in trigger:
            snapshot_entry["source_pdf"] = trigger["source_pdf"]

        metadata["snapshots"].append(snapshot_entry)

        # Update metadata
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["total_snapshots"] = len(metadata["snapshots"])

        # Save metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
