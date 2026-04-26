"""
Snapshot Manager for Knowledge Graph Snapshots

Creates and manages snapshots of knowledge graphs after each extraction.
Provides version history and enables rollback capabilities.
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict


class SnapshotManager:
    """Manages creation and storage of knowledge graph snapshots."""

    def __init__(self, output_dir: str = "/workspace/app/output"):
        """
        Initialize SnapshotManager.

        Args:
            output_dir: Base output directory for all projects
        """
        self.output_dir = Path(output_dir)

    def create_snapshot(
        self,
        project: str,
        trigger_type: str = "extraction",
        source_pdf: Optional[str] = None,
        extraction_result_file: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new snapshot of the current knowledge graph.

        Args:
            project: Project name
            trigger_type: Type of trigger ("extraction", "optimization", "manual")
            source_pdf: Path to PDF that triggered this snapshot
            extraction_result_file: Path to extraction result file
            notes: Optional notes about this snapshot

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
            raise FileNotFoundError(
                f"No knowledge.json found for project '{project}' at {knowledge_file}"
            )

        # Load current knowledge graph
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)

        # Generate snapshot ID
        snapshot_id = self._generate_snapshot_id()

        # Handle potential collision (same-second extraction)
        snapshots_dir = project_dir / "snapshots"
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

        # Compute counts and statistics
        counts = self._compute_counts(knowledge)

        # Compute source PDF hash if provided
        source_pdf_hash = None
        if source_pdf and os.path.exists(source_pdf):
            source_pdf_hash = self._compute_file_hash(source_pdf)

        # Create snapshot metadata
        snapshot_metadata = {
            "snapshot_id": snapshot_id,
            "project": project,
            "timestamp": snapshot_id.split('_')[0],  # Remove microseconds if present
            "created_at": datetime.now().isoformat(),
            "trigger": {
                "type": trigger_type,
                "source_pdf": source_pdf,
                "source_pdf_hash": source_pdf_hash,
                "extraction_result_file": extraction_result_file
            },
            "counts": counts
        }

        if notes:
            snapshot_metadata["notes"] = notes

        # Save snapshot metadata
        snapshot_metadata_file = snapshot_dir / "snapshot.json"
        with open(snapshot_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_metadata, f, ensure_ascii=False, indent=2)

        # Update index metadata
        self._update_metadata_index(project, snapshot_metadata)

        return snapshot_metadata

    def _generate_snapshot_id(self) -> str:
        """
        Generate timestamp-based snapshot ID.

        Returns:
            Snapshot ID in format YYYYMMDDHHMMSS
        """
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
            "classes_by_type": dict(classes_by_type)
        }

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

    def _update_metadata_index(
        self,
        project: str,
        snapshot_metadata: Dict[str, Any]
    ) -> None:
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
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "project": project,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_snapshots": 0,
                "snapshots": []
            }

        # Add new snapshot to index
        metadata["snapshots"].append({
            "snapshot_id": snapshot_metadata["snapshot_id"],
            "timestamp": snapshot_metadata["timestamp"],
            "trigger_type": snapshot_metadata["trigger"]["type"],
            "source_pdf": snapshot_metadata["trigger"].get("source_pdf"),
            "counts": snapshot_metadata["counts"]
        })

        # Update metadata
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["total_snapshots"] = len(metadata["snapshots"])

        # Save metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
