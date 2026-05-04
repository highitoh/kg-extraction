"""
Snapshot Repository for Knowledge Graph Snapshots

Handles storage, retrieval, and management of knowledge graph snapshots.
Provides query interface and rollback functionality.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class SnapshotRepository:
    """Repository for storing and retrieving knowledge graph snapshots."""

    def __init__(self, output_dir: str = "/workspace/app/output"):
        """
        Initialize SnapshotRepository.

        Args:
            output_dir: Base output directory for all projects
        """
        self.output_dir = Path(output_dir)

    def list_snapshots(
        self,
        project: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all snapshots for a project.

        Args:
            project: Project name
            limit: Optional limit on number of snapshots to return (most recent)

        Returns:
            List of snapshot metadata summaries, sorted by timestamp (newest first)

        Raises:
            FileNotFoundError: If project or snapshots directory doesn't exist
        """
        project_dir = self.output_dir / project
        snapshots_dir = project_dir / "snapshots"
        metadata_file = snapshots_dir / "metadata.json"

        if not snapshots_dir.exists():
            raise FileNotFoundError(
                f"No snapshots directory found for project '{project}' at {snapshots_dir}"
            )

        if not metadata_file.exists():
            # No metadata file means no snapshots yet
            return []

        # Load metadata index
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        snapshots = metadata.get("snapshots", [])

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply limit if specified
        if limit is not None:
            snapshots = snapshots[:limit]

        return snapshots

    def get_snapshot(
        self,
        project: str,
        snapshot_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific snapshot.

        Args:
            project: Project name
            snapshot_id: Snapshot ID (timestamp)

        Returns:
            Dict with 'metadata' and 'knowledge' keys, or None if not found

        Raises:
            FileNotFoundError: If snapshot doesn't exist
        """
        project_dir = self.output_dir / project
        snapshot_dir = project_dir / "snapshots" / snapshot_id

        if not snapshot_dir.exists():
            raise FileNotFoundError(
                f"Snapshot '{snapshot_id}' not found for project '{project}'"
            )

        # Load snapshot metadata
        metadata_file = snapshot_dir / "snapshot.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Snapshot metadata not found: {metadata_file}"
            )

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load knowledge graph
        knowledge_file = snapshot_dir / "knowledge.json"
        if not knowledge_file.exists():
            raise FileNotFoundError(
                f"Snapshot knowledge.json not found: {knowledge_file}"
            )

        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)

        return {
            "metadata": metadata,
            "knowledge": knowledge
        }

    def get_latest_snapshot(
        self,
        project: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent snapshot for a project.

        Args:
            project: Project name

        Returns:
            Dict with 'metadata' and 'knowledge' keys, or None if no snapshots

        Raises:
            FileNotFoundError: If project has no snapshots
        """
        snapshots = self.list_snapshots(project, limit=1)

        if not snapshots:
            raise FileNotFoundError(
                f"No snapshots found for project '{project}'"
            )

        latest_snapshot_id = snapshots[0]["snapshot_id"]
        return self.get_snapshot(project, latest_snapshot_id)

    def rollback_to_snapshot(
        self,
        project: str,
        snapshot_id: str,
        backup: bool = True
    ) -> Dict[str, Any]:
        """
        Restore knowledge.json to a specific snapshot state.

        Args:
            project: Project name
            snapshot_id: Snapshot ID to restore
            backup: If True, create backup of current knowledge.json before rollback

        Returns:
            Dict with rollback result info (paths, backup location, etc.)

        Raises:
            FileNotFoundError: If snapshot doesn't exist
            IOError: If rollback fails
        """
        project_dir = self.output_dir / project
        current_knowledge_file = project_dir / "knowledge.json"
        snapshot_dir = project_dir / "snapshots" / snapshot_id
        snapshot_knowledge_file = snapshot_dir / "knowledge.json"

        # Validate snapshot exists
        if not snapshot_dir.exists():
            available_snapshots = self.list_snapshots(project)
            available_ids = [s["snapshot_id"] for s in available_snapshots]
            raise FileNotFoundError(
                f"Snapshot '{snapshot_id}' not found for project '{project}'. "
                f"Available snapshots: {', '.join(available_ids)}"
            )

        if not snapshot_knowledge_file.exists():
            raise FileNotFoundError(
                f"Snapshot knowledge.json not found: {snapshot_knowledge_file}"
            )

        result = {
            "project": project,
            "snapshot_id": snapshot_id,
            "rollback_timestamp": datetime.now().isoformat(),
            "backup_created": False,
            "backup_file": None,
            "success": False
        }

        # Create backup if requested and current knowledge.json exists
        if backup and current_knowledge_file.exists():
            backup_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file = project_dir / f"knowledge.json.backup_{backup_timestamp}"
            shutil.copy2(current_knowledge_file, backup_file)
            result["backup_created"] = True
            result["backup_file"] = str(backup_file)

        # Restore snapshot knowledge.json to current
        shutil.copy2(snapshot_knowledge_file, current_knowledge_file)
        result["success"] = True
        result["restored_file"] = str(current_knowledge_file)

        return result

    def snapshot_exists(self, project: str, snapshot_id: str) -> bool:
        """
        Check if a snapshot exists.

        Args:
            project: Project name
            snapshot_id: Snapshot ID

        Returns:
            True if snapshot exists, False otherwise
        """
        project_dir = self.output_dir / project
        snapshot_dir = project_dir / "snapshots" / snapshot_id
        return snapshot_dir.exists()

    def get_snapshot_count(self, project: str) -> int:
        """
        Get total number of snapshots for a project.

        Args:
            project: Project name

        Returns:
            Number of snapshots
        """
        try:
            snapshots = self.list_snapshots(project)
            return len(snapshots)
        except FileNotFoundError:
            return 0

    def find_latest_for_board(
        self,
        board_id: str,
        project: str
    ) -> Optional[str]:
        """
        Find the latest snapshot for a given Miro board.

        Args:
            board_id: Miro board ID (e.g., "uXjVGf-nTm0=")
            project: Project name

        Returns:
            Snapshot ID if found, None otherwise

        Example:
            >>> repo = SnapshotRepository()
            >>> snapshot_id = repo.find_latest_for_board("uXjVGf-nTm0=", "myproject")
            >>> print(snapshot_id)
            "20260503100502"
        """
        try:
            snapshots = self.list_snapshots(project)
        except FileNotFoundError:
            return None

        # Find snapshots with matching sourceIdentifier
        target_source = f"miro:{board_id}"

        # list_snapshots() returns newest first (sorted at line 66)
        for snapshot in snapshots:
            source_id = snapshot.get("source", "")
            if source_id == target_source:
                return snapshot["snapshot_id"]

        return None
