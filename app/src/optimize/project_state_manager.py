"""
Project State Manager - Manages project state for re-extraction while preserving optimization artifacts.
"""

import glob
import re
import shutil
from pathlib import Path
from typing import Any, Dict


class ProjectStateManager:
    """Manages project state for iterative extraction workflows."""

    def __init__(self, output_dir: str = "/workspace/app/output"):
        """
        Initialize the ProjectStateManager.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)

    @staticmethod
    def extract_project_from_batch(batch_file: str) -> str:
        """
        Extract project name from batch file name.

        Batch files follow the pattern: batch-{project}.txt
        Example: /workspace/doc/batch-cocoa.txt -> cocoa

        Args:
            batch_file: Path to the batch file

        Returns:
            Project name extracted from filename

        Raises:
            ValueError: If filename doesn't match expected pattern
        """
        filename = Path(batch_file).stem  # Get filename without extension
        match = re.match(r"^batch-(.+)$", filename)

        if not match:
            raise ValueError(
                f"Batch file name doesn't match expected pattern 'batch-{{project}}.txt': {batch_file}\n"
                f"Got filename: {filename}"
            )

        return match.group(1)

    def get_project_dir(self, project: str) -> Path:
        """Get the project directory path."""
        return self.output_dir / project

    def get_project_status(self, project: str) -> Dict[str, Any]:
        """
        Get the current status of a project.

        Args:
            project: Project name

        Returns:
            Dictionary with status information
        """
        project_dir = self.get_project_dir(project)

        status = {
            "project": project,
            "exists": project_dir.exists(),
            "has_knowledge": False,
            "has_extraction_result": False,
            "has_validator_result": False,
            "has_reward_calculation": False,
            "knowledge_file": None,
            "extraction_result_files": [],
            "validator_result_files": [],
            "reward_calculation_files": [],
        }

        if not project_dir.exists():
            return status

        # Check for knowledge.json
        knowledge_file = project_dir / "knowledge.json"
        if knowledge_file.exists():
            status["has_knowledge"] = True
            status["knowledge_file"] = str(knowledge_file)

        # Check for extraction results
        extraction_files = sorted(glob.glob(str(project_dir / "extraction_result_*.json")), reverse=True)
        if extraction_files:
            status["has_extraction_result"] = True
            status["extraction_result_files"] = extraction_files

        # Check for validator results
        validator_files = sorted(glob.glob(str(project_dir / "validator_result_*.json")), reverse=True)
        if validator_files:
            status["has_validator_result"] = True
            status["validator_result_files"] = validator_files

        # Check for reward calculations
        reward_files = sorted(glob.glob(str(project_dir / "reward_calculation_*.json")), reverse=True)
        if reward_files:
            status["has_reward_calculation"] = True
            status["reward_calculation_files"] = reward_files

        return status

    def reset_for_extraction(self, project: str) -> Dict[str, Any]:
        """
        Reset project for fresh extraction.

        This clears knowledge.json to allow re-extraction while keeping
        evaluation and optimization artifacts.

        Args:
            project: Project name

        Returns:
            Dictionary with reset information
        """
        project_dir = self.get_project_dir(project)
        result = {"project": project, "knowledge_cleared": False, "message": ""}

        if not project_dir.exists():
            result["message"] = f"Project directory does not exist: {project_dir}"
            return result

        # Clear knowledge.json
        knowledge_file = project_dir / "knowledge.json"
        if knowledge_file.exists():
            knowledge_file.unlink()
            result["knowledge_cleared"] = True
            result["message"] = f"Cleared knowledge.json for project: {project}"
        else:
            result["message"] = f"No knowledge.json to clear for project: {project}"

        return result

    def archive_iteration_artifacts(self, run_id: str, iteration: int, project: str) -> Dict[str, Any]:
        """
        Archive current project artifacts to iterations directory.

        Copies (not moves) current files to:
        /workspace/app/output/iterations/{run_id}/{project}-{iteration}/

        Args:
            run_id: Unique identifier for this iteration run
            iteration: Iteration number (1-based)
            project: Project name

        Returns:
            Dictionary with archive information
        """
        project_dir = self.get_project_dir(project)
        archive_dir = self.output_dir / "iterations" / run_id / f"{project}-{iteration}"

        result = {
            "project": project,
            "iteration": iteration,
            "archive_dir": str(archive_dir),
            "files_archived": [],
            "success": False,
        }

        if not project_dir.exists():
            result["error"] = f"Project directory does not exist: {project_dir}"
            return result

        # Create archive directory
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Files to archive
        files_to_archive = [
            "knowledge.json",
            "knowledge_graph_*.ttl",
            "nodes.csv",
            "relationships.csv",
            "extraction_result_*.json",
            "validator_result_*.json",
            "reward_calculation_*.json",
        ]

        for pattern in files_to_archive:
            if "*" in pattern:
                # Glob pattern
                matched_files = glob.glob(str(project_dir / pattern))
                for src_file in matched_files:
                    dest_file = archive_dir / Path(src_file).name
                    shutil.copy2(src_file, dest_file)
                    result["files_archived"].append(Path(src_file).name)
            else:
                # Exact filename
                src_file = project_dir / pattern
                if src_file.exists():
                    dest_file = archive_dir / pattern
                    shutil.copy2(src_file, dest_file)
                    result["files_archived"].append(pattern)

        result["success"] = True
        return result

    def archive_optimize_artifacts(self, run_id: str, iteration: int) -> Dict[str, Any]:
        """
        Archive optimization artifacts to iterations directory.

        Copies (not moves) current files from output/optimize/ to:
        /workspace/app/output/iterations/{run_id}/optimize-{iteration}/

        Args:
            run_id: Unique identifier for this iteration run
            iteration: Iteration number (1-based)

        Returns:
            Dictionary with archive information
        """
        optimize_dir = self.output_dir / "optimize"
        archive_dir = self.output_dir / "iterations" / run_id / f"optimize-{iteration}"

        result = {"iteration": iteration, "archive_dir": str(archive_dir), "files_archived": [], "success": False}

        if not optimize_dir.exists():
            result["error"] = f"Optimize directory does not exist: {optimize_dir}"
            return result

        # Create archive directory
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Files to archive
        files_to_archive = [
            "validator_result_*.json",
            "issue_classification_*.json",
            "issue_category_*.json",
            "strategy_plan_*.json",
            "action_proposals_*.json",
            "action_application_*.json",
        ]

        for pattern in files_to_archive:
            matched_files = glob.glob(str(optimize_dir / pattern))
            for src_file in matched_files:
                dest_file = archive_dir / Path(src_file).name
                shutil.copy2(src_file, dest_file)
                result["files_archived"].append(Path(src_file).name)

        result["success"] = True
        return result

    def cleanup_optimize_artifacts(self, keep_last_n: int = 0) -> Dict[str, Any]:
        """
        Remove old optimize workflow artifacts, keeping only the most recent ones.

        Args:
            keep_last_n: Number of most recent files to keep for each artifact type.
                        Set to 0 to remove all matched files.

        Returns:
            Dictionary with cleanup results.
        """
        if keep_last_n < 0:
            raise ValueError(f"keep_last_n must be >= 0, got {keep_last_n}")

        optimize_dir = self.output_dir / "optimize"
        result = {
            "optimize_dir": str(optimize_dir),
            "keep_last_n": keep_last_n,
            "files_removed": [],
            "files_kept": [],
            "errors": [],
            "no_op": False,
            "success": False,
        }

        # No optimize directory is a valid no-op state for fresh runs.
        if not optimize_dir.exists():
            result["no_op"] = True
            result["success"] = True
            result["message"] = f"Optimize directory does not exist: {optimize_dir}"
            return result

        patterns = [
            "validator_result_*.json",
            "issue_classification_*.json",
            "issue_category_*.json",
            "strategy_plan_*.json",
            "action_proposals_*.json",
            "action_application_*.json",
        ]

        for pattern in patterns:
            files = sorted(glob.glob(str(optimize_dir / pattern)), reverse=True)
            kept_files = files[:keep_last_n] if keep_last_n > 0 else []
            removable_files = files[keep_last_n:] if keep_last_n > 0 else files

            for file_path in kept_files:
                result["files_kept"].append(Path(file_path).name)

            for file_path in removable_files:
                try:
                    Path(file_path).unlink()
                    result["files_removed"].append(Path(file_path).name)
                except OSError as e:
                    result["errors"].append({"file": Path(file_path).name, "error": str(e)})

        if not result["files_removed"] and not result["files_kept"] and not result["errors"]:
            result["no_op"] = True
            result["message"] = "No optimize artifacts found to clean."

        result["success"] = len(result["errors"]) == 0
        return result

    def cleanup_old_artifacts(self, project: str, keep_last_n: int = 3) -> Dict[str, Any]:
        """
        Remove old timestamped artifacts, keeping only the most recent ones.

        Args:
            project: Project name
            keep_last_n: Number of most recent files to keep for each type

        Returns:
            Dictionary with cleanup information
        """
        project_dir = self.get_project_dir(project)
        result = {"project": project, "files_removed": [], "success": False}

        if not project_dir.exists():
            result["error"] = f"Project directory does not exist: {project_dir}"
            return result

        # Patterns for timestamped files
        patterns = [
            "extraction_result_*.json",
            "validator_result_*.json",
            "reward_calculation_*.json",
            "knowledge_graph_*.ttl",
        ]

        for pattern in patterns:
            files = sorted(glob.glob(str(project_dir / pattern)), reverse=True)
            # Remove all but the most recent N files
            for old_file in files[keep_last_n:]:
                Path(old_file).unlink()
                result["files_removed"].append(Path(old_file).name)

        result["success"] = True
        return result

    def ensure_project_exists(self, project: str) -> Path:
        """
        Ensure project directory exists.

        Args:
            project: Project name

        Returns:
            Path to project directory
        """
        project_dir = self.get_project_dir(project)
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir
