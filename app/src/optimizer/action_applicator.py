"""Action Applicator - Applies action proposals to target files."""

import json
import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import Logger


class ActionApplicator:
    """Applies action proposals to target files (custom prompts and metamodel)."""

    def __init__(
        self,
        project: str = "default",
        progress: bool = True,
        backup: bool = True,
    ):
        """
        Initialize the ActionApplicator.

        Args:
            project: Project name (resolves to /workspace/app/output/{project}/)
            progress: Whether to show progress messages
            backup: Whether to create backups before modifying files
        """
        self.project = project
        self.progress = progress
        self.backup = backup
        self.output_base_dir = "/workspace/app/output/optimizer"

        # Resolve action proposals file path from project name
        self.action_proposals_file = self._resolve_action_proposals_path(project)

        # Load action proposals data
        self.action_proposals_data = self._load_json(self.action_proposals_file)

        # Setup paths
        self.app_dir = Path(__file__).parent.parent.parent
        self.custom_prompts_dir = self.app_dir / "prompts" / "custom"
        self.metamodel_path = self.app_dir / "metamodel" / "metamodel.json"
        self.backup_dir = self.app_dir / "log" / "optimizer"

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Track backups created
        self.backups_created = []

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_action_proposals_path(self, project: str) -> str:
        """
        Resolve action_proposals file path from project name.

        Args:
            project: Project name

        Returns:
            Path to the most recent action_proposals_*.json file

        Raises:
            FileNotFoundError: If required files don't exist
        """
        project_dir = Path(self.output_base_dir) / project

        # Check if project directory exists
        if not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory not found: {project_dir}\n"
                f"Available projects: {', '.join([d.name for d in Path(self.output_base_dir).iterdir() if d.is_dir()])}"
            )

        # Find most recent action_proposals file
        action_proposals_pattern = str(project_dir / "action_proposals_*.json")
        action_proposals_files = sorted(glob.glob(action_proposals_pattern), reverse=True)

        if not action_proposals_files:
            raise FileNotFoundError(
                f"No action_proposals_*.json files found in project directory: {project_dir}\n"
                f"Pattern searched: {action_proposals_pattern}\n"
                f"Please run ActionPlanner first."
            )

        # Use the most recent action proposals
        action_proposals_file = action_proposals_files[0]

        if self.progress:
            print(f"[ActionApplicator] Resolved action proposals file: {action_proposals_file}")

        return str(action_proposals_file)

    def _create_backup(self, file_path: Path) -> Optional[str]:
        """
        Create timestamped backup of a file in /app/log/optimizer/.

        Args:
            file_path: Path to the file to backup

        Returns:
            Path to the backup file, or None if file doesn't exist
        """
        if not file_path.exists():
            if self.progress:
                print(f"[ActionApplicator] Warning: File not found for backup: {file_path}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_filename

        shutil.copy2(file_path, backup_path)

        if self.progress:
            print(f"[ActionApplicator] Created backup: {backup_path}")

        return str(backup_path)

    def _update_custom_prompt(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a custom prompt file.

        Args:
            proposal: Proposal containing target and updated_content

        Returns:
            Result dictionary with status and backup info
        """
        target = proposal["target"]
        updated_content = proposal["updated_content"]

        custom_path = self.custom_prompts_dir / target

        result = {
            "target_type": "custom_prompt",
            "target": target,
            "status": "success"
        }

        # Backup if enabled
        if self.backup:
            backup_file = self._create_backup(custom_path)
            if backup_file:
                result["backup_file"] = backup_file
                self.backups_created.append(backup_file)

        # Write updated content (overwrite)
        with open(custom_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        if self.progress:
            print(f"[ActionApplicator] ✓ Updated custom prompt: {target}")

        return result

    def _update_metamodel_rules(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update metamodel extract_rules or prohibit_rules.

        Args:
            proposal: Proposal containing target_type, target, and updated_content

        Returns:
            Result dictionary with status and backup info
        """
        target_type = proposal["target_type"]  # "extract_rules" or "prohibit_rules"
        target = proposal["target"]            # Class or property name
        updated_rules = proposal["updated_content"]  # Array of strings

        result = {
            "target_type": target_type,
            "target": target,
            "status": "success"
        }

        # Backup if enabled
        if self.backup:
            backup_file = self._create_backup(self.metamodel_path)
            if backup_file:
                result["backup_file"] = backup_file
                # Only add to backups_created once per metamodel file
                if backup_file not in self.backups_created:
                    self.backups_created.append(backup_file)

        # Load metamodel
        with open(self.metamodel_path, 'r', encoding='utf-8') as f:
            metamodel = json.load(f)

        # Find and update target
        updated = False

        # Search in classes
        for cls in metamodel.get("classes", []):
            if cls.get("name") == target:
                cls[target_type] = updated_rules
                updated = True
                break

        # Search in properties if not found in classes
        if not updated:
            for prop in metamodel.get("properties", []):
                if prop.get("name") == target:
                    prop[target_type] = updated_rules
                    updated = True
                    break

        if not updated:
            raise ValueError(f"Target '{target}' not found in metamodel")

        # Save updated metamodel (overwrite)
        with open(self.metamodel_path, 'w', encoding='utf-8') as f:
            json.dump(metamodel, f, indent=4, ensure_ascii=False)

        if self.progress:
            print(f"[ActionApplicator] ✓ Updated metamodel {target_type}: {target}")

        return result

    def apply_updates(self) -> Dict[str, Any]:
        """
        Apply all proposals from action_proposals file.

        Returns:
            Summary of updates applied
        """
        proposals = self.action_proposals_data.get("proposals", [])

        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "action_proposals_file": self.action_proposals_file,
            "total_proposals": len(proposals),
            "updates_applied": [],
            "errors": [],
            "backups_created": []
        }

        if self.progress:
            print(f"[ActionApplicator] Processing {len(proposals)} proposals...")

        for i, proposal in enumerate(proposals, 1):
            try:
                target_type = proposal["target_type"]
                target = proposal["target"]

                if self.progress:
                    print(f"[ActionApplicator] [{i}/{len(proposals)}] Applying {target_type}:{target}...")

                if target_type == "custom_prompt":
                    update_result = self._update_custom_prompt(proposal)
                elif target_type in ["extract_rules", "prohibit_rules"]:
                    update_result = self._update_metamodel_rules(proposal)
                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

                results["updates_applied"].append(update_result)

            except Exception as e:
                error_info = {
                    "target_type": proposal.get("target_type"),
                    "target": proposal.get("target"),
                    "error": str(e)
                }
                results["errors"].append(error_info)

                if self.progress:
                    print(f"[ActionApplicator] ✗ Error: {error_info}")

        # Add all backups to results
        results["backups_created"] = self.backups_created

        if self.progress:
            print(f"[ActionApplicator] ✓ Applied {len(results['updates_applied'])}/{len(proposals)} updates")
            if results["errors"]:
                print(f"[ActionApplicator] ✗ {len(results['errors'])} errors occurred")

        return results

    def save_result(self, result: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """
        Save application results to file.

        Args:
            result: Results dictionary
            output_dir: Optional output directory (defaults to project directory)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            # Default: save to project directory
            output_dir = Path(self.output_base_dir) / self.project
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result['timestamp']
        output_file = output_dir / f"action_application_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Applicator - Applies action proposals to target files")
    parser.add_argument(
        "--project",
        required=True,
        help="Project name (resolves to /workspace/app/output/{project}/)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation before modifying files"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages"
    )

    args = parser.parse_args()

    try:
        # Create applicator
        print(f"[ActionApplicator] Initializing ActionApplicator for project: {args.project}")
        applicator = ActionApplicator(
            project=args.project,
            progress=not args.no_progress,
            backup=not args.no_backup
        )

        # Apply updates
        print("[ActionApplicator] Starting application of proposals...")
        result = applicator.apply_updates()

        # Save result
        output_file = applicator.save_result(result)
        print(f"\n[ActionApplicator] Application completed!")
        print(f"[ActionApplicator] Result saved to: {output_file}")

        # Print summary
        updates_applied = result.get('updates_applied', [])
        errors = result.get('errors', [])
        backups_created = result.get('backups_created', [])

        print(f"\n[ActionApplicator] Summary:")
        print(f"  Total proposals: {result.get('total_proposals', 0)}")
        print(f"  Successfully applied: {len(updates_applied)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Backups created: {len(backups_created)}")

        if updates_applied:
            print(f"\n[ActionApplicator] Updates Applied:")
            for i, update in enumerate(updates_applied, 1):
                target_type = update.get('target_type', '?')
                target = update.get('target', '?')
                backup_info = f" (backup: {update.get('backup_file', 'none')})" if 'backup_file' in update else ""
                print(f"  {i}. {target_type}:{target}{backup_info}")

        if errors:
            print(f"\n[ActionApplicator] Errors:")
            for i, error in enumerate(errors, 1):
                target_type = error.get('target_type', '?')
                target = error.get('target', '?')
                error_msg = error.get('error', '?')
                print(f"  {i}. {target_type}:{target} - {error_msg}")

        if backups_created:
            print(f"\n[ActionApplicator] Backups saved to: {applicator.backup_dir}")

    except FileNotFoundError as e:
        print(f"\n[ActionApplicator] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ActionApplicator] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
