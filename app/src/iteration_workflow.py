"""
Iteration Workflow Orchestrator
Coordinates the iterative extraction → evaluation → optimization → apply cycle.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

from src.evaluate_workflow import run_evaluation
from src.extract_workflow import run_knowledge_extraction
from src.optimize.action_applicator import ActionApplicator
from src.optimize.project_state_manager import ProjectStateManager
from src.optimize.quality_tracker import QualityTracker
from src.optimize_workflow import run_optimization_workflow
from src.utils.llm_factory import DEFAULT_LLM_MODEL

load_dotenv()


class IterationWorkflow:
    """Orchestrates iterative extraction-evaluation-optimization cycles."""

    def __init__(
        self,
        batch_files: List[str],
        max_iterations: int = 5,
        quality_threshold: float = 2.0,
        model: str | None = None,
        apply_changes: bool = True,
        output_dir: str = "/workspace/app/output",
    ):
        """
        Initialize the IterationWorkflow.

        Args:
            batch_files: List of batch file paths (project derived from filename)
            max_iterations: Maximum number of iterations
            quality_threshold: Stop when combined score >= threshold
            model: LLM model name
            apply_changes: Whether to apply optimization proposals
            output_dir: Base output directory
        """
        self.batch_files = [os.path.abspath(bf) for bf in batch_files]
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.model = model or DEFAULT_LLM_MODEL
        self.apply_changes = apply_changes
        self.output_dir = output_dir
        self.is_resume_mode = False

        # Derive projects from batch file names
        self.state_manager = ProjectStateManager(output_dir)
        self.projects = [self.state_manager.extract_project_from_batch(bf) for bf in self.batch_files]

        # Create a mapping from project to batch file
        self.project_batch_map = dict(zip(self.projects, self.batch_files, strict=True))

        # Generate run ID
        self.run_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Initialize quality tracker
        self.quality_tracker = QualityTracker(run_id=self.run_id, projects=self.projects, output_dir=output_dir)

        # State file path
        self.iterations_dir = Path(output_dir) / "iterations" / self.run_id
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.iterations_dir / "iteration_state.json"

        # Initialize state
        self.state = self._initialize_state()

    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize or load iteration state."""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "run_id": self.run_id,
            "batch_files": self.batch_files,
            "projects": self.projects,
            "config": {
                "max_iterations": self.max_iterations,
                "quality_threshold": self.quality_threshold,
                "model": self.model,
                "apply_changes": self.apply_changes,
            },
            "current_iteration": 0,
            "status": "initialized",
            "iterations": [],
            "stop_reason": None,
        }

    def _save_state(self) -> None:
        """Save current state to file."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _parse_batch_file(self, batch_file: str) -> List[str]:
        """
        Parse a batch file to get list of PDF paths.

        Args:
            batch_file: Path to batch file

        Returns:
            List of absolute paths to PDF files
        """
        batch_dir = Path(batch_file).parent
        pdf_paths = []

        with open(batch_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Resolve path relative to batch file directory
                if os.path.isabs(line):
                    pdf_path = line
                else:
                    pdf_path = str(batch_dir / line)

                if os.path.exists(pdf_path):
                    pdf_paths.append(pdf_path)
                else:
                    print(f"  Warning: PDF file not found: {pdf_path}")

        return pdf_paths

    def _run_extraction_phase(self, iteration: int) -> Dict[str, Any]:
        """
        Run extraction phase for all projects.

        Args:
            iteration: Current iteration number

        Returns:
            Phase results
        """
        print("\n--- Phase 2: Batch Extraction ---")
        results = {"projects": {}, "success": True}

        for project in self.projects:
            batch_file = self.project_batch_map[project]
            pdf_paths = self._parse_batch_file(batch_file)

            print(f"\n  Project: {project} ({len(pdf_paths)} files)")
            project_results = {"files": [], "success_count": 0, "fail_count": 0}

            for idx, pdf_path in enumerate(pdf_paths, 1):
                print(f"    [{idx}/{len(pdf_paths)}] Processing: {Path(pdf_path).name}")
                try:
                    run_knowledge_extraction(
                        pdf_path=pdf_path, output_dir=self.output_dir, model=self.model, project=project
                    )
                    project_results["files"].append({"file": pdf_path, "success": True})
                    project_results["success_count"] += 1
                    print("      Extraction complete")
                except Exception as e:
                    project_results["files"].append({"file": pdf_path, "success": False, "error": str(e)})
                    project_results["fail_count"] += 1
                    print(f"      Error: {e}")

            results["projects"][project] = project_results
            print(f"  Project {project} complete: {project_results['success_count']}/{len(pdf_paths)} successful")

            if project_results["success_count"] == 0:
                results["success"] = False

        return results

    def _run_evaluation_phase(self, iteration: int) -> Dict[str, Dict[str, Any]]:
        """
        Run evaluation phase for all projects.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary mapping project to evaluation results
        """
        print("\n--- Phase 3: Evaluation ---")
        results = {}

        for project in self.projects:
            print(f"\n  Evaluating project: {project}")
            try:
                eval_results = run_evaluation(project=project, model=self.model)
                results[project] = eval_results
                print(f"  Project {project} evaluation complete")
            except Exception as e:
                print(f"  Error evaluating project {project}: {e}")
                results[project] = {"error": str(e)}

        return results

    def _run_optimization_phase(self, iteration: int) -> Dict[str, Any]:
        """
        Run optimization phase across all projects.

        Args:
            iteration: Current iteration number

        Returns:
            Optimization results
        """
        print("\n--- Phase 4: Optimization ---")
        try:
            # Gather reward history for strategy planner
            recent_rewards = self.quality_tracker.get_recent_rewards(last_n=5)

            results = run_optimization_workflow(
                projects=self.projects,
                model=self.model,
                output_dir=self.output_dir,
                previous_rewards=recent_rewards,
                run_id=self.run_id,
            )
            print(f"  Optimization complete: {len(results.get('stages_completed', []))} stages")
            return results
        except Exception as e:
            print(f"  Error during optimization: {e}")
            return {"error": str(e)}

    def _run_apply_phase(self, iteration: int) -> Dict[str, Any]:
        """
        Run application phase to apply optimization proposals.

        Args:
            iteration: Current iteration number

        Returns:
            Application results
        """
        print("\n--- Phase 5: Apply Proposals ---")

        if not self.apply_changes:
            print("  Skipping (--no-apply flag set)")
            return {"skipped": True}

        try:
            applicator = ActionApplicator(progress=True, backup=True)
            results = applicator.apply_updates()
            applicator.save_result(results)

            applied_count = len(results.get("updates_applied", []))
            print(f"  Applied {applied_count} proposals")
            return results
        except FileNotFoundError as e:
            print(f"  No proposals to apply: {e}")
            return {"no_proposals": True, "error": str(e)}
        except Exception as e:
            print(f"  Error applying proposals: {e}")
            return {"error": str(e)}

    def _load_reward_reports(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the most recent reward calculation reports for all projects.

        Returns:
            Dictionary mapping project to reward report
        """
        import glob

        reports = {}

        for project in self.projects:
            project_dir = Path(self.output_dir) / project
            pattern = str(project_dir / "reward_calculation_*.json")
            files = sorted(glob.glob(pattern), reverse=True)

            if files:
                with open(files[0], "r", encoding="utf-8") as f:
                    reports[project] = json.load(f)
            else:
                reports[project] = {"summary": {}}

        return reports

    def _run_single_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Execute a single iteration cycle.

        Args:
            iteration: Iteration number (1-based)

        Returns:
            Iteration results
        """
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {iteration}/{self.max_iterations} ===")
        print(f"{'=' * 60}")

        iteration_result = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
            "phases_completed": [],
            "combined_quality_score": None,
            "proposals_applied": 0,
        }

        # Phase 1: Reset for extraction
        print("\n--- Phase 1: Reset for Extraction ---")
        for project in self.projects:
            # Archive previous artifacts (if not first iteration)
            if iteration > 1:
                archive_result = self.state_manager.archive_iteration_artifacts(
                    run_id=self.run_id, iteration=iteration - 1, project=project
                )
                print(f"  Archived {len(archive_result.get('files_archived', []))} files for {project}")

            # Reset knowledge.json
            reset_result = self.state_manager.reset_for_extraction(project)
            if reset_result.get("knowledge_cleared"):
                print(f"  Cleared knowledge.json for {project}")
            else:
                print(f"  {reset_result.get('message', 'Reset complete')} for {project}")

        iteration_result["phases_completed"].append("reset")
        self._save_state()

        # Phase 2: Extraction
        extraction_results = self._run_extraction_phase(iteration)
        iteration_result["phases_completed"].append("extract")
        self._save_state()

        # Check if extraction succeeded
        if not extraction_results.get("success", False):
            print("  Warning: Some extractions failed, continuing with available data")

        # Phase 3: Evaluation
        self._run_evaluation_phase(iteration)
        iteration_result["phases_completed"].append("evaluate")
        self._save_state()

        # Phase 4: Optimization
        self._run_optimization_phase(iteration)
        iteration_result["phases_completed"].append("optimize")

        # Archive optimization artifacts
        optimize_archive = self.state_manager.archive_optimize_artifacts(run_id=self.run_id, iteration=iteration)
        print(f"  Archived {len(optimize_archive.get('files_archived', []))} optimization files")
        self._save_state()

        # Phase 5: Apply proposals
        apply_results = self._run_apply_phase(iteration)
        iteration_result["phases_completed"].append("apply")
        iteration_result["proposals_applied"] = len(apply_results.get("updates_applied", []))
        self._save_state()

        # Phase 6: Record quality metrics
        print("\n--- Phase 6: Quality Recording ---")
        reward_reports = self._load_reward_reports()
        quality_entry = self.quality_tracker.record_quality(iteration, reward_reports)
        iteration_result["combined_quality_score"] = quality_entry.get("combined_average_score")

        print(f"  Combined average score: {iteration_result['combined_quality_score']:.2f}")

        # Show improvement from previous iteration
        if iteration > 1:
            improvement = self.quality_tracker.get_improvement(iteration - 1, iteration)
            if improvement:
                delta = improvement["delta"]
                print(f"  Change from iteration {iteration - 1}: {delta:+.2f}")

        return iteration_result

    def _check_stopping_criteria(self, iteration: int, iteration_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if the iteration loop should stop.

        Args:
            iteration: Current iteration number
            iteration_result: Results from current iteration

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check max iterations
        if iteration >= self.max_iterations:
            return (True, "max_iterations")

        # Check quality threshold
        should_stop, reason = self.quality_tracker.should_stop(
            quality_threshold=self.quality_threshold, degradation_tolerance=2
        )
        if should_stop:
            return (True, reason)

        # Check if no proposals were generated
        if iteration_result.get("proposals_applied", 0) == 0:
            # Only stop if apply was not skipped and no error
            apply_results = iteration_result.get("apply_results", {})
            if not apply_results.get("skipped") and apply_results.get("no_proposals"):
                return (True, "no_proposals")

        return (False, None)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete iteration workflow.

        Returns:
            Final results summary
        """
        print("\n" + "=" * 60)
        print("=== Iterative Knowledge Extraction Optimization ===")
        print("=" * 60)
        print(f"Run ID: {self.run_id}")
        print(f"Projects: {', '.join(self.projects)}")
        print(f"Batch files: {len(self.batch_files)}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Model: {self.model}")
        print(f"Apply changes: {self.apply_changes}")

        # Determine start iteration (skip already-completed iterations on resume)
        completed_iterations = len(self.state.get("iterations", []))
        start_iteration = completed_iterations + 1

        if start_iteration > self.max_iterations:
            print(f"\nAll {self.max_iterations} iterations already completed. Nothing to resume.")
            self._display_final_summary()
            return self.state

        if start_iteration > 1:
            print(f"\nResuming from iteration {start_iteration} ({completed_iterations} already completed)")

        self.state["status"] = "in_progress"
        self._save_state()

        try:
            # Fresh run only: clear stale optimize artifacts to avoid cross-run contamination.
            if start_iteration == 1 and not self.is_resume_mode:
                print("\n--- Pre-run Cleanup: Optimize Artifacts ---")
                cleanup_result = self.state_manager.cleanup_optimize_artifacts(keep_last_n=0)
                removed_count = len(cleanup_result.get("files_removed", []))
                kept_count = len(cleanup_result.get("files_kept", []))
                errors = cleanup_result.get("errors", [])

                print(f"  Removed: {removed_count}")
                print(f"  Kept: {kept_count}")
                if cleanup_result.get("no_op"):
                    print("  No-op: no existing optimize artifacts to clean")

                if errors:
                    print(f"  Cleanup errors: {len(errors)}")

                if not cleanup_result.get("success", False):
                    error_details = "; ".join(
                        f"{err.get('file', 'unknown')}: {err.get('error', 'unknown')}" for err in errors
                    )
                    raise RuntimeError(
                        "Failed to clean optimize artifacts before fresh iteration run: "
                        f"{error_details or 'unknown error'}"
                    )

            for iteration in range(start_iteration, self.max_iterations + 1):
                self.state["current_iteration"] = iteration

                # Run single iteration
                iteration_result = self._run_single_iteration(iteration)
                self.state["iterations"].append(iteration_result)
                self._save_state()

                # Check stopping criteria
                should_stop, reason = self._check_stopping_criteria(iteration, iteration_result)
                if should_stop:
                    self.state["stop_reason"] = reason
                    self.state["status"] = "stopped_early" if iteration < self.max_iterations else "completed"
                    break
            else:
                # Loop completed without stopping early
                self.state["stop_reason"] = "max_iterations"
                self.state["status"] = "completed"

        except Exception as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            print(f"\nError during iteration: {e}")
            import traceback

            traceback.print_exc()

        self._save_state()
        self._display_final_summary()

        return self.state

    def _display_final_summary(self) -> None:
        """Display final summary of the iteration workflow."""
        print("\n" + "=" * 60)
        print("=== Iteration Complete ===")
        print("=" * 60)

        print(f"Run ID: {self.run_id}")
        print(f"Status: {self.state.get('status', 'unknown')}")
        print(f"Stop reason: {self.state.get('stop_reason', 'N/A')}")
        print(f"Iterations completed: {len(self.state.get('iterations', []))}")

        # Show quality history
        self.quality_tracker.print_summary()

        # Show generated files
        print("\nGenerated files:")
        print(f"  - {self.state_file}")
        print(f"  - {self.quality_tracker.history_file}")

    @classmethod
    def resume(cls, run_id: str, output_dir: str = "/workspace/app/output") -> "IterationWorkflow":
        """
        Resume a previous iteration run.

        Args:
            run_id: Run ID to resume
            output_dir: Base output directory

        Returns:
            IterationWorkflow instance with restored state
        """
        state_file = Path(output_dir) / "iterations" / run_id / "iteration_state.json"

        if not state_file.exists():
            raise FileNotFoundError(f"No iteration state found for run ID: {run_id}")

        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Resume must restore the exact configuration of the interrupted run;
        # substituting defaults would silently change how the run behaves.
        config = state.get("config", {})
        missing = [k for k in ("max_iterations", "quality_threshold", "model", "apply_changes") if k not in config]
        if missing:
            raise ValueError(f"Cannot resume run {run_id}: state file is missing config keys: {', '.join(missing)}")

        instance = cls(
            batch_files=state.get("batch_files", []),
            max_iterations=config["max_iterations"],
            quality_threshold=config["quality_threshold"],
            model=config["model"],
            apply_changes=config["apply_changes"],
            output_dir=output_dir,
        )

        # Override run_id and state
        instance.run_id = run_id
        instance.state = state
        instance.iterations_dir = Path(output_dir) / "iterations" / run_id
        instance.state_file = instance.iterations_dir / "iteration_state.json"
        instance.quality_tracker = QualityTracker(run_id=run_id, projects=instance.projects, output_dir=output_dir)
        instance.is_resume_mode = True

        return instance


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Run iterative extraction-evaluation-optimization workflow")
    parser.add_argument(
        "--batch", nargs="+", help="One or more batch files (project derived from filename: batch-{project}.txt)"
    )
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations (default: 5)")
    parser.add_argument(
        "--quality-threshold", type=float, default=2.0, help="Stop when combined score >= threshold (default: 2.0)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model name (default: %(default)s)")
    parser.add_argument("--no-apply", action="store_true", help="Dry-run optimization without applying changes")
    parser.add_argument("--resume", type=str, help="Resume from a previous run ID")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/app/output",
        help="Base output directory (default: /workspace/app/output)",
    )

    args = parser.parse_args()

    try:
        if args.resume:
            # Resume mode
            print(f"Resuming run: {args.resume}")
            workflow = IterationWorkflow.resume(run_id=args.resume, output_dir=args.output_dir)
        else:
            # New run mode
            if not args.batch:
                parser.error("--batch is required when not using --resume")

            workflow = IterationWorkflow(
                batch_files=args.batch,
                max_iterations=args.max_iterations,
                quality_threshold=args.quality_threshold,
                model=args.model,
                apply_changes=not args.no_apply,
                output_dir=args.output_dir,
            )

        results = workflow.run()

        # Exit with appropriate code
        if results.get("status") == "failed":
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
