"""
Quality Tracker - Tracks quality scores across iterations to measure improvement.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class QualityTracker:
    """Tracks and analyzes quality scores across iterations."""

    def __init__(self, run_id: str, projects: List[str], output_dir: str = "/workspace/app/output"):
        """
        Initialize the QualityTracker.

        Args:
            run_id: Unique identifier for this iteration run (timestamp format: YYYYMMDDhhmmss)
            projects: List of project names being tracked
            output_dir: Base output directory
        """
        self.run_id = run_id
        self.projects = projects
        self.output_dir = output_dir
        self.iterations_dir = Path(output_dir) / "iterations" / run_id

        # Ensure iterations directory exists
        self.iterations_dir.mkdir(parents=True, exist_ok=True)

        # Quality history file path
        self.history_file = self.iterations_dir / "quality_history.json"

        # Load existing history or initialize new one
        self.history_data = self._load_or_initialize_history()

    def _load_or_initialize_history(self) -> Dict[str, Any]:
        """Load existing history or initialize a new one."""
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {"run_id": self.run_id, "projects": self.projects, "history": []}

    def _save_history(self) -> None:
        """Save the history data to file."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history_data, f, indent=2, ensure_ascii=False)

    def record_quality(self, iteration: int, project_reward_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Record quality metrics for an iteration.

        Args:
            iteration: Iteration number (1-based)
            project_reward_reports: Dictionary mapping project names to their reward reports
                                   Each report should have a 'summary' key with metrics

        Returns:
            The recorded entry
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Extract metrics for each project
        project_metrics = {}
        total_score = 0.0
        total_items = 0

        for project, report in project_reward_reports.items():
            summary = report.get("summary", {})
            metrics = {
                "total_classes": summary.get("total_classes", 0),
                "total_properties": summary.get("total_properties", 0),
                "total_class_quality_score": summary.get("total_class_quality_score", 0),
                "total_property_quality_score": summary.get("total_property_quality_score", 0),
                "combined_total_quality_score": summary.get("combined_total_quality_score", 0),
                "average_class_score": summary.get("average_class_score", 0),
                "average_property_score": summary.get("average_property_score", 0),
                "overall_average_score": summary.get("overall_average_score", 0),
            }
            project_metrics[project] = metrics

            # Accumulate for combined average
            total_score += summary.get("combined_total_quality_score", 0)
            total_items += summary.get("total_classes", 0) + summary.get("total_properties", 0)

        # Calculate combined average across all projects
        combined_average_score = round(total_score / total_items, 2) if total_items > 0 else 0.0

        entry = {
            "iteration": iteration,
            "timestamp": timestamp,
            "project_metrics": project_metrics,
            "combined_average_score": combined_average_score,
        }

        # Check if this iteration already exists (update) or is new (append)
        existing_idx = None
        for idx, hist_entry in enumerate(self.history_data["history"]):
            if hist_entry["iteration"] == iteration:
                existing_idx = idx
                break

        if existing_idx is not None:
            self.history_data["history"][existing_idx] = entry
        else:
            self.history_data["history"].append(entry)

        # Save to file
        self._save_history()

        return entry

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get all recorded quality metrics.

        Returns:
            List of quality history entries
        """
        return self.history_data.get("history", [])

    def get_latest_score(self) -> Optional[float]:
        """
        Get the most recent combined average score.

        Returns:
            The latest combined average score, or None if no history
        """
        history = self.get_history()
        if not history:
            return None
        return history[-1].get("combined_average_score")

    def get_trend(self, last_n: int = 3) -> str:
        """
        Analyze the recent quality trend.

        Args:
            last_n: Number of recent iterations to analyze

        Returns:
            One of: "improving", "degrading", "stable", "insufficient_data"
        """
        history = self.get_history()

        if len(history) < 2:
            return "insufficient_data"

        # Get last N entries
        recent = history[-last_n:] if len(history) >= last_n else history

        if len(recent) < 2:
            return "insufficient_data"

        # Calculate trend based on score changes
        scores = [entry.get("combined_average_score", 0) for entry in recent]
        changes = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]

        # Determine trend
        positive_changes = sum(1 for c in changes if c > 0.01)  # Small threshold to avoid noise
        negative_changes = sum(1 for c in changes if c < -0.01)

        if positive_changes > negative_changes:
            return "improving"
        elif negative_changes > positive_changes:
            return "degrading"
        else:
            return "stable"

    def get_improvement(self, from_iteration: int, to_iteration: int) -> Optional[Dict[str, Any]]:
        """
        Compare quality between two iterations.

        Args:
            from_iteration: Starting iteration number
            to_iteration: Ending iteration number

        Returns:
            Dictionary with delta values, or None if iterations not found
        """
        history = self.get_history()

        from_entry = None
        to_entry = None

        for entry in history:
            if entry["iteration"] == from_iteration:
                from_entry = entry
            if entry["iteration"] == to_iteration:
                to_entry = entry

        if from_entry is None or to_entry is None:
            return None

        from_score = from_entry.get("combined_average_score", 0)
        to_score = to_entry.get("combined_average_score", 0)

        return {
            "from_iteration": from_iteration,
            "to_iteration": to_iteration,
            "from_score": from_score,
            "to_score": to_score,
            "delta": round(to_score - from_score, 2),
            "percent_change": round(((to_score - from_score) / abs(from_score)) * 100, 2) if from_score != 0 else 0,
        }

    def should_stop(self, quality_threshold: float, degradation_tolerance: int = 2) -> Tuple[bool, Optional[str]]:
        """
        Check if the iteration loop should stop.

        Args:
            quality_threshold: Stop when combined average score >= threshold
            degradation_tolerance: Stop if quality degrades for N consecutive iterations

        Returns:
            Tuple of (should_stop, reason)
            reason is one of: "quality_achieved", "quality_degrading", None (if should not stop)
        """
        history = self.get_history()

        if not history:
            return (False, None)

        # Check if quality threshold achieved
        latest_score = history[-1].get("combined_average_score", 0)
        if latest_score >= quality_threshold:
            return (True, "quality_achieved")

        # Check for degradation
        if len(history) >= degradation_tolerance + 1:
            recent_scores = [
                entry.get("combined_average_score", 0) for entry in history[-(degradation_tolerance + 1) :]
            ]
            # Check if all recent changes are negative
            consecutive_degradations = 0
            for i in range(len(recent_scores) - 1):
                if recent_scores[i + 1] < recent_scores[i]:
                    consecutive_degradations += 1
                else:
                    consecutive_degradations = 0  # Reset on non-degradation

            if consecutive_degradations >= degradation_tolerance:
                return (True, "quality_degrading")

        return (False, None)

    def get_recent_rewards(self, last_n: int = 5) -> List[float]:
        """
        Get the most recent combined_total_quality_score values.

        Sums combined_total_quality_score across all projects for each iteration.

        Args:
            last_n: Number of recent iterations to return

        Returns:
            List of combined total quality scores (most recent last)
        """
        history = self.get_history()
        recent = history[-last_n:] if history else []

        rewards = []
        for entry in recent:
            project_metrics = entry.get("project_metrics", {})
            total = sum(pm.get("combined_total_quality_score", 0) for pm in project_metrics.values())
            rewards.append(total)

        return rewards

    def print_summary(self) -> None:
        """Print a human-readable summary of quality history."""
        history = self.get_history()

        print("\n" + "=" * 80)
        print("QUALITY HISTORY SUMMARY")
        print("=" * 80)
        print(f"Run ID: {self.run_id}")
        print(f"Projects: {', '.join(self.projects)}")
        print(f"Total Iterations: {len(history)}")
        print("-" * 80)

        if not history:
            print("No quality data recorded yet.")
            return

        print("\nIteration History:")
        for entry in history:
            iteration = entry.get("iteration", "?")
            score = entry.get("combined_average_score", 0)
            timestamp = entry.get("timestamp", "N/A")
            print(f"  Iteration {iteration}: {score:+.2f} (recorded: {timestamp})")

        # Show improvement from first to last
        if len(history) >= 2:
            improvement = self.get_improvement(history[0]["iteration"], history[-1]["iteration"])
            if improvement:
                print("-" * 80)
                print(f"Overall improvement: {improvement['delta']:+.2f}")
                print(f"  From: {improvement['from_score']:.2f} (iteration {improvement['from_iteration']})")
                print(f"  To:   {improvement['to_score']:.2f} (iteration {improvement['to_iteration']})")

        # Show current trend
        trend = self.get_trend()
        print(f"\nCurrent trend: {trend}")
        print("=" * 80)
