"""Reward Calculator - Calculates quality scores for extracted classes and properties."""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import glob

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class RewardCalculator:
    """Calculates quality scores for extracted knowledge based on validation results."""

    def __init__(
        self,
        project: str = "default",
        output_base_dir: str = "/workspace/app/output",
    ):
        """
        Initialize the RewardCalculator.

        Args:
            project: Project name (resolves to /workspace/app/output/{project}/)
            output_base_dir: Base directory for project outputs
        """
        self.project = project
        self.output_base_dir = output_base_dir

        # Resolve validator result file path from project name
        self.validator_result_file = self._resolve_validator_result_path(project, output_base_dir)

        # Load validator result data
        self.validator_data = self._load_json(self.validator_result_file)

        # Load metamodel to get class/property counts for reward weighting
        metamodel_path = str(Path(__file__).parent.parent.parent / "metamodel" / "metamodel.json")
        metamodel = self._load_json(metamodel_path)
        self.metamodel_class_count = len(metamodel.get("classes", []))
        self.metamodel_property_count = len(metamodel.get("properties", []))

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_validator_result_path(self, project: str, output_base_dir: str) -> str:
        """
        Resolve validator_result file path from project name.

        Args:
            project: Project name
            output_base_dir: Base directory for project outputs

        Returns:
            Path to the most recent validator_result_*.json file

        Raises:
            FileNotFoundError: If required files don't exist
        """
        project_dir = Path(output_base_dir) / project

        # Check if project directory exists
        if not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory not found: {project_dir}\n"
                f"Available projects: {', '.join([d.name for d in Path(output_base_dir).iterdir() if d.is_dir()])}"
            )

        # Find most recent validator_result file
        validator_result_pattern = str(project_dir / "validator_result_*.json")
        validator_result_files = sorted(glob.glob(validator_result_pattern), reverse=True)

        if not validator_result_files:
            raise FileNotFoundError(
                f"No validator_result_*.json files found in project directory: {project_dir}\n"
                f"Pattern searched: {validator_result_pattern}\n"
                f"Please run KnowledgeValidator first to generate validation results."
            )

        # Use the most recent validator result
        validator_result_file = validator_result_files[0]

        print(f"Resolved files for project '{project}':")
        print(f"  - Validator result: {validator_result_file}")

        return str(validator_result_file)

    def _calculate_class_quality_score(self, validation: Dict[str, Any]) -> float:
        """
        Calculate quality score for a single class.

        Formula:
        quality_score = 1 + ((necessity - 4) * 2) + (understandability - 4) +
                        (unity - 4) + (boundary_suitability - 4) +
                        (structural_validity - 4) + (text_grounding - 4)

        Args:
            validation: Class validation dictionary with dimension scores

        Returns:
            Quality score as a float
        """
        necessity = validation.get("necessity", {}).get("score", 4)
        understandability = validation.get("understandability", {}).get("score", 4)
        unity = validation.get("unity", {}).get("score", 4)
        boundary_suitability = validation.get("boundary_suitability", {}).get("score", 4)
        structural_validity = validation.get("structural_validity", {}).get("score", 4)
        text_grounding = validation.get("text_grounding", {}).get("score", 4)

        quality_score = (
            1 +
            ((necessity - 4) * 2) +
            (understandability - 4) +
            (unity - 4) +
            (boundary_suitability - 4) +
            (structural_validity - 4) +
            (text_grounding - 4)
        )

        return float(quality_score)

    def _calculate_property_quality_score(self, validation: Dict[str, Any]) -> float:
        """
        Calculate quality score for a single property.

        Formula:
        quality_score = 1 + ((relation_applicability - 4) * 2) + (relation_unity - 4) +
                        (strength_validity - 4) + (contextual_grounding - 4)

        Args:
            validation: Property validation dictionary with dimension scores

        Returns:
            Quality score as a float
        """
        relation_applicability = validation.get("relation_applicability", {}).get("score", 4)
        relation_unity = validation.get("relation_unity", {}).get("score", 4)
        strength_validity = validation.get("strength_validity", {}).get("score", 4)
        contextual_grounding = validation.get("contextual_grounding", {}).get("score", 4)

        quality_score = (
            1 +
            ((relation_applicability - 4) * 2) +
            (relation_unity - 4) +
            (strength_validity - 4) +
            (contextual_grounding - 4)
        )

        return float(quality_score)

    def calculate_all(self) -> Dict[str, Any]:
        """
        Calculate quality scores for all classes and properties.

        Returns:
            Complete report dictionary with class_scores, property_scores, and summary
        """
        # Extract validations from validator result
        class_validations = self.validator_data.get('class_validations', [])
        property_validations = self.validator_data.get('property_validations', [])

        # Calculate class scores
        class_scores = []
        total_class_quality_score = 0.0

        for validation in class_validations:
            quality_score = self._calculate_class_quality_score(validation)
            total_class_quality_score += quality_score

            class_score = {
                "class_id": validation.get("class_id", ""),
                "class_label": validation.get("class_label", ""),
                "dimension_scores": {
                    "necessity": validation.get("necessity", {}).get("score", 4),
                    "understandability": validation.get("understandability", {}).get("score", 4),
                    "unity": validation.get("unity", {}).get("score", 4),
                    "boundary_suitability": validation.get("boundary_suitability", {}).get("score", 4),
                    "structural_validity": validation.get("structural_validity", {}).get("score", 4),
                    "text_grounding": validation.get("text_grounding", {}).get("score", 4)
                },
                "quality_score": quality_score
            }
            class_scores.append(class_score)

        # Calculate property scores
        property_scores = []
        total_property_quality_score = 0.0

        for validation in property_validations:
            quality_score = self._calculate_property_quality_score(validation)
            total_property_quality_score += quality_score

            property_score = {
                "property_id": validation.get("property_id", ""),
                "src_label": validation.get("src_label", ""),
                "property_iri": validation.get("property_iri", ""),
                "dest_label": validation.get("dest_label", ""),
                "dimension_scores": {
                    "relation_applicability": validation.get("relation_applicability", {}).get("score", 4),
                    "relation_unity": validation.get("relation_unity", {}).get("score", 4),
                    "strength_validity": validation.get("strength_validity", {}).get("score", 4),
                    "contextual_grounding": validation.get("contextual_grounding", {}).get("score", 4)
                },
                "quality_score": quality_score
            }
            property_scores.append(property_score)

        # Calculate summary statistics
        total_classes = len(class_scores)
        total_properties = len(property_scores)
        property_weight = self.metamodel_class_count / (2 * self.metamodel_property_count) if self.metamodel_property_count > 0 else 1.0
        combined_total_quality_score = total_class_quality_score + (total_property_quality_score * property_weight)

        average_class_score = total_class_quality_score / total_classes if total_classes > 0 else 0.0
        average_property_score = total_property_quality_score / total_properties if total_properties > 0 else 0.0
        overall_average_score = combined_total_quality_score / (total_classes + total_properties) if (total_classes + total_properties) > 0 else 0.0

        summary = {
            "total_classes": total_classes,
            "total_properties": total_properties,
            "total_class_quality_score": round(total_class_quality_score, 2),
            "total_property_quality_score": round(total_property_quality_score, 2),
            "property_weight": round(property_weight, 4),
            "combined_total_quality_score": round(combined_total_quality_score, 2),
            "average_class_score": round(average_class_score, 2),
            "average_property_score": round(average_property_score, 2),
            "overall_average_score": round(overall_average_score, 2)
        }

        # Build report
        report = {
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
            "class_scores": class_scores,
            "property_scores": property_scores,
            "summary": summary
        }

        return report

    def save_report(self, report: Dict[str, Any]) -> str:
        """
        Save the reward calculation report to a JSON file.

        Args:
            report: Report dictionary to save

        Returns:
            Path to the saved file
        """
        # Generate output filename with timestamp
        timestamp = report.get("timestamp", datetime.now().strftime("%Y%m%d%H%M%S"))
        output_filename = f"reward_calculation_{timestamp}.json"

        # Determine output path
        project_dir = Path(self.output_base_dir) / self.project
        output_path = project_dir / output_filename

        # Ensure project directory exists
        project_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nReward calculation report saved to: {output_path}")
        return str(output_path)

    def print_summary(self, report: Dict[str, Any]) -> None:
        """
        Print a human-readable summary of the reward calculation.

        Args:
            report: Report dictionary
        """
        summary = report.get("summary", {})

        print("\n" + "=" * 80)
        print("REWARD CALCULATION SUMMARY")
        print("=" * 80)
        print(f"Project: {self.project}")
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print("-" * 80)
        print(f"Total Classes: {summary.get('total_classes', 0)}")
        print(f"Total Properties: {summary.get('total_properties', 0)}")
        print("-" * 80)
        print(f"Total Class Quality Score: {summary.get('total_class_quality_score', 0)}")
        print(f"Total Property Quality Score: {summary.get('total_property_quality_score', 0)}")
        print(f"Combined Total Quality Score: {summary.get('combined_total_quality_score', 0)}")
        print("-" * 80)
        print(f"Average Class Score: {summary.get('average_class_score', 0)}")
        print(f"Average Property Score: {summary.get('average_property_score', 0)}")
        print(f"Overall Average Score: {summary.get('overall_average_score', 0)}")
        print("=" * 80)


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Calculate quality scores for extracted knowledge based on validation results."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="default",
        help="Project name (default: 'default')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/app/output",
        help="Base output directory (default: '/workspace/app/output')"
    )

    args = parser.parse_args()

    try:
        # Initialize calculator
        print(f"Initializing RewardCalculator for project: {args.project}")
        calculator = RewardCalculator(
            project=args.project,
            output_base_dir=args.output_dir
        )

        # Calculate scores
        print("\nCalculating quality scores...")
        report = calculator.calculate_all()

        # Print summary
        calculator.print_summary(report)

        # Save report
        output_file = calculator.save_report(report)

        print(f"\nSuccess! Reward calculation completed.")
        print(f"Output file: {output_file}")

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
