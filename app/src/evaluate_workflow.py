"""
Knowledge Extraction Evaluation
Evaluates extracted knowledge quality by running validation and reward calculation
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Add the app directory to sys.path to enable src imports
app_dir = os.path.dirname(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Load environment variables from .env file
load_dotenv()

from src.evaluate.knowledge_validator import KnowledgeValidator
from src.evaluate.reward_calculator import RewardCalculator


def run_evaluation(project: str = "default", model: str = "gemini-3.1-flash-lite-preview", verbose: bool = False) -> Dict[str, Any]:
    """
    Execute knowledge quality evaluation

    Args:
        project: Project name to evaluate
        model: LLM model for validation

    Returns:
        Combined evaluation results
    """
    output_dir = "/workspace/app/output"

    results = {
        "project": project,
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "stages_completed": [],
        "errors": []
    }

    # Stage 1: Knowledge Validation
    print("\n=== Stage 1: Knowledge Validation ===")
    try:
        validator = KnowledgeValidator(
            project=project,
            model=model,
            output_base_dir=output_dir,
            verbose=verbose
        )
        validation_report = validator.validate()
        validation_file = validator.save_report(validation_report)

        results["validation_file"] = validation_file
        results["validation_summary"] = {
            "classes_validated": len(validation_report["class_validations"]),
            "properties_validated": len(validation_report["property_validations"])
        }
        results["stages_completed"].append("validation")

        print(f"\nValidation completed!")
        print(f"  - Classes validated: {results['validation_summary']['classes_validated']}")
        print(f"  - Properties validated: {results['validation_summary']['properties_validated']}")
        print(f"  - Report saved to: {validation_file}")

    except Exception as e:
        error_msg = f"Validation failed: {e}"
        results["errors"].append(error_msg)
        print(f"\nError: {error_msg}")
        import traceback
        traceback.print_exc()
        return results

    # Stage 2: Reward Calculation
    print("\n=== Stage 2: Reward Calculation ===")
    try:
        calculator = RewardCalculator(
            project=project,
            output_base_dir=output_dir
        )
        reward_report = calculator.calculate_all()
        reward_file = calculator.save_report(reward_report)

        results["reward_file"] = reward_file
        results["reward_summary"] = reward_report["summary"]
        results["stages_completed"].append("reward_calculation")

        # Print summary
        calculator.print_summary(reward_report)

    except Exception as e:
        error_msg = f"Reward calculation failed: {e}"
        results["errors"].append(error_msg)
        print(f"\nError: {error_msg}")
        import traceback
        traceback.print_exc()
        return results

    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate knowledge extraction quality")
    parser.add_argument(
        "--project",
        type=str,
        default="default",
        help="Project name to evaluate (default: default)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="LLM model for validation (default: gemini-3.1-flash-lite-preview)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Output detailed justifications for each validation score (default: scores only)"
    )

    args = parser.parse_args()

    print("\n=== Knowledge Extraction Evaluation ===")
    print(f"Project: {args.project}")
    print(f"Model: {args.model}")

    try:
        results = run_evaluation(
            project=args.project,
            model=args.model,
            verbose=args.verbose
        )

        # Final summary
        print("\n=== Evaluation Complete ===")
        print(f"Stages completed: {', '.join(results['stages_completed'])}")

        if results.get("errors"):
            print("\n=== Errors ===")
            for error in results["errors"]:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("\nAll evaluation stages completed successfully!")
            if "validation_file" in results:
                print(f"  - Validation report: {results['validation_file']}")
            if "reward_file" in results:
                print(f"  - Reward report: {results['reward_file']}")

            if "reward_summary" in results:
                summary = results["reward_summary"]
                print(f"\nQuality Metrics:")
                print(f"  - Overall average score: {summary.get('overall_average_score', 'N/A')}")
                print(f"  - Total classes: {summary.get('total_classes', 0)}")
                print(f"  - Total properties: {summary.get('total_properties', 0)}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
