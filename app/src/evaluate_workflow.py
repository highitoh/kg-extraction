"""
Knowledge Extraction Evaluation
Evaluates extracted knowledge quality by running validation and reward calculation
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Add the app directory to sys.path to enable src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

from src.evaluate.knowledge_validator import KnowledgeValidator
from src.evaluate.reward_calculator import RewardCalculator
from src.utils.llm_factory import DEFAULT_EVAL_LLM_MODEL, DEFAULT_REASONING_EFFORT, SUPPORTED_REASONING_EFFORTS

# Load environment variables from .env file
load_dotenv()


def run_evaluation(
    project: str = "default",
    model: str | None = None,
    max_concurrency: int = 4,
) -> Dict[str, Any]:
    """
    Execute knowledge quality evaluation

    Args:
        project: Project name to evaluate
        model: LLM model for validation
        max_concurrency: Maximum concurrent validation LLM calls

    Returns:
        Combined evaluation results
    """
    output_dir = "/workspace/app/output"

    results = {
        "project": project,
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "stages_completed": [],
        "errors": [],
    }

    # Stage 1: Knowledge Validation
    print("\n=== Stage 1: Knowledge Validation ===")
    try:
        validator = KnowledgeValidator(
            project=project,
            model=model,
            output_base_dir=output_dir,
            max_concurrency=max_concurrency,
        )
        validation_report = validator.validate()
        validation_file = validator.save_report(validation_report)

        results["validation_file"] = validation_file
        results["validation_summary"] = {
            "classes_validated": len(validation_report["class_validations"]),
            "properties_validated": len(validation_report["property_validations"]),
        }
        results["stages_completed"].append("validation")

        print("\nValidation completed!")
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
        calculator = RewardCalculator(project=project, output_base_dir=output_dir)
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
    parser.add_argument("--project", type=str, default="default", help="Project name to evaluate (default: default)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EVAL_LLM_MODEL,
        help="LLM model for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        choices=sorted(SUPPORTED_REASONING_EFFORTS),
        default=None,
        help=f"LLM reasoning effort (default: {DEFAULT_REASONING_EFFORT})",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4, help="Maximum concurrent validation LLM calls (default: 4)"
    )

    args = parser.parse_args()

    # LLMFactory reads the reasoning effort from LLM_REASONING_EFFORT
    if args.reasoning:
        os.environ["LLM_REASONING_EFFORT"] = args.reasoning

    print("\n=== Knowledge Extraction Evaluation ===")
    print(f"Project: {args.project}")
    print(f"Model: {args.model}")
    print(f"Reasoning effort: {os.getenv('LLM_REASONING_EFFORT') or DEFAULT_REASONING_EFFORT}")

    try:
        results = run_evaluation(
            project=args.project,
            model=args.model,
            max_concurrency=args.max_concurrency,
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
                print("\nQuality Metrics:")
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
