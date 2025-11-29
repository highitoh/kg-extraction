"""
Optimization Workflow Orchestrator
Coordinates the optimization pipeline from validator results to action proposals
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Path setup
app_dir = os.path.dirname(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

load_dotenv()

# Component imports
from src.optimize.issue_classifier import IssueClassifier
from src.optimize.issue_categorizer import IssueCategorizer
from src.optimize.strategy_planner import StrategyPlanner
from src.optimize.action_planner import ActionPlanner


def aggregate_validator_results(projects: List[str], output_dir: str) -> str:
    """
    Aggregate validator results from multiple projects into optimize directory.

    For each project:
    1. Find most recent validator_result_*.json in project directory
    2. Load and extract class_validations and property_validations
    3. Merge all validations into single aggregated file
    4. Save to optimize directory with timestamp

    Args:
        projects: List of project names to aggregate
        output_dir: Base output directory

    Returns:
        Path to aggregated validator_result file

    Raises:
        FileNotFoundError: If project directory or validator results don't exist
    """
    optimize_dir = Path(output_dir) / "optimize"
    optimize_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    all_class_validations = []
    all_property_validations = []
    source_files = []

    # Validate all projects first
    for project in projects:
        project_dir = Path(output_dir) / project
        if not project_dir.exists():
            available = [d.name for d in Path(output_dir).iterdir() if d.is_dir() and d.name != 'optimize']
            raise FileNotFoundError(
                f"Project directory not found: {project_dir}\n"
                f"Available projects: {', '.join(available)}"
            )

    # Aggregate validator results from all projects
    for project in projects:
        # Find most recent validator_result for this project
        project_dir = Path(output_dir) / project
        pattern = str(project_dir / "validator_result_*.json")
        files = sorted(glob.glob(pattern), reverse=True)

        if not files:
            raise FileNotFoundError(
                f"No validator results found for project: {project}\n"
                f"Please run: python src/evaluate_workflow.py --project {project}"
            )

        validator_file = files[0]
        print(f"  - Loading {project}: {validator_file}")

        # Load and merge validations
        with open(validator_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_class_validations.extend(data.get('class_validations', []))
        all_property_validations.extend(data.get('property_validations', []))
        source_files.append({
            "project": project,
            "validator_result_file": validator_file,
            "knowledge_file": data.get('knowledge_file', ''),
            "extraction_result_file": data.get('extraction_result_file', '')
        })

    # Use the first project's files as reference for metamodel and knowledge
    # (All projects should share the same metamodel)
    first_project_knowledge_file = source_files[0]["knowledge_file"]
    first_project_extraction_file = source_files[0]["extraction_result_file"]

    # Create aggregated file
    aggregated = {
        "timestamp": timestamp,
        "aggregated_from": source_files,
        "class_validations": all_class_validations,
        "property_validations": all_property_validations,
        "knowledge_file": first_project_knowledge_file,
        "extraction_result_file": first_project_extraction_file
    }

    # Save to optimize directory
    output_file = optimize_dir / f"validator_result_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print(f"  - Aggregated {len(all_class_validations)} class validations")
    print(f"  - Aggregated {len(all_property_validations)} property validations")
    print(f"  - Saved to: {output_file}")

    return str(output_file)


def run_optimization_workflow(
    projects: List[str],
    model: str = "gpt-5-mini",
    output_dir: str = "/workspace/app/output"
) -> Dict[str, Any]:
    """
    Execute the complete optimization workflow.

    Args:
        projects: List of project names to optimize
        model: LLM model name
        output_dir: Base output directory

    Returns:
        Dictionary containing workflow results and file paths

    Raises:
        FileNotFoundError: If required files don't exist
        Exception: If any stage fails
    """
    results = {
        "projects": projects,
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "stages_completed": [],
        "errors": []
    }

    try:
        # Stage 0: Aggregate validator results
        print("\n=== Stage 0: Aggregating Validator Results ===")
        print(f"Processing {len(projects)} projects: {', '.join(projects)}")
        validator_file = aggregate_validator_results(projects, output_dir)
        results["validator_result_file"] = validator_file
        results["stages_completed"].append("aggregate_validator_results")

        # Stage 1: Issue Classification
        print("\n=== Stage 1: Issue Classification ===")
        classifier = IssueClassifier(model=model)
        classification_report = classifier.classify()
        classification_file = classifier.save_report(classification_report)
        results["classification_file"] = classification_file
        results["stages_completed"].append("issue_classification")

        # Stage 2: Issue Categorization
        print("\n=== Stage 2: Issue Categorization ===")
        categorizer = IssueCategorizer(model=model)
        category_report = categorizer.categorize()
        category_file = categorizer.save_report(category_report)
        results["category_file"] = category_file
        results["stages_completed"].append("issue_categorization")

        # Stage 3: Strategy Planning
        print("\n=== Stage 3: Strategy Planning ===")
        planner = StrategyPlanner(model=model, progress=True)
        strategy_plan = planner.analyze_and_plan()
        strategy_file = planner.save_result(strategy_plan)
        results["strategy_file"] = strategy_file
        results["stages_completed"].append("strategy_planning")

        # Stage 4: Action Planning
        print("\n=== Stage 4: Action Planning ===")
        action_planner = ActionPlanner(model=model, progress=True)
        action_proposals = action_planner.generate_proposals()
        action_file = action_planner.save_result(action_proposals)
        results["action_file"] = action_file
        results["stages_completed"].append("action_planning")

    except Exception as e:
        results["errors"].append(str(e))
        raise

    return results


def display_summary(results: Dict[str, Any]) -> None:
    """
    Display workflow summary.

    Args:
        results: Workflow results dictionary
    """
    print("\n=== Optimization Workflow Complete ===")
    print(f"Projects: {', '.join(results['projects'])}")
    print(f"Stages completed: {', '.join(results['stages_completed'])}")

    print("\nGenerated files:")
    for key in ['validator_result_file', 'classification_file', 'category_file', 'strategy_file', 'action_file']:
        if key in results:
            print(f"  - {results[key]}")

    if results.get('errors'):
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimize knowledge extraction workflow by analyzing validation results"
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        required=True,
        help="Project names to optimize (space-separated)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="LLM model name (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/app/output",
        help="Base output directory (default: /workspace/app/output)"
    )

    args = parser.parse_args()

    print("\n=== Knowledge Extraction Optimization Workflow ===")
    print(f"Projects: {', '.join(args.projects)}")
    print(f"Model: {args.model}")

    try:
        results = run_optimization_workflow(
            projects=args.projects,
            model=args.model,
            output_dir=args.output_dir
        )

        display_summary(results)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
