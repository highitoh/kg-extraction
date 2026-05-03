"""
Knowledge Graph Extraction Main
Extract knowledge from PDF files and generate Turtle and Neo4j CSV files
"""

import os
import json
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

from src.extract.core.graph import create_knowledge_extraction_graph
from src.extract.core.state import KnowledgeExtractionState
from src.utils.snapshot_manager import SnapshotManager


def run_knowledge_extraction(pdf_path: str, output_dir: str = "/workspace/app/output", model: str = None, project: str = "default", create_snapshot: bool = True) -> Dict[str, Any]:
    """
    Execute knowledge extraction from PDF file

    Args:
        pdf_path: PDF file path to extract from
        output_dir: Output directory
        model: LLM model name (if None, use default for each node)
        project: Project name (output files classified by project)
        create_snapshot: If True, create snapshot after successful extraction

    Returns:
        Extraction result dictionary
    """
    # Create graph
    app = create_knowledge_extraction_graph()

    # Prepare initial State
    initial_state: KnowledgeExtractionState = {
        "files": [{"path": pdf_path}],
        "output_dir": output_dir,
        "project": project,
        "errors": []
    }

    # Add model to State if specified
    if model:
        initial_state["model"] = model

    # Execute graph
    result = app.invoke(initial_state)

    # Persist extraction result for downstream evaluation/reward pipeline
    project_output_dir = os.path.join(output_dir, project)
    os.makedirs(project_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_json = os.path.join(project_output_dir, f"extraction_result_{timestamp}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        result_copy = result.copy()
        # Exclude turtle content because it can be very large
        result_copy.pop("turtle_content", None)
        json.dump(result_copy, f, ensure_ascii=False, indent=2)

    # Create snapshot if requested and extraction was successful
    if create_snapshot and result.get("summary"):
        try:
            snapshot_mgr = SnapshotManager(output_dir=output_dir)

            # Check if Miro data exists in result
            raw_source_data = result.get("miro_raw_data")

            snapshot_result = snapshot_mgr.create_snapshot(
                project=project,
                trigger_type="extraction",
                source_pdf=pdf_path if not raw_source_data else None,
                raw_source_data=raw_source_data,
                extraction_result_file=output_json
            )
            result["snapshot"] = snapshot_result
            print(f"  - Snapshot created: {snapshot_result['snapshot_id']}")
        except Exception as e:
            # Don't fail extraction if snapshot creation fails
            print(f"\nWarning: Failed to create snapshot: {e}")
            result["snapshot_error"] = str(e)

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract knowledge from PDF files and generate Turtle and Neo4j CSV files")
    parser.add_argument("pdf_path", nargs="?", default="../doc/sample.pdf", help="PDF file path to extract from")
    parser.add_argument("--model", type=str, default=None, help="LLM model name to use (default: default model for each node)")
    parser.add_argument("--output-dir", type=str, default="/workspace/app/output", help="Output directory (default: /workspace/app/output)")
    parser.add_argument("--project", type=str, default="default", help="Project name (default: default)")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip snapshot creation (default: create snapshot)")

    args = parser.parse_args()
    pdf_path = args.pdf_path

    # Check if input is a URL
    is_url = pdf_path.startswith("http://") or pdf_path.startswith("https://")

    if not is_url and not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print("\n=== Knowledge Extraction ===")
    if is_url:
        print(f"Source: {pdf_path}")
    else:
        print(f"PDF file: {pdf_path}")
    print(f"Project: {args.project}")
    if args.model:
        print(f"LLM model: {args.model}")
    else:
        print(f"LLM model: Using default models for each node")

    try:
        create_snapshot = not args.no_snapshot
        result = run_knowledge_extraction(pdf_path, output_dir=args.output_dir, model=args.model, project=args.project, create_snapshot=create_snapshot)

        # Check for errors
        if result.get("errors"):
            print("\n=== Warnings/Errors ===")
            for error in result["errors"]:
                print(f"  - {error}")

        # Display results
        if "summary" in result:
            print("\n=== Extraction Complete ===")
            print(f"Final stage: {result.get('current_stage', 'unknown')}")
            print(f"\nGenerated files:")
            for file_path in result["summary"]["files_generated"]:
                print(f"  - {file_path}")

            print(f"\nSummary:")
            print(f"  - Classes extracted: {result['summary']['classes_count']}")
            print(f"  - Properties extracted: {result['summary']['properties_count']}")

            print("  - Result saved to project output as extraction_result_*.json")

        else:
            print("\n=== Extraction Failed ===")
            print(f"Extraction did not complete successfully.")
            if result.get("errors"):
                print("\nErrors encountered:")
                for error in result["errors"]:
                    print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
