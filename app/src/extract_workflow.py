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


def run_knowledge_extraction(pdf_path: str, output_dir: str = "/workspace/app/output", model: str = None, project: str = "default") -> Dict[str, Any]:
    """
    Execute knowledge extraction from PDF file

    Args:
        pdf_path: PDF file path to extract from
        output_dir: Output directory
        model: LLM model name (if None, use default for each node)
        project: Project name (output files classified by project)

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

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract knowledge from PDF files and generate Turtle and Neo4j CSV files")
    parser.add_argument("pdf_path", nargs="?", default="../doc/sample.pdf", help="PDF file path to extract from")
    parser.add_argument("--model", type=str, default=None, help="LLM model name to use (default: default model for each node)")
    parser.add_argument("--output-dir", type=str, default="/workspace/app/output", help="Output directory (default: /workspace/app/output)")
    parser.add_argument("--project", type=str, default="default", help="Project name (default: default)")

    args = parser.parse_args()
    pdf_path = args.pdf_path

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print("\n=== Knowledge Extraction ===")
    print(f"PDF file: {pdf_path}")
    print(f"Project: {args.project}")
    if args.model:
        print(f"LLM model: {args.model}")
    else:
        print(f"LLM model: Using default models for each node")

    try:
        result = run_knowledge_extraction(pdf_path, output_dir=args.output_dir, model=args.model, project=args.project)

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

            # Save results to JSON file
            project_output_dir = os.path.join(args.output_dir, args.project)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_json = os.path.join(project_output_dir, f"extraction_result_{timestamp}.json")
            with open(output_json, "w", encoding="utf-8") as f:
                # Exclude turtle_content when saving as it's too large
                result_copy = result.copy()
                result_copy.pop("turtle_content", None)
                json.dump(result_copy, f, ensure_ascii=False, indent=2)

            print(f"  - Result saved to: {output_json}")

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