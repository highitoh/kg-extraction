"""Issue Categorizer - Categorizes similar issue labels into canonical categories using LLM."""

import argparse
import asyncio
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_factory import LLMFactory
from utils.llm_retry import call_with_retry
from utils.prompt_utils import fill_prompt_template


class IssueCategorizer:
    """Categorizes similar issue labels into canonical categories using LLM."""

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        output_base_dir: str = "/workspace/app/output/optimize",
        progress: bool = True,
    ):
        """
        Initialize the IssueCategorizer.

        Args:
            model: LLM model name
            temperature: LLM temperature setting
            output_base_dir: Base directory for optimize outputs
            progress: Whether to show progress messages
        """
        self.model = model
        self.temperature = temperature
        self.output_base_dir = output_base_dir
        self.progress = progress

        # Resolve issue classification file path
        self.issue_classification_file = self._resolve_issue_classification_path(output_base_dir)

        # Load issue classification data
        self.issue_classification_data = self._load_json(self.issue_classification_file)

        # Extract metadata
        self.validator_result_file = self.issue_classification_data.get("validator_result_file", "")
        self.knowledge_file = self.issue_classification_data.get("knowledge_file", "")
        self.extraction_result_file = self.issue_classification_data.get("extraction_result_file", "")

        # Create LLM instance
        self.llm = LLMFactory.create_llm(model=model, temperature=temperature, timeout=180.0, reasoning_effort="medium")

        # Load schema (path: /workspace/app/schemas/issue-categorizer/)
        self.schemas_dir = Path(__file__).parent.parent.parent / "schemas" / "issue-categorizer"
        self.llm_schema = self._load_schema("llm.schema.json")

        # Bind LLM with JSON schema
        self.llm_bound = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "issue_categorization_response", "schema": self.llm_schema, "strict": True},
            }
        )

        # Load prompt template (path: /workspace/app/prompts/)
        self.prompt_template = self._load_prompt("issue_categorizer.txt")

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema file."""
        schema_path = self.schemas_dir / schema_name
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self, prompt_filename: str) -> str:
        """Load a prompt template."""
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_issue_classification_path(self, output_base_dir: str) -> str:
        """
        Resolve issue_classification file path.

        Args:
            output_base_dir: Base directory for optimize outputs

        Returns:
            Path to the most recent issue_classification_*.json file

        Raises:
            FileNotFoundError: If required files don't exist
        """
        optimize_dir = Path(output_base_dir)

        # Check if optimize directory exists
        if not optimize_dir.exists():
            raise FileNotFoundError(f"Optimize directory not found: {optimize_dir}")

        # Find most recent issue_classification file
        issue_classification_pattern = str(optimize_dir / "issue_classification_*.json")
        issue_classification_files = sorted(glob.glob(issue_classification_pattern), reverse=True)

        if not issue_classification_files:
            raise FileNotFoundError(
                f"No issue_classification_*.json files found in optimize directory: {optimize_dir}\n"
                f"Pattern searched: {issue_classification_pattern}\n"
                f"Please run IssueClassifier first."
            )

        # Use the most recent issue classification
        issue_classification_file = issue_classification_files[0]

        print(f"[IssueCategorizer] Resolved issue classification file: {issue_classification_file}")

        return str(issue_classification_file)

    def _extract_issue_labels(self) -> tuple[List[str], List[str]]:
        """
        Extract unique issue labels from class and property validations.

        Returns:
            Tuple of (class_issue_labels, property_issue_labels)
        """
        class_validations = self.issue_classification_data.get("class_validations", [])
        property_validations = self.issue_classification_data.get("property_validations", [])

        # Extract unique class issue labels
        class_issue_labels = set()
        for validation in class_validations:
            labels = validation.get("issue_labels", [])
            class_issue_labels.update(labels)

        # Extract unique property issue labels
        property_issue_labels = set()
        for validation in property_validations:
            labels = validation.get("issue_labels", [])
            property_issue_labels.update(labels)

        # Sort for consistent output
        return sorted(class_issue_labels), sorted(property_issue_labels)

    def _create_categorization_prompt(self, class_issue_labels: List[str], property_issue_labels: List[str]) -> str:
        """Create categorization prompt with all issue labels."""
        # Format class issue labels
        class_labels_text = (
            "\n".join([f"- {label}" for label in class_issue_labels]) if class_issue_labels else "(None)"
        )

        # Format property issue labels
        property_labels_text = (
            "\n".join([f"- {label}" for label in property_issue_labels]) if property_issue_labels else "(None)"
        )

        # Replace placeholders in template
        prompt = fill_prompt_template(
            self.prompt_template, class_issue_labels=class_labels_text, property_issue_labels=property_labels_text
        )

        return prompt

    async def categorize_async(self) -> Dict[str, Any]:
        """Categorize all issue labels."""
        print("[IssueCategorizer] Extracting issue labels from classifications...")

        # Extract unique issue labels
        class_issue_labels, property_issue_labels = self._extract_issue_labels()

        total_labels = len(class_issue_labels) + len(property_issue_labels)
        print(f"[IssueCategorizer] Found {len(class_issue_labels)} unique class issue labels")
        print(f"[IssueCategorizer] Found {len(property_issue_labels)} unique property issue labels")
        print(f"[IssueCategorizer] Total: {total_labels} unique issue labels")

        if total_labels == 0:
            print("[IssueCategorizer] No issue labels found. Nothing to categorize.")
            return {
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                "issue_classification_file": self.issue_classification_file,
                "categories": [],
            }

        # Create categorization prompt
        prompt = self._create_categorization_prompt(class_issue_labels, property_issue_labels)

        print("[IssueCategorizer] Invoking LLM for issue categorization...")

        try:
            # Invoke LLM
            response = await call_with_retry(
                lambda: self.llm_bound.ainvoke(prompt),
                timeout=180.0,
                label="IssueCategorizer",
            )

            # Handle response content
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                text_record = next((item for item in response.content if item.get("type") == "text"), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get("text", "{}"))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            if "categories" not in result:
                raise ValueError(f"Expected 'categories' in response, got: {result.keys()}")

            categories = result["categories"]
            print(f"[IssueCategorizer] LLM created {len(categories)} categories")

            # Map validations to categories
            categorized_output = self._map_validations_to_categories(categories)

            print("[IssueCategorizer] Categorization completed!")

            return categorized_output

        except Exception as e:
            print(f"[IssueCategorizer] Error during categorization: {e}")
            raise

    def _map_validations_to_categories(self, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map validation entries to their corresponding categories.

        Args:
            categories: List of category definitions from LLM

        Returns:
            Complete categorization output with validations
        """
        class_validations = self.issue_classification_data.get("class_validations", [])
        property_validations = self.issue_classification_data.get("property_validations", [])

        # Build category output with validations
        categorized = []

        for category in categories:
            category_name = category.get("category", "Unknown")
            category_description = category.get("description", "")
            category_labels = set(category.get("issue_labels", []))

            # Find all class validations that have at least one label in this category
            matching_class_validations = []
            for validation in class_validations:
                validation_labels = set(validation.get("issue_labels", []))
                if validation_labels & category_labels:  # Intersection check
                    matching_class_validations.append(validation)

            # Find all property validations that have at least one label in this category
            matching_property_validations = []
            for validation in property_validations:
                validation_labels = set(validation.get("issue_labels", []))
                if validation_labels & category_labels:  # Intersection check
                    matching_property_validations.append(validation)

            print(
                f"[IssueCategorizer] Category '{category_name}': {len(matching_class_validations)} classes, "
                f"{len(matching_property_validations)} properties"
            )

            categorized.append(
                {
                    "category": category_name,
                    "description": category_description,
                    "issue_labels": list(category_labels),
                    "validations": {
                        "class_validations": matching_class_validations,
                        "property_validations": matching_property_validations,
                    },
                }
            )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return {
            "timestamp": timestamp,
            "issue_classification_file": self.issue_classification_file,
            "categories": categorized,
        }

    def categorize(self) -> Dict[str, Any]:
        """Categorize all issue labels (synchronous wrapper)."""
        return asyncio.run(self.categorize_async())

    def save_report(self, report: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save categorization report to file."""
        if output_dir is None:
            # Default: save to optimize directory
            output_dir = Path(self.output_base_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = report["timestamp"]
        output_file = output_dir / f"issue_category_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Issue Categorizer - Categorizes similar issue labels into canonical categories"
    )
    parser.add_argument(
        "--model", default="gemini-3.1-flash-lite", help="LLM model name (default: gemini-3.1-flash-lite)"
    )

    args = parser.parse_args()

    try:
        # Create categorizer
        print(f"[IssueCategorizer] Initializing IssueCategorizer with model: {args.model}")
        categorizer = IssueCategorizer(model=args.model)

        # Run categorization
        print("[IssueCategorizer] Starting categorization process...")
        report = categorizer.categorize()

        # Save report
        output_file = categorizer.save_report(report)
        print("\n[IssueCategorizer] Categorization completed!")
        print(f"[IssueCategorizer] Report saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"\n[IssueCategorizer] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[IssueCategorizer] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
