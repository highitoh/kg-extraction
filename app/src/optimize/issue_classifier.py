"""Issue Classifier - Classifies issues in poor extractions using LLM with parallel execution."""

import asyncio
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import uuid
import glob

from openai import APIConnectionError, RateLimitError, APIError

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_factory import LLMFactory
from utils.llm_retry import call_with_retry
from utils.prompt_utils import fill_prompt_template


class IssueClassifier:
    """Classifies issues in poor extractions using LLM with parallel execution."""

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        temperature: float = 0.5,
        output_base_dir: str = "/workspace/app/output/optimize",
        progress: bool = True,
    ):
        """
        Initialize the IssueClassifier.

        Args:
            model: LLM model name
            temperature: LLM temperature setting (higher for creative label generation)
            output_base_dir: Base directory for optimize outputs
            progress: Whether to show progress messages
        """
        self.model = model
        self.temperature = temperature
        self.output_base_dir = output_base_dir
        self.progress = progress

        # Resolve validator result file path
        self.validator_result_file = self._resolve_validator_result_path(output_base_dir)

        # Load validator result data
        self.validator_result_data = self._load_json(self.validator_result_file)

        # Extract metadata
        self.knowledge_file = self.validator_result_data.get('knowledge_file', '')
        self.knowledge_files = self._resolve_knowledge_files(self.validator_result_data)
        self.extraction_result_file = self.validator_result_data.get('extraction_result_file', '')

        # Load extraction result to get metamodel
        self.extraction_result_data = self._load_json(self.extraction_result_file) if self.extraction_result_file else {}
        self.metamodel_data = self.extraction_result_data.get('metamodel', {})

        # Create LLM instance
        self.llm = LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="medium"
        )

        # Load schemas (path: /workspace/app/schemas/issue-classifier/)
        self.schemas_dir = Path(__file__).parent.parent.parent / "schemas" / "issue-classifier"
        self.class_schema = self._load_schema("llm-class.schema.json")
        self.property_schema = self._load_schema("llm-property.schema.json")

        # Bind LLM with JSON schemas (separate for class and property classification)
        self.llm_class = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_issue_classification_response",
                    "schema": self.class_schema,
                    "strict": True
                }
            }
        )

        self.llm_property = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "property_issue_classification_response",
                    "schema": self.property_schema,
                    "strict": True
                }
            }
        )

        # Load prompt templates (path: /workspace/app/prompts/)
        self.class_prompt_template = self._load_prompt("issue_classifier_class.txt")
        self.property_prompt_template = self._load_prompt("issue_classifier_property.txt")

        # Prepare metamodel context
        self.metamodel_context = self._create_metamodel_context()

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema file."""
        schema_path = self.schemas_dir / schema_name
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_prompt(self, prompt_filename: str) -> str:
        """Load a prompt template."""
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_filename
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_knowledge_files(self, validator_result_data: Dict[str, Any]) -> List[str]:
        """
        Resolve all knowledge files referenced by an aggregated validator result.

        Priority:
        1. validator_result_data["aggregated_from"][*]["knowledge_file"]
        2. validator_result_data["knowledge_file"] (fallback / backward compatibility)
        """
        knowledge_files: List[str] = []
        seen = set()

        for source in validator_result_data.get("aggregated_from", []):
            path = source.get("knowledge_file")
            if path and path not in seen:
                knowledge_files.append(path)
                seen.add(path)

        fallback = validator_result_data.get("knowledge_file")
        if fallback and fallback not in seen:
            knowledge_files.append(fallback)

        return knowledge_files

    def _resolve_validator_result_path(self, output_base_dir: str) -> str:
        """
        Resolve validator_result file path.

        Args:
            output_base_dir: Base directory for optimize outputs

        Returns:
            Path to the most recent validator_result_*.json file

        Raises:
            FileNotFoundError: If required files don't exist
        """
        optimize_dir = Path(output_base_dir)

        # Check if optimize directory exists
        if not optimize_dir.exists():
            raise FileNotFoundError(
                f"Optimize directory not found: {optimize_dir}"
            )

        # Find most recent validator_result file
        validator_result_pattern = str(optimize_dir / "validator_result_*.json")
        validator_result_files = sorted(glob.glob(validator_result_pattern), reverse=True)

        if not validator_result_files:
            raise FileNotFoundError(
                f"No validator_result_*.json files found in optimize directory: {optimize_dir}\n"
                f"Pattern searched: {validator_result_pattern}\n"
                f"Please run KnowledgeValidator first."
            )

        # Use the most recent validator result
        validator_result_file = validator_result_files[0]

        print(f"[IssueClassifier] Resolved validator result file: {validator_result_file}")

        return str(validator_result_file)

    def _create_metamodel_context(self) -> str:
        """Create metamodel context string for prompts."""
        if not self.metamodel_data:
            return "No metamodel information available."

        context_parts = []

        # Add class information
        classes = self.metamodel_data.get('classes', [])
        if classes:
            context_parts.append("**Classes:**")
            for cls in classes:
                cls_name = cls.get('name', 'Unknown')
                cls_iri = cls.get('iri', 'Unknown')
                cls_desc = cls.get('description', '')
                context_parts.append(f"- {cls_name} ({cls_iri}): {cls_desc}")

        # Add property information
        properties = self.metamodel_data.get('properties', [])
        if properties:
            context_parts.append("\n**Properties:**")
            for prop in properties:
                prop_name = prop.get('name', 'Unknown')
                prop_iri = prop.get('iri', 'Unknown')
                prop_desc = prop.get('description', '')
                src_class = prop.get('src_class', 'Unknown')
                dest_class = prop.get('dest_class', 'Unknown')
                context_parts.append(
                    f"- {prop_name} ({prop_iri}): {prop_desc} "
                    f"[{src_class} → {dest_class}]"
                )

        return "\n".join(context_parts)

    def _filter_poor_extractions(
        self,
        validations: List[Dict[str, Any]],
        dimension_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter validations with at least one dimension scored ≤ 3.

        Args:
            validations: List of validation results
            dimension_keys: List of dimension keys to check

        Returns:
            List of poor validations
        """
        poor_validations = []
        for validation in validations:
            has_poor_score = False
            for key in dimension_keys:
                if key in validation:
                    score = validation[key].get('score', 5)
                    if score <= 3:
                        has_poor_score = True
                        break

            if has_poor_score:
                poor_validations.append(validation)

        return poor_validations

    def _group_by_class_iri(
        self,
        class_validations: List[Dict[str, Any]],
        property_validations: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Group classes and properties by class IRI.

        Args:
            class_validations: List of class validation results (poor only)
            property_validations: List of property validation results (poor only)

        Returns:
            Dict mapping class_iri to (classes, properties) tuple
        """
        from collections import defaultdict

        # Create mapping: class_id -> class_iri
        class_id_to_iri = {}

        # Create mapping: class_label -> class_iri (for properties)
        class_label_to_iri = {}

        # Load all available knowledge files to build unified mappings.
        for knowledge_file in self.knowledge_files:
            knowledge_data = self._load_json(knowledge_file)

            for cls in knowledge_data.get('classes', []):
                cls_id = cls.get('id')
                cls_iri = cls.get('class_iri')
                if cls_id and cls_iri:
                    class_id_to_iri[cls_id] = cls_iri

                cls_label = cls.get('label')
                if cls_label and cls_iri:
                    class_label_to_iri[cls_label] = cls_iri

        # Group classes by IRI
        groups = defaultdict(lambda: ([], []))

        for cls_validation in class_validations:
            cls_id = cls_validation.get('class_id')
            cls_iri = class_id_to_iri.get(cls_id, 'Unknown')
            groups[cls_iri][0].append(cls_validation)

        # Group properties by source class IRI
        for prop_validation in property_validations:
            src_label = prop_validation.get('src_label')
            src_iri = class_label_to_iri.get(src_label, 'Unknown')
            groups[src_iri][1].append(prop_validation)

        # Convert defaultdict to regular dict
        return dict(groups)

    def _create_class_classification_prompt(self, class_iri: str, class_group: List[Dict[str, Any]]) -> str:
        """Create classification prompt for a group of classes with poor scores."""
        if not class_group:
            raise ValueError("class_group cannot be empty")

        # Format class instances with validation results
        class_instances_parts = []
        for idx, class_validation in enumerate(class_group, 1):
            class_id = class_validation.get('class_id', 'Unknown')
            class_label = class_validation.get('class_label', 'Unknown')

            # Format validation scores and justifications
            necessity = class_validation.get('necessity', {})
            understandability = class_validation.get('understandability', {})
            boundary = class_validation.get('boundary_suitability', {})
            structure = class_validation.get('structural_validity', {})

            instance_info = f"""
**Class {idx}:**
- ID: {class_id}
- Label: {class_label}
- Validation Results:
  - Necessity: {necessity.get('score', 'N/A')} - {necessity.get('justification', 'N/A')}
  - Understandability: {understandability.get('score', 'N/A')} - {understandability.get('justification', 'N/A')}
  - Boundary Suitability: {boundary.get('score', 'N/A')} - {boundary.get('justification', 'N/A')}
  - Structural Validity: {structure.get('score', 'N/A')} - {structure.get('justification', 'N/A')}
"""
            class_instances_parts.append(instance_info)

        class_instances = "\n".join(class_instances_parts)

        prompt = fill_prompt_template(
            self.class_prompt_template,
            metamodel_context=self.metamodel_context,
            class_iri=class_iri,
            class_instances=class_instances
        )

        return prompt

    def _create_property_classification_prompt(self, class_iri: str, property_group: List[Dict[str, Any]]) -> str:
        """Create classification prompt for a group of properties with poor scores."""
        if not property_group:
            raise ValueError("property_group cannot be empty")

        # Format property instances with validation results
        property_instances_parts = []
        for idx, property_validation in enumerate(property_group, 1):
            property_id = property_validation.get('property_id', 'Unknown')
            src_label = property_validation.get('src_label', 'Unknown')
            property_iri = property_validation.get('property_iri', 'Unknown')
            dest_label = property_validation.get('dest_label', 'Unknown')

            # Format validation scores and justifications
            relation_type = property_validation.get('relation_applicability', {})
            strength = property_validation.get('strength_validity', {})

            instance_info = f"""
**Property {idx}:**
- ID: {property_id}
- Source → Destination: {src_label} → {dest_label}
- Property IRI: {property_iri}
- Validation Results:
  - Relation Applicability: {relation_type.get('score', 'N/A')} - {relation_type.get('justification', 'N/A')}
  - Strength Validity: {strength.get('score', 'N/A')} - {strength.get('justification', 'N/A')}
"""
            property_instances_parts.append(instance_info)

        property_instances = "\n".join(property_instances_parts)

        prompt = fill_prompt_template(
            self.property_prompt_template,
            metamodel_context=self.metamodel_context,
            class_iri=class_iri,
            property_instances=property_instances
        )

        return prompt

    async def _classify_class_issues_async(
        self,
        class_iri: str,
        class_group: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Classify issues for a group of classes asynchronously."""
        if not class_group:
            return []

        prompt = self._create_class_classification_prompt(class_iri, class_group)

        try:
            # Invoke LLM with class schema
            print(f"[IssueClassifier] Invoking LLM for class issue classification: {class_iri} ({len(class_group)} instances)")
            response = await call_with_retry(
                lambda: self.llm_class.ainvoke(prompt),
                timeout=90.0,
                label="IssueClassifier-Class",
            )

            # Handle response content
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                text_record = next((item for item in response.content if item.get('type') == 'text'), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            # Return the classifications array
            if 'class_issues' in result:
                return result['class_issues']
            else:
                raise ValueError(f"Expected 'class_issues' in response, got: {result.keys()}")
        except Exception as e:
            print(f"[IssueClassifier] Error classifying class issues for IRI {class_iri}: {e}")
            # Return empty issue labels for all classes in the group
            return [
                {
                    "class_id": cls.get('class_id', 'Unknown'),
                    "issue_labels": []
                }
                for cls in class_group
            ]

    async def _classify_property_issues_async(
        self,
        class_iri: str,
        property_group: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Classify issues for a group of properties asynchronously."""
        if not property_group:
            return []

        prompt = self._create_property_classification_prompt(class_iri, property_group)

        try:
            # Invoke LLM with property schema
            print(f"[IssueClassifier] Invoking LLM for property issue classification: {class_iri} ({len(property_group)} instances)")
            response = await call_with_retry(
                lambda: self.llm_property.ainvoke(prompt),
                timeout=90.0,
                label="IssueClassifier-Property",
            )

            # Handle response content
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                text_record = next((item for item in response.content if item.get('type') == 'text'), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            # Return the classifications array
            if 'property_issues' in result:
                return result['property_issues']
            else:
                raise ValueError(f"Expected 'property_issues' in response, got: {result.keys()}")
        except Exception as e:
            print(f"[IssueClassifier] Error classifying property issues for IRI {class_iri}: {e}")
            # Return empty issue labels for all properties in the group
            return [
                {
                    "property_id": prop.get('property_id', 'Unknown'),
                    "issue_labels": []
                }
                for prop in property_group
            ]

    async def _classify_iri_group_async(
        self,
        class_iri: str,
        classes: List[Dict[str, Any]],
        properties: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Classify issues for an IRI group (both classes and properties)."""
        tasks = []

        # Add class classification task if there are classes
        if classes:
            tasks.append(self._classify_class_issues_async(class_iri, classes))
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        # Add property classification task if there are properties
        if properties:
            tasks.append(self._classify_property_issues_async(class_iri, properties))
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        # Execute both tasks concurrently
        class_issues, property_issues = await asyncio.gather(*tasks)

        return class_issues, property_issues

    async def classify_all_async(self) -> Dict[str, Any]:
        """Classify all issues in parallel (grouped by IRI)."""
        class_validations = self.validator_result_data.get('class_validations', [])
        property_validations = self.validator_result_data.get('property_validations', [])

        print(f"[IssueClassifier] Total validations: {len(class_validations)} classes, {len(property_validations)} properties")

        # Filter poor extractions (score ≤ 3)
        poor_classes = self._filter_poor_extractions(
            class_validations,
            ['necessity', 'understandability', 'boundary_suitability', 'structural_validity']
        )
        poor_properties = self._filter_poor_extractions(
            property_validations,
            ['relation_applicability', 'strength_validity']
        )

        print(f"[IssueClassifier] Poor extractions: {len(poor_classes)} classes, {len(poor_properties)} properties")

        # Group by class IRI
        iri_groups = self._group_by_class_iri(poor_classes, poor_properties)

        print(f"[IssueClassifier] {len(iri_groups)} IRI groups found")
        print(f"[IssueClassifier] Starting parallel classification: {len(iri_groups)} IRI groups")

        # Classify all IRI groups in parallel with progress tracking
        total_groups = len(iri_groups)
        group_counter = 0
        group_lock = asyncio.Lock()

        async def _wrapped_iri_classification(
            class_iri: str,
            classes: List[Dict[str, Any]],
            properties: List[Dict[str, Any]]
        ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
            nonlocal group_counter
            class_issues, property_issues = await self._classify_iri_group_async(class_iri, classes, properties)

            async with group_lock:
                group_counter += 1
                print(f"[{group_counter}/{total_groups}] IssueClassifier: {class_iri} classified ({len(classes)} classes, {len(properties)} properties)")

            return class_iri, class_issues, property_issues

        tasks = [
            _wrapped_iri_classification(iri, classes, properties)
            for iri, (classes, properties) in iri_groups.items()
        ]
        classification_results = await asyncio.gather(*tasks)

        # Build issue label mappings
        class_issue_map = {}  # class_id -> issue_labels
        property_issue_map = {}  # property_id -> issue_labels

        for iri, class_issues, property_issues in classification_results:
            for issue in class_issues:
                class_id = issue.get('class_id')
                issue_labels = issue.get('issue_labels', [])
                if class_id:
                    class_issue_map[class_id] = issue_labels

            for issue in property_issues:
                property_id = issue.get('property_id')
                issue_labels = issue.get('issue_labels', [])
                if property_id:
                    property_issue_map[property_id] = issue_labels

        print(f"[IssueClassifier] Classification completed: {len(class_issue_map)} classes, {len(property_issue_map)} properties with issue labels")

        # Merge issue labels into original validation data
        merged_class_validations = []
        for cls_validation in class_validations:
            cls_id = cls_validation.get('class_id')
            issue_labels = class_issue_map.get(cls_id, [])

            # Create a copy and add issue_labels
            merged_cls = cls_validation.copy()
            merged_cls['issue_labels'] = issue_labels
            merged_class_validations.append(merged_cls)

        merged_property_validations = []
        for prop_validation in property_validations:
            prop_id = prop_validation.get('property_id')
            issue_labels = property_issue_map.get(prop_id, [])

            # Create a copy and add issue_labels
            merged_prop = prop_validation.copy()
            merged_prop['issue_labels'] = issue_labels
            merged_property_validations.append(merged_prop)

        # Create classification report
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        report = {
            "classification_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "validator_result_file": self.validator_result_file,
            "knowledge_file": self.knowledge_file,
            "knowledge_files": self.knowledge_files,
            "extraction_result_file": self.extraction_result_file,
            "class_validations": merged_class_validations,
            "property_validations": merged_property_validations
        }

        print(f"[IssueClassifier] Classification completed!")

        return report

    def classify(self) -> Dict[str, Any]:
        """Classify all issues (synchronous wrapper)."""
        return asyncio.run(self.classify_all_async())

    def save_report(self, report: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save classification report to file."""
        if output_dir is None:
            # Default: save to optimize directory
            output_dir = Path(self.output_base_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = report['timestamp']
        output_file = output_dir / f"issue_classification_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue Classifier - Classifies issues in poor extractions using LLM")
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        help="LLM model name (default: gemini-3.1-flash-lite-preview)"
    )

    args = parser.parse_args()

    try:
        # Create classifier
        print(f"[IssueClassifier] Initializing IssueClassifier with model: {args.model}")
        classifier = IssueClassifier(
            model=args.model
        )

        # Run classification
        print("[IssueClassifier] Starting classification process...")
        report = classifier.classify()

        # Save report
        output_file = classifier.save_report(report)
        print(f"\n[IssueClassifier] Classification completed!")
        print(f"[IssueClassifier] Report saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"\n[IssueClassifier] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[IssueClassifier] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
