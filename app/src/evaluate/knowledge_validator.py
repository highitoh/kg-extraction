"""Knowledge Validator - Validates classes and properties using LLM with parallel execution."""

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


class KnowledgeValidator:
    """Validates knowledge quality using LLM with parallel execution."""

    def __init__(
        self,
        project: str = "default",
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        output_base_dir: str = "/workspace/app/output",
    ):
        """
        Initialize the KnowledgeValidator.

        Args:
            project: Project name (resolves to /workspace/app/output/{project}/)
            model: LLM model name
            temperature: LLM temperature setting
            output_base_dir: Base directory for project outputs
        """
        self.project = project
        self.model = model
        self.temperature = temperature
        self.output_base_dir = output_base_dir

        # Resolve file paths from project name (do this FIRST before LLM initialization)
        self.knowledge_file, self.extraction_result_file = self._resolve_project_paths(project, output_base_dir)

        # Load data (do this EARLY to fail fast if files are invalid)
        self.knowledge_data = self._load_json(self.knowledge_file)
        self.extraction_result_data = self._load_json(self.extraction_result_file)

        # Cache metamodel data for heuristics / prompts
        self.metamodel_data = self.extraction_result_data.get('metamodel', {})
        self.metamodel_property_map = {
            prop.get('iri'): prop
            for prop in self.metamodel_data.get('properties', [])
            if prop.get('iri')
        }

        # Create LLM instance (only after files are validated)
        self.llm = LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            timeout=300.0,
            reasoning_effort="medium"
        )

        # Load schemas (path: /workspace/app/schemas/knowledge-validator/)
        self.schemas_dir = Path(__file__).parent.parent.parent / "schemas" / "knowledge-validator"
        self.class_schema = self._load_schema("llm-class.schema.json")
        self.property_schema = self._load_schema("llm-property.schema.json")

        # Bind LLM with JSON schemas (separate for class and property validation)
        self.llm_class = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_validation_response",
                    "schema": self.class_schema,
                    "strict": True
                }
            }
        )

        self.llm_property = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "property_validation_response",
                    "schema": self.property_schema,
                    "strict": True
                }
            }
        )

        # Load prompt templates (path: /workspace/app/prompts/)
        self.class_prompt_template = self._load_prompt("knowledge_validator_class.txt")
        self.property_prompt_template = self._load_prompt("knowledge_validator_property.txt")

        # Extract context
        self.document_context = self._extract_document_context()

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

    def _resolve_project_paths(self, project: str, output_base_dir: str) -> Tuple[str, str]:
        """
        Resolve knowledge.json and extraction_result file paths from project name.

        Args:
            project: Project name
            output_base_dir: Base directory for project outputs

        Returns:
            Tuple of (knowledge_file_path, extraction_result_file_path)

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

        # Find knowledge.json
        knowledge_file = project_dir / "knowledge.json"
        if not knowledge_file.exists():
            raise FileNotFoundError(
                f"knowledge.json not found in project directory: {project_dir}\n"
                f"Expected file: {knowledge_file}"
            )

        # Find most recent extraction_result file
        extraction_result_pattern = str(project_dir / "extraction_result_*.json")
        extraction_result_files = sorted(glob.glob(extraction_result_pattern), reverse=True)

        if not extraction_result_files:
            raise FileNotFoundError(
                f"No extraction_result_*.json files found in project directory: {project_dir}\n"
                f"Pattern searched: {extraction_result_pattern}"
            )

        # Use the most recent extraction result
        extraction_result_file = extraction_result_files[0]

        print(f"Resolved files for project '{project}':")
        print(f"  - Knowledge file: {knowledge_file}")
        print(f"  - Extraction result: {extraction_result_file}")

        return str(knowledge_file), str(extraction_result_file)

    def _extract_document_context(self) -> str:
        """Extract document context from extraction result."""
        context_parts = []

        # File information
        file_path = self.extraction_result_data.get('file_path', 'Unknown')
        file_type = self.extraction_result_data.get('file_type', 'Unknown')
        context_parts.append(f"**File**: {file_path}")
        context_parts.append(f"**File Type**: {file_type}")

        # Summary if available
        if 'summary' in self.extraction_result_data:
            summary = self.extraction_result_data['summary']
            if summary:
                context_parts.append(f"\n**Document Summary**:\n{summary}")

        return "\n".join(context_parts)

    def _create_class_validation_prompt(self, class_group: List[Dict[str, Any]]) -> str:
        """Create validation prompt for a group of classes with the same class_iri."""
        if not class_group:
            raise ValueError("class_group cannot be empty")

        class_iri = class_group[0].get('class_iri', 'Unknown')

        # Format class instances
        class_instances_parts = []
        for idx, class_data in enumerate(class_group, 1):
            class_id = class_data.get('id', 'Unknown')
            class_label = class_data.get('label', 'Unknown')
            sources = class_data.get('sources', [])

            instance_info = f"""
**Class {idx}:**
- ID: {class_id}
- Label: {class_label}
- Sources: {json.dumps(sources, indent=2, ensure_ascii=False)}
"""
            class_instances_parts.append(instance_info)

        class_instances = "\n".join(class_instances_parts)

        prompt = fill_prompt_template(
            self.class_prompt_template,
            document_context=self.document_context,
            class_iri=class_iri,
            class_instances=class_instances
        )

        return prompt

    def _create_property_validation_prompt(self, property_group: List[Dict[str, Any]]) -> str:
        """Create validation prompt for a group of properties with the same property_iri."""
        if not property_group:
            raise ValueError("property_group cannot be empty")

        property_iri = property_group[0].get('property_iri', 'Unknown')

        # Find property definition in metamodel
        property_def = self.metamodel_property_map.get(property_iri)
        property_def_str = f"**Property Definition:** {json.dumps(property_def, indent=2, ensure_ascii=False)}" if property_def else ""

        # Format property instances
        property_instances_parts = []
        for idx, property_data in enumerate(property_group, 1):
            property_id = property_data.get('id', 'Unknown')
            src_label = property_data.get('src_label', 'Unknown')
            dest_label = property_data.get('dest_label', 'Unknown')
            strength = property_data.get('strength', 0)

            instance_info = f"""
**Property {idx}:**
- ID: {property_id}
- Source Class: {src_label}
- Destination Class: {dest_label}
- Current Strength: {strength}
"""
            property_instances_parts.append(instance_info)

        property_instances = "\n".join(property_instances_parts)

        prompt = fill_prompt_template(
            self.property_prompt_template,
            document_context=self.document_context,
            property_iri=property_iri,
            property_def=property_def_str,
            property_instances=property_instances
        )

        return prompt

    async def _validate_class_async(self, class_group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a group of classes with the same class_iri asynchronously."""
        if not class_group:
            return []

        class_iri = class_group[0].get('class_iri', 'Unknown')
        prompt = self._create_class_validation_prompt(class_group)

        try:
            # Invoke LLM with class schema
            print(f"[KnowledgeValidator] Invoking LLM for class validation: {class_iri} ({len(class_group)} instances)")
            response = await call_with_retry(
                lambda: self.llm_class.ainvoke(prompt),
                timeout=300.0,
                progress=True,
                label="KnowledgeValidator-Class",
            )
            # Handle response content (can be string or structured content)
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                # Find the record whose type is "text"
                text_record = next((item for item in response.content if item.get('type') == 'text'), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            # Return the validations array
            if 'class_validations' in result:
                return result['class_validations']
            else:
                raise ValueError(f"Expected 'class_validations' in response, got: {result.keys()}")
        except Exception as e:
            print(f"[KnowledgeValidator] Error validating class for IRI {class_iri}: {e}")
            # Return default validations for all classes in the group
            return [
                {
                    "class_id": cls.get('id', 'Unknown'),
                    "class_iri": cls.get('class_iri', 'Unknown'),
                    "class_label": cls.get('label', 'Unknown'),
                    "necessity": {"score": 0, "justification": f"Error: {str(e)}"},
                    "understandability": {"score": 0, "justification": f"Error: {str(e)}"},
                    "unity": {"score": 0, "justification": f"Error: {str(e)}"},
                    "boundary_suitability": {"score": 0, "justification": f"Error: {str(e)}"},
                    "structural_validity": {"score": 0, "justification": f"Error: {str(e)}"},
                    "text_grounding": {"score": 0, "justification": f"Error: {str(e)}"}
                }
                for cls in class_group
            ]

    async def _validate_property_async(self, property_group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a group of properties with the same property_iri asynchronously."""
        if not property_group:
            return []

        property_iri = property_group[0].get('property_iri', 'Unknown')
        prompt = self._create_property_validation_prompt(property_group)

        try:
            # Invoke LLM with property schema
            print(f"[KnowledgeValidator] Invoking LLM for property validation: {property_iri} ({len(property_group)} instances)")
            response = await call_with_retry(
                lambda: self.llm_property.ainvoke(prompt),
                timeout=300.0,
                progress=True,
                label="KnowledgeValidator-Property",
            )
            # Handle response content (can be string or structured content)
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                # Find the record whose type is "text"
                text_record = next((item for item in response.content if item.get('type') == 'text'), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            # Return the validations array
            if 'property_validations' in result:
                return result['property_validations']
            else:
                raise ValueError(f"Expected 'property_validations' in response, got: {result.keys()}")
        except Exception as e:
            print(f"[KnowledgeValidator] Error validating property for IRI {property_iri}: {e}")
            # Return default validations for all properties in the group
            return [
                {
                    "property_id": prop.get('id', 'Unknown'),
                    "src_label": prop.get('src_label', 'Unknown'),
                    "property_iri": prop.get('property_iri', 'Unknown'),
                    "dest_label": prop.get('dest_label', 'Unknown'),
                    "relation_applicability": {"score": 0, "justification": f"Error: {str(e)}"},
                    "relation_unity": {"score": 0, "justification": f"Error: {str(e)}"},
                    "strength_validity": {"score": 0, "justification": f"Error: {str(e)}"},
                    "contextual_grounding": {"score": 0, "justification": f"Error: {str(e)}"}
                }
                for prop in property_group
            ]

    async def validate_all_async(self) -> Dict[str, Any]:
        """Validate all classes and properties in parallel (grouped by IRI)."""
        classes = self.knowledge_data.get('classes', [])
        properties = self.knowledge_data.get('properties', [])

        print(f"[KnowledgeValidator] Starting validation: {len(classes)} classes, {len(properties)} properties")

        # Group classes by class_iri
        from collections import defaultdict
        class_groups = defaultdict(list)
        for cls in classes:
            class_iri = cls.get('class_iri', 'Unknown')
            class_groups[class_iri].append(cls)

        print(f"[KnowledgeValidator] {len(class_groups)} class types found")
        print(f"[KnowledgeValidator] Starting parallel validation: {len(class_groups)} class types")

        # Validate classes with progress tracking
        total_class_groups = len(class_groups)
        class_counter = 0
        class_lock = asyncio.Lock()

        async def _wrapped_class_validation(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal class_counter
            result = await self._validate_class_async(group)

            async with class_lock:
                class_counter += 1
                if group:
                    class_iri = group[0].get('class_iri', 'Unknown')
                    print(f"[{class_counter}/{total_class_groups}] KnowledgeValidator: {class_iri} validated ({len(group)} instances)")

            return result

        class_tasks = [_wrapped_class_validation(group) for group in class_groups.values()]
        class_validation_groups = await asyncio.gather(*class_tasks)

        # Flatten the results
        class_validations = []
        for group in class_validation_groups:
            class_validations.extend(group)

        print(f"[KnowledgeValidator] Class validation completed: {len(class_validations)} validations generated")

        # Group properties by property_iri
        property_groups = defaultdict(list)
        for prop in properties:
            property_iri = prop.get('property_iri', 'Unknown')
            property_groups[property_iri].append(prop)

        print(f"[KnowledgeValidator] {len(property_groups)} property types found")
        print(f"[KnowledgeValidator] Starting parallel validation: {len(property_groups)} property types")

        # Validate properties with progress tracking
        total_property_groups = len(property_groups)
        property_counter = 0
        property_lock = asyncio.Lock()

        async def _wrapped_property_validation(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal property_counter
            result = await self._validate_property_async(group)

            async with property_lock:
                property_counter += 1
                if group:
                    property_iri = group[0].get('property_iri', 'Unknown')
                    print(f"[{property_counter}/{total_property_groups}] KnowledgeValidator: {property_iri} validated ({len(group)} instances)")

            return result

        property_tasks = [_wrapped_property_validation(group) for group in property_groups.values()]
        property_validation_groups = await asyncio.gather(*property_tasks)

        # Flatten the results
        property_validations = []
        for group in property_validation_groups:
            property_validations.extend(group)

        print(f"[KnowledgeValidator] Property validation completed: {len(property_validations)} validations generated")

        # Create validation report
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        report = {
            "validation_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "knowledge_file": self.knowledge_file,
            "extraction_result_file": self.extraction_result_file,
            "class_validations": class_validations,
            "property_validations": property_validations
        }

        print(f"[KnowledgeValidator] Validation completed: {len(class_validations)} classes, {len(property_validations)} properties validated")

        return report

    def validate(self) -> Dict[str, Any]:
        """Validate all classes and properties (synchronous wrapper)."""
        return asyncio.run(self.validate_all_async())

    def save_report(self, report: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save validation report to file."""
        if output_dir is None:
            # Default: save to project directory
            output_dir = Path(self.output_base_dir) / self.project
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = report['timestamp']
        # Changed filename from knowledge_validator to validator_result
        output_file = output_dir / f"validator_result_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Validator - Validates classes and properties using LLM")
    parser.add_argument(
        "--project",
        default="default",
        help="Project name (resolves to /workspace/app/output/{project}/) (default: default)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model name (default: gpt-5-mini)"
    )

    args = parser.parse_args()

    try:
        # Create validator
        print(f"Initializing KnowledgeValidator with model: {args.model}")
        validator = KnowledgeValidator(
            project=args.project,
            model=args.model
        )

        # Run validation
        print("Starting validation process...")
        report = validator.validate()

        # Save report
        output_file = validator.save_report(report)
        print(f"\nValidation completed!")
        print(f"Report saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
