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

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_factory import LLMFactory


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

        # Load prompt template (path: /workspace/app/prompts/)
        self.prompt_template = self._load_prompt()

        # Extract context
        self.document_context = self._extract_document_context()
        self.metamodel_info = self._extract_metamodel_info()

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema file."""
        schema_path = self.schemas_dir / schema_name
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load the prompt template."""
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "knowledge_validator.txt"
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

    def _extract_metamodel_info(self) -> str:
        """Extract metamodel information from extraction result."""
        metamodel = self.metamodel_data

        info_parts = []
        info_parts.append("**Metamodel Classes:**")

        # Extract class information
        classes = metamodel.get('classes', [])
        for cls in classes:
            class_name = cls.get('name', 'Unknown')
            class_desc = cls.get('description', '')
            info_parts.append(f"- {class_name}: {class_desc}")

        info_parts.append("\n**Metamodel Properties:**")

        # Extract property information
        properties = metamodel.get('properties', [])
        for prop in properties:
            prop_name = prop.get('name', 'Unknown')
            prop_desc = prop.get('description', '')
            src_class = prop.get('src_class', '').split('/')[-1]
            dest_class = prop.get('dest_class', '').split('/')[-1]
            info_parts.append(f"- {prop_name}: {src_class} → {dest_class}")
            if prop_desc:
                info_parts.append(f"  {prop_desc}")

        return "\n".join(info_parts)

    def _create_class_validation_prompt(self, class_data: Dict[str, Any]) -> str:
        """Create validation prompt for a class."""
        class_id = class_data.get('id', 'Unknown')
        class_label = class_data.get('label', 'Unknown')
        class_iri = class_data.get('class_iri', 'Unknown')
        sources = class_data.get('sources', [])

        validation_type = f"""
### CLASS VALIDATION

**Class Information:**
- ID: {class_id}
- Label: {class_label}
- IRI: {class_iri}
- Sources: {json.dumps(sources, indent=2)}

**Validation Perspectives:**

1. **Necessity** (Score 1-5)
   - Evaluates whether the extracted class represents a concept that is fundamentally necessary within the purpose of the document and the domain.
   - Consider: Is this class essential to the domain? Does it represent a core concept?

2. **Understandability** (Score 1-5)
   - Evaluates whether the class name is intuitive and whether the meaning of the concept can be clearly understood.
   - Consider: Is the name clear and self-explanatory? Would domain experts understand this class immediately?

3. **Boundary Suitability** (Score 1-5)
   - Evaluates whether the scope of information included in the class is appropriate and checks whether it incorrectly contains elements that should be extracted as separate classes.
   - Consider: Is the scope well-defined? Are there concepts mixed in that should be separate?

4. **Conceptual Structure Validity** (Score 1-5)
   - Evaluates whether the level of abstraction and the hierarchical relationships of the class are appropriate within the domain.
   - Consider: Is the abstraction level appropriate? Does it fit well in the domain hierarchy?

Evaluate this class and provide scores with detailed justifications for each perspective.
"""

        output_format = """
{
  "class_validation": {
    "class_id": "{class_id}",
    "class_label": "{class_label}",
    "necessity": {
      "score": <1-5>,
      "justification": "<detailed explanation>"
    },
    "understandability": {
      "score": <1-5>,
      "justification": "<detailed explanation>"
    },
    "boundary_suitability": {
      "score": <1-5>,
      "justification": "<detailed explanation>"
    },
    "conceptual_structure_validity": {
      "score": <1-5>,
      "justification": "<detailed explanation>"
    }
  }
}
""".replace("{class_id}", class_id).replace("{class_label}", class_label)

        prompt = self.prompt_template.replace("{DOCUMENT_CONTEXT}", self.document_context)
        prompt = prompt.replace("{METAMODEL_INFO}", self.metamodel_info)
        prompt = prompt.replace("{VALIDATION_TYPE}", validation_type)
        prompt = prompt.replace("{OUTPUT_FORMAT}", output_format)

        return prompt

    def _create_property_validation_prompt(self, property_data: Dict[str, Any]) -> str:
        """Create validation prompt for a property."""
        property_id = property_data.get('id', 'Unknown')
        src_label = property_data.get('src_label', 'Unknown')
        property_iri = property_data.get('property_iri', 'Unknown')
        dest_label = property_data.get('dest_label', 'Unknown')
        strength = property_data.get('strength', 0)

        # Find property definition in metamodel
        metamodel = self.extraction_result_data.get('metamodel', {})
        properties = metamodel.get('properties', [])
        property_def = None
        for prop in properties:
            if prop.get('iri') == property_iri:
                property_def = prop
                break

        property_info = f"- Property Definition: {json.dumps(property_def, indent=2)}" if property_def else ""

        validation_type = f"""
### PROPERTY VALIDATION

**Property Information:**
- ID: {property_id}
- Source Class: {src_label}
- Property IRI: {property_iri}
- Destination Class: {dest_label}
- Current Strength: {strength}

{property_info}

**Validation Perspectives:**

1. **Relation Type Validity** (Score 1-5)
   - Evaluates whether the extracted property is correctly classified according to the relationship types defined in the metamodel.
   - Consider: Does this relationship match the expected property type? Are source and destination classes correct?

2. **Strength Validity** (Score 1-5)
   - Evaluates whether the strength of the relationship expressed by the property aligns with the actual strength of the conceptual relationship within the domain.
   - Strength scale: 0 (invalid), 1 (weak), 2 (moderate), 3 (strong)
   - Consider: Is the assigned strength appropriate for this relationship? Does it reflect the actual importance/strength in the domain?

Evaluate this property and provide scores with detailed justifications for each perspective.
"""

        output_format = """
{
  "property_validation": {
    "property_id": "{property_id}",
    "src_label": "{src_label}",
    "property_iri": "{property_iri}",
    "dest_label": "{dest_label}",
    "relation_type_validity": {
      "score": <1-5>,
      "justification": "<detailed explanation>"
    },
    "strength_validity": {
      "score": <1-5>,
      "justification": "<detailed explanation>"
    }
  }
}
""".replace("{property_id}", property_id).replace("{src_label}", src_label).replace("{property_iri}", property_iri).replace("{dest_label}", dest_label)

        prompt = self.prompt_template.replace("{DOCUMENT_CONTEXT}", self.document_context)
        prompt = prompt.replace("{METAMODEL_INFO}", self.metamodel_info)
        prompt = prompt.replace("{VALIDATION_TYPE}", validation_type)
        prompt = prompt.replace("{OUTPUT_FORMAT}", output_format)

        return prompt

    async def _validate_class_async(self, class_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single class asynchronously."""
        prompt = self._create_class_validation_prompt(class_data)

        try:
            # Invoke LLM with class schema
            response = await self.llm_class.ainvoke(prompt)
            # Handle response content (can be string or structured content)
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                # Extract text from structured content
                result = json.loads(response.content[0].get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            # Return the result directly if it has the validation structure
            if 'class_validation' in result:
                return result['class_validation']
            else:
                # The response might already be the class_validation object
                return result
        except Exception as e:
            print(f"Error validating class {class_data.get('id')}: {e}")
            # Return default validation on error
            return {
                "class_id": class_data.get('id', 'Unknown'),
                "class_label": class_data.get('label', 'Unknown'),
                "necessity": {"score": 0, "justification": f"Error: {str(e)}"},
                "understandability": {"score": 0, "justification": f"Error: {str(e)}"},
                "boundary_suitability": {"score": 0, "justification": f"Error: {str(e)}"},
                "conceptual_structure_validity": {"score": 0, "justification": f"Error: {str(e)}"}
            }

    async def _validate_property_async(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single property asynchronously."""
        prompt = self._create_property_validation_prompt(property_data)

        try:
            # Invoke LLM with property schema
            response = await self.llm_property.ainvoke(prompt)
            # Handle response content (can be string or structured content)
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                # Extract text from structured content
                result = json.loads(response.content[0].get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            # Return the result directly if it has the validation structure
            if 'property_validation' in result:
                return result['property_validation']
            else:
                # The response might already be the property_validation object
                return result
        except Exception as e:
            print(f"Error validating property {property_data.get('id')}: {e}")
            # Return default validation on error
            return {
                "property_id": property_data.get('id', 'Unknown'),
                "src_label": property_data.get('src_label', 'Unknown'),
                "property_iri": property_data.get('property_iri', 'Unknown'),
                "dest_label": property_data.get('dest_label', 'Unknown'),
                "relation_type_validity": {"score": 0, "justification": f"Error: {str(e)}"},
                "strength_validity": {"score": 0, "justification": f"Error: {str(e)}"}
            }

    async def validate_all_async(self) -> Dict[str, Any]:
        """Validate all classes and properties in parallel."""
        classes = self.knowledge_data.get('classes', [])
        properties = self.knowledge_data.get('properties', [])

        print(f"Starting validation: {len(classes)} classes, {len(properties)} properties")

        # Validate all classes in parallel
        print(f"Validating {len(classes)} classes in parallel...")
        class_tasks = [self._validate_class_async(cls) for cls in classes]
        class_validations = await asyncio.gather(*class_tasks)

        # Validate all properties in parallel
        print(f"Validating {len(properties)} properties in parallel...")
        property_tasks = [self._validate_property_async(prop) for prop in properties]
        property_validations = await asyncio.gather(*property_tasks)

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


def main():
    """Main entry point for CLI."""
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for validation report (default: /workspace/app/output/{project}/)"
    )
    parser.add_argument(
        "--output-base-dir",
        default="/workspace/app/output",
        help="Base directory for project outputs (default: /workspace/app/output)"
    )

    args = parser.parse_args()

    try:
        # Create validator
        print(f"Initializing KnowledgeValidator with model: {args.model}")
        validator = KnowledgeValidator(
            project=args.project,
            model=args.model,
            temperature=args.temperature,
            output_base_dir=args.output_base_dir
        )

        # Run validation
        print("Starting validation process...")
        report = validator.validate()

        # Save report
        output_file = validator.save_report(report, args.output_dir)
        print(f"\nValidation completed!")
        print(f"Report saved to: {output_file}")

        # Print summary
        print(f"\nSummary:")
        print(f"- Classes validated: {len(report['class_validations'])}")
        print(f"- Properties validated: {len(report['property_validations'])}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
