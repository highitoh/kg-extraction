from typing import Dict, Any, List, Union
import asyncio
import json
import os

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_utils import fill_prompt_template
from src.utils.llm_retry import call_with_retry


class PropertyConsolidator(Runnable):
    """
    Class to consolidate property candidates
    Determines similarity between newly extracted and existing properties using LLM and consolidates them

    Processing flow:
    1. Build class ID mapping (corresponding to classes consolidated by ClassConsolidator)
    2. Normalize src_id/dest_id of existing properties
    3. Group new + existing properties by property_iri
    4. Detect similar properties with LLM
    5. Merge similar properties (select highest strength)
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gemini-3.1-flash-lite-preview",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/property_consolidator"
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # Load metamodel
        self.metamodel = self._load_metamodel()
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        # Load custom rules
        self.custom_rules = self._load_custom_rules()

    def _load_metamodel(self) -> dict:
        """Load metamodel file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """Load prompt template file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        prompt_path = os.path.join(app_dir, "prompts", "property_consolidator.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_custom_rules(self) -> str:
        """Load custom rules file if it exists"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        custom_path = os.path.join(app_dir, "prompts", "custom", "property_consolidator_custom.txt")

        # Return empty string if file doesn't exist
        if not os.path.exists(custom_path):
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return "\n" + content if content else ""

    def _build_extraction_rules(self, property_iri: str) -> str:
        """
        Get extract_rules for corresponding property from metamodel
        and generate text of rules related to property consolidation

        Parameters
        ----------
        property_iri : str
            Target property IRI

        Returns
        -------
        str
            Extraction rules string
        """
        extract_rules = []
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri") == property_iri:
                extract_rules = prop.get("extract_rules", [])
                break

        return "\n".join(extract_rules) if extract_rules else ""

    def _build_schema(self) -> dict:
        """Load JSON schema for LLM"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        schema_path = os.path.join(app_dir, "schemas", "property-consolidator", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Main entry point for property consolidation processing

        Parameters
        ----------
        input: PropertyConsolidatorInput
          - properties: List of newly extracted properties
          - existing_properties: List of existing properties (optional)
          - class_info: Class information (for ID normalization)

        Returns
        -------
        output: PropertyConsolidatorOutput (= PropertyExtractor output format)
          - properties: List of consolidated properties
        """
        new_properties = input.get("properties", [])
        existing_properties = input.get("existing_properties", [])
        class_info = input.get("class_info", {})

        if self.progress:
            print(f"[PropertyConsolidator] Processing {len(new_properties)} new + {len(existing_properties)} existing properties")

        # Return new properties as-is if no existing properties
        if not existing_properties:
            if self.progress:
                print("[PropertyConsolidator] No existing properties to merge")
            return {"properties": new_properties}

        # Build class ID mapping
        class_id_mapping = self._build_class_id_mapping(class_info)

        # Normalize IDs of existing properties
        normalized_existing = self._normalize_property_ids(existing_properties, class_id_mapping)

        # Filter out properties with invalid class references
        valid_normalized = []
        invalid_properties = []
        class_ids_set = set(class_id_mapping.values())  # All valid canonical IDs

        for prop in normalized_existing:
            src_id = prop.get("src_id", "")
            dest_id = prop.get("dest_id", "")

            if src_id in class_ids_set and dest_id in class_ids_set:
                valid_normalized.append(prop)
            else:
                # Track invalid properties for logging
                invalid_properties.append({
                    "property": prop,
                    "reason": f"src_id or dest_id not in current classes (src_id={src_id}, dest_id={dest_id})"
                })

        if invalid_properties and self.progress:
            print(f"[PropertyConsolidator] Filtered {len(invalid_properties)} properties with invalid class references")

        # Merge new + existing
        all_properties = new_properties + valid_normalized

        if self.progress:
            print(f"[PropertyConsolidator] Total {len(all_properties)} properties to consolidate")

        # Group by property_iri
        property_groups = self._group_by_property_iri(all_properties)

        if self.progress:
            print(f"[PropertyConsolidator] {len(property_groups)} property types found")

        # Execute consolidation in parallel for each property
        all_consolidated_properties = asyncio.run(
            self._process_properties_parallel(property_groups, class_info)
        )

        # Build output
        output = {"properties": all_consolidated_properties}

        if self.progress:
            print(f"[PropertyConsolidator] Completed: {len(all_properties)} -> {len(all_consolidated_properties)} properties")

        self.logger.save_log(output, filename_prefix="property_consolidator_output_")

        return output

    def _build_class_id_mapping(self, class_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Build class ID mapping
        Convert merged class IDs to canonical IDs using consolidated_ids

        Returns:
            mapping: {class_id: canonical_id} for all classes
                     Includes both current classes (identity) and merged IDs
        """
        mapping = {}

        for cls in class_info.get("classes", []):
            canonical_id = cls.get("id")
            if canonical_id:
                # Current class: identity mapping
                mapping[canonical_id] = canonical_id

                # Merged classes: map old IDs to this canonical ID
                consolidated_ids = cls.get("consolidated_ids", [])
                for old_id in consolidated_ids:
                    mapping[old_id] = canonical_id

        return mapping

    def _normalize_property_ids(
        self,
        properties: List[Dict[str, Any]],
        class_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Normalize src_id/dest_id of properties
        Convert old class IDs to new canonical IDs
        """
        normalized = []
        for prop in properties:
            normalized_prop = prop.copy()
            normalized_prop["src_id"] = class_mapping.get(
                prop.get("src_id", ""),
                prop.get("src_id", "")
            )
            normalized_prop["dest_id"] = class_mapping.get(
                prop.get("dest_id", ""),
                prop.get("dest_id", "")
            )
            normalized.append(normalized_prop)
        return normalized

    def _group_by_property_iri(self, properties: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group by property_iri"""
        groups = {}
        for prop in properties:
            prop_iri = prop.get("property_iri", "")
            if prop_iri not in groups:
                groups[prop_iri] = []
            groups[prop_iri].append(prop)
        return groups

    def _get_property_name_from_iri(self, property_iri: str) -> str:
        """Get property name from property_iri"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri", "") == property_iri:
                return prop.get("name", "unknown")
        return property_iri.split("#")[-1] if "#" in property_iri else property_iri

    async def _process_properties_parallel(
        self,
        property_groups: Dict[str, List[Dict[str, Any]]],
        class_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process property groups in parallel"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(property_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[PropertyConsolidator] Starting parallel processing: {total} property types (concurrency={self.max_concurrency})")

        async def _wrapped(property_iri: str, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal counter
            async with sem:
                property_name = self._get_property_name_from_iri(property_iri)

                consolidated = await self._consolidate_properties_async(
                    properties, property_iri, class_info
                )

                async with lock:
                    counter += 1
                    if self.progress:
                        print(f"[{counter}/{total}] PropertyConsolidator: {property_name} processed ({len(properties)} -> {len(consolidated)} properties)")

                return consolidated

        results = await asyncio.gather(
            *[_wrapped(iri, props) for iri, props in property_groups.items()]
        )

        # Flatten results
        all_consolidated = []
        for result in results:
            all_consolidated.extend(result)

        return all_consolidated

    async def _consolidate_properties_async(
        self,
        properties: List[Dict[str, Any]],
        property_iri: str,
        class_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Consolidate property list with the same property_iri (async version)

        Args:
            properties: List of properties to consolidate
            property_iri: Target property IRI
            class_info: Class information (for label retrieval)

        Returns:
            List of consolidated properties
        """
        if not properties:
            return []

        # No consolidation needed if only one property
        if len(properties) == 1:
            return properties

        # Create class index
        class_index = self._index_classes_by_id(class_info)

        # Detect similar properties with LLM
        try:
            similar_groups = await self._detect_similar_properties_async(
                properties, property_iri, class_index
            )
        except Exception as e:
            if self.progress:
                print(f"[PropertyConsolidator] LLM detection failed for {property_iri}: {e}")
            # On LLM failure, return all individually as if no duplicates
            return properties

        # Merge each group
        consolidated_properties = []
        processed_indices = set()

        for group_indices in similar_groups:
            if not group_indices:
                continue

            group_properties = [properties[i] for i in group_indices if i < len(properties)]
            if not group_properties:
                continue

            # Merge group
            merged = self._merge_properties(group_properties)
            consolidated_properties.append(merged)

            # Record processed indices
            processed_indices.update(group_indices)

        # Add properties that were not merged
        for i, prop in enumerate(properties):
            if i not in processed_indices:
                consolidated_properties.append(prop)

        return consolidated_properties

    async def _detect_similar_properties_async(
        self,
        properties: List[Dict[str, Any]],
        property_iri: str,
        class_index: Dict[str, Dict[str, Any]]
    ) -> List[List[int]]:
        """
        Detect similar properties using LLM

        Args:
            properties: Property list
            property_iri: Property IRI
            class_index: Mapping of class_id -> class information

        Returns:
            List of index groups for similar properties
            Example: [[0, 2, 5], [1, 3], [4]] - 0,2,5 are similar, 1,3 are similar, 4 is independent
        """
        # Format property information
        property_list_text = self._format_properties_for_llm(
            properties, property_iri, class_index
        )

        # Build prompt
        property_name = self._get_property_name_from_iri(property_iri)
        extraction_rules = self._build_extraction_rules(property_iri)
        prompt = fill_prompt_template(
            self.prompt_template,
            property_name=property_name,
            extraction_rules=extraction_rules,
            custom_rules=self.custom_rules,
            property_list=property_list_text
        )

        # Build LLM schema
        schema = self._build_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "property_consolidator",
                    "schema": schema
                }
            }
        )

        msg = HumanMessage(content=[{"type": "text", "text": prompt}])

        # Execute grouping with LLM (with retry)
        response = await call_with_retry(
            lambda: llm_json.ainvoke([msg]),
            timeout=90.0,
            label="PropertyConsolidator",
        )

        # Extract text from response
        result_text = self._to_text(response.content)

        # Parse JSON
        try:
            parsed = json.loads(result_text)
            groups = parsed.get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print(f"[PropertyConsolidator] JSON parse error: {result_text}")
            return []

        # Extract index lists of groups
        similar_groups = []
        for group in groups:
            indices = group.get("indices", [])
            if len(indices) > 1:  # Only when 2 or more properties are similar
                similar_groups.append(indices)

        return similar_groups

    def _format_properties_for_llm(
        self,
        properties: List[Dict[str, Any]],
        property_iri: str,
        class_index: Dict[str, Dict[str, Any]]
    ) -> str:
        """Format property list for LLM"""
        lines = []
        for i, prop in enumerate(properties):
            src_id = prop.get("src_id", "")
            dest_id = prop.get("dest_id", "")

            src = class_index.get(src_id, {})
            dest = class_index.get(dest_id, {})

            src_label = src.get("label", "N/A")
            dest_label = dest.get("label", "N/A")

            # Include source text (if available)
            src_sources = src.get("sources", [])
            dest_sources = dest.get("sources", [])
            src_source = src_sources[0] if src_sources else ""
            dest_source = dest_sources[0] if dest_sources else ""

            lines.append(f"{i}:")
            lines.append(f"  source: {src_label}")
            lines.append(f"    context: {src_source[:200]}" if src_source else "")
            lines.append(f"  destination: {dest_label}")
            lines.append(f"    context: {dest_source[:200]}" if dest_source else "")
            lines.append(f"  strength: {prop.get('strength', 0)}")
            lines.append("")

        return "\n".join(lines)

    def _merge_properties(self, group_properties: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge property group
        Select property with highest strength

        Args:
            group_properties: List of properties to merge

        Returns:
            Merged property
        """
        if len(group_properties) == 1:
            return group_properties[0]

        # Sort by strength (descending)
        sorted_props = sorted(
            group_properties,
            key=lambda p: p.get("strength", 0),
            reverse=True
        )

        # Return property with highest strength
        return sorted_props[0].copy()

    @staticmethod
    def _to_text(content: Union[str, List[Any]]) -> str:
        """Extract text from LLM response"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return ""

    @staticmethod
    def _index_classes_by_id(class_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create mapping of class_id -> class information"""
        index = {}
        for cls in class_info.get("classes", []):
            cls_id = cls.get("id")
            if cls_id:
                index[cls_id] = cls
        return index


if __name__ == "__main__":
    import glob

    # Get latest property_extractor log file from log directory
    extractor_log_dir = "/workspace/app/log/property_extractor"
    extractor_log_files = glob.glob(os.path.join(extractor_log_dir, "property_extractor_output_*.json"))

    # Get class consolidator logs
    class_log_dir = "/workspace/app/log/class_consolidator"
    class_log_files = glob.glob(os.path.join(class_log_dir, "class_consolidator_output_*.json"))

    if not extractor_log_files or not class_log_files:
        print("Error: Required log files not found")
        print(f"  PropertyExtractor logs: {len(extractor_log_files)}")
        print(f"  ClassConsolidator logs: {len(class_log_files)}")
        exit(1)

    # Sort PropertyExtractor logs by modification time (newest first)
    sorted_extractor_logs = sorted(extractor_log_files, key=os.path.getmtime, reverse=True)

    # Get latest and second latest PropertyExtractor logs
    latest_extractor_log = sorted_extractor_logs[0]
    second_latest_extractor_log = sorted_extractor_logs[1] if len(sorted_extractor_logs) >= 2 else None

    # Get latest ClassConsolidator log
    latest_class_log = max(class_log_files, key=os.path.getmtime)

    print(f"Loading new properties from: {latest_extractor_log}")
    if second_latest_extractor_log:
        print(f"Loading existing properties from: {second_latest_extractor_log}")
    else:
        print("No second PropertyExtractor log found - using empty existing_properties")
    print(f"Loading ClassConsolidator output from: {latest_class_log}")

    with open(latest_extractor_log, "r", encoding="utf-8") as f:
        property_data = json.load(f)

    # Load existing properties from second latest log if available
    existing_property_data = {}
    if second_latest_extractor_log:
        with open(second_latest_extractor_log, "r", encoding="utf-8") as f:
            existing_property_data = json.load(f)

    with open(latest_class_log, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    # Build test input data
    sample_input = {
        "id": property_data.get("id", "test-id"),
        "properties": property_data.get("properties", []),
        "existing_properties": existing_property_data.get("properties", []),
        "class_info": class_data
    }

    print(f"\nInput data:")
    print(f"  - New properties: {len(sample_input['properties'])}")
    print(f"  - Existing properties: {len(sample_input['existing_properties'])}")
    print(f"  - Classes: {len(class_data.get('classes', []))}")

    # Execute PropertyConsolidator
    consolidator = PropertyConsolidator(progress=True)
    result = consolidator.invoke(sample_input)

    print(f"\nOutput data:")
    print(f"  - Consolidated properties: {len(result.get('properties', []))}")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:500])
