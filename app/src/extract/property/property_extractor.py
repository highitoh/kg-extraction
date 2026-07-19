import asyncio
import json
import os
from typing import Any, Dict, List, Union

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.utils.id_generator import generate_property_id
from src.utils.llm_factory import LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger
from src.utils.prompt_utils import fill_prompt_template


def _normalize_label(label: str) -> str:
    """Normalize a label for matching: trim, lowercase, collapse whitespace."""
    if not label:
        return ""
    return " ".join(label.split()).strip().lower()


class PropertyExtractor(Runnable):
    """
    Property (relationship) extraction
    Input: PropertyChainInput (class_info, metamodel)
    Output: PropertyChainOutput (properties)
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/property_extractor",
        batch_size: int = 20,
        metamodel_path: str = None,
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )

        # Load JSON schema
        schema = self._load_schema()
        self.llm_json = self.llm.bind(
            response_format={"type": "json_schema", "json_schema": {"name": "property_extractor", "schema": schema}}
        )
        self.max_concurrency = max_concurrency
        self.progress = progress
        self.batch_size = batch_size

        # Load metamodel
        self.metamodel_path = metamodel_path
        self.metamodel = self._load_metamodel()

        # Load prompt
        self.prompt = self._load_prompt()

        # Load custom rules
        self.custom_rules = self._load_custom_rules()

        # Logger setup
        self.logger = Logger(log_dir)

    def _load_schema(self) -> dict:
        """Load JSON schema file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        schema_path = os.path.join(app_dir, "schemas", "property-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metamodel(self) -> dict:
        """Load metamodel file"""
        metamodel_path = self.metamodel_path
        if not metamodel_path:
            # Go up 3 levels from src/extract/XXX/ to /workspace/app/
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load prompt"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        prompt_path = os.path.join(app_dir, "prompts", "property_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_custom_rules(self) -> str:
        """Load custom rules file if it exists"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        custom_path = os.path.join(app_dir, "prompts", "custom", "property_extractor_custom.txt")

        # Return empty string if file doesn't exist
        if not os.path.exists(custom_path):
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return "\n" + content if content else ""

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """Convert LLM output to text format"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    @staticmethod
    def _get_sources(cls: Dict[str, Any], max_sources: int = 3) -> List[str]:
        """Get class sources (up to max_sources) with backward compatibility."""
        sources = cls.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        if not sources and cls.get("source"):
            sources = [cls.get("source", "")]
        return [s for s in sources[:max_sources] if isinstance(s, str)]

    def _format_instances(self, instances: List[Dict[str, Any]]) -> str:
        """Format class instances with source excerpts for extractor prompts."""
        lines = []
        for inst in instances:
            lines.append(f"- ID: {inst.get('id', '')}, Label: {inst.get('label', '')}")
            sources = self._get_sources(inst)
            if sources:
                for i, source in enumerate(sources):
                    lines.append(f"  - source[{i}]: {source[:200]}")
            else:
                lines.append("  - source[0]: ")
        return "\n".join(lines)

    async def _extract_properties_for_batch(
        self,
        property_label: str,
        property_iri: str,
        property_definition: str,
        property_extract_rules: str,
        src_class_label: str,
        dest_class_label: str,
        src_class_definition: str,
        dest_class_definition: str,
        src_batch: List[Dict[str, Any]],
        dest_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Execute property extraction per batch"""

        # Format input to pass to LLM
        src_instances_text = self._format_instances(src_batch)
        dest_instances_text = self._format_instances(dest_batch)

        # Embed data into prompt template
        prompt_text = fill_prompt_template(
            self.prompt,
            src_class_label=src_class_label,
            dest_class_label=dest_class_label,
            property_label=property_label,
            src_class_definition=src_class_definition,
            dest_class_definition=dest_class_definition,
            property_definition=property_definition,
            extract_rules=property_extract_rules,
            custom_rules=self.custom_rules,
            src_instances=src_instances_text,
            dest_instances=dest_instances_text,
        )

        msg = HumanMessage(content=[{"type": "text", "text": prompt_text}])

        ai = await call_with_retry(
            lambda: self.llm_json.ainvoke([msg]),
            timeout=90.0,
            label="PropertyExtractor",
        )
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        # Organize property information
        properties = []
        extracted_properties = data.get("properties", [])
        if isinstance(extracted_properties, list):
            for prop in extracted_properties:
                if not isinstance(prop, dict):
                    continue
                src_id = prop.get("src_id", "")
                dest_id = prop.get("dest_id", "")
                if src_id and dest_id:
                    properties.append(
                        {"src_id": src_id.strip(), "property_iri": property_iri, "dest_id": dest_id.strip()}
                    )

        return properties

    async def _extract_properties_for_one_type(
        self, property_def: Dict[str, Any], classes: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract relationships between classes for a specific property type"""

        property_label = property_def.get("name", "")
        property_iri = property_def.get("iri", "")
        property_definition = property_def.get("description", "")
        property_extract_rules = property_def.get("extract_rules", [])
        src_class_iri = property_def.get("src_class", "")
        dest_class_iri = property_def.get("dest_class", "")

        # Get class definitions from metamodel
        classes_def = self.metamodel.get("classes", [])
        src_class_def = next((c for c in classes_def if c.get("iri") == src_class_iri), None)
        dest_class_def = next((c for c in classes_def if c.get("iri") == dest_class_iri), None)

        if not src_class_def or not dest_class_def:
            return []

        src_class_label = src_class_def.get("name", "")
        dest_class_label = dest_class_def.get("name", "")
        src_class_definition = src_class_def.get("description", "")
        dest_class_definition = dest_class_def.get("description", "")

        # Extract instances of target classes
        src_classes = [c for c in classes if c.get("class_iri") == src_class_iri]
        dest_classes = [c for c in classes if c.get("class_iri") == dest_class_iri]

        # Skip if no matching class instances
        if not src_classes or not dest_classes:
            return []

        # Optimization: Only extract properties for pairs involving at least one new class
        # Separate new and existing classes
        src_new = [c for c in src_classes if c.get("_is_new", False)]
        src_existing = [c for c in src_classes if not c.get("_is_new", False)]
        dest_new = [c for c in dest_classes if c.get("_is_new", False)]
        dest_existing = [c for c in dest_classes if not c.get("_is_new", False)]

        # If no new classes on either side, skip this property type entirely
        if not src_new and not dest_new:
            if self.progress:
                print(f"  [SKIP] {property_label}: No new classes for {src_class_label}→{dest_class_label}")
            return []

        # Create batches for three scenarios: new×new, new×existing, existing×new
        # (Skip existing×existing - already extracted in previous runs)
        batch_pairs = []

        # 1. new×new pairs
        if src_new and dest_new:
            src_new_batches = [src_new[i : i + self.batch_size] for i in range(0, len(src_new), self.batch_size)]
            dest_new_batches = [dest_new[i : i + self.batch_size] for i in range(0, len(dest_new), self.batch_size)]
            batch_pairs.extend([(s, d) for s in src_new_batches for d in dest_new_batches])

        # 2. new×existing pairs
        if src_new and dest_existing:
            src_new_batches = [src_new[i : i + self.batch_size] for i in range(0, len(src_new), self.batch_size)]
            dest_existing_batches = [
                dest_existing[i : i + self.batch_size] for i in range(0, len(dest_existing), self.batch_size)
            ]
            batch_pairs.extend([(s, d) for s in src_new_batches for d in dest_existing_batches])

        # 3. existing×new pairs
        if src_existing and dest_new:
            src_existing_batches = [
                src_existing[i : i + self.batch_size] for i in range(0, len(src_existing), self.batch_size)
            ]
            dest_new_batches = [dest_new[i : i + self.batch_size] for i in range(0, len(dest_new), self.batch_size)]
            batch_pairs.extend([(s, d) for s in src_existing_batches for d in dest_new_batches])

        if self.progress:
            total_pairs = sum(len(s) * len(d) for s, d in batch_pairs)
            print(f"  {property_label}: Processing {total_pairs} class pairs (new classes only)")

        # Generate tasks from filtered batch pairs
        tasks = []
        for src_batch, dest_batch in batch_pairs:
            tasks.append(
                self._extract_properties_for_batch(
                    property_label=property_label,
                    property_iri=property_iri,
                    property_definition=property_definition,
                    property_extract_rules="\n".join(
                        f"- {r}" for r in property_extract_rules if isinstance(r, str) and r.strip()
                    ),
                    src_class_label=src_class_label,
                    dest_class_label=dest_class_label,
                    src_class_definition=src_class_definition,
                    dest_class_definition=dest_class_definition,
                    src_batch=src_batch,
                    dest_batch=dest_batch,
                )
            )

        # Execute in parallel
        results = await asyncio.gather(*tasks)

        # Consolidate all results
        all_properties = []
        for result in results:
            all_properties.extend(result)

        return all_properties

    async def _extract_properties_from_classes(self, classes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract property candidates from class information"""

        # Get LLM-extractable property definitions from the metamodel
        # (deterministically derived properties are skipped)
        properties_def = [
            p for p in self.metamodel.get("properties", []) if p.get("extraction", "llm") != "deterministic"
        ]

        if not properties_def:
            return []

        total = len(properties_def)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(
                f"[PropertyExtractor] Starting parallel processing: {total} property types "
                f"(concurrency={self.max_concurrency})"
            )

        # Parallel processing for each property type
        async def _wrapped(property_def: Dict[str, Any]) -> List[Dict[str, str]]:
            nonlocal counter
            property_name = property_def.get("name", "")
            result = await self._extract_properties_for_one_type(property_def, classes)

            async with lock:
                counter += 1
                if self.progress:
                    print(
                        f"[{counter}/{total}] PropertyExtractor: {property_name} processed "
                        f"(extracted {len(result)} items)"
                    )

            return result

        results = await asyncio.gather(*[_wrapped(pd) for pd in properties_def])

        # Consolidate all results
        all_properties = []
        for result in results:
            all_properties.extend(result)

        return all_properties

    def _build_label_index(self, classes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Map normalized label -> list of class instances.

        Indexes both the primary `label` and any `consolidated_labels` so that
        instances merged away during consolidation are still resolvable.
        """
        index: Dict[str, List[Dict[str, Any]]] = {}
        for cls in classes:
            labels = [cls.get("label", "")] + list(cls.get("consolidated_labels", []) or [])
            for label in labels:
                norm = _normalize_label(label)
                if not norm:
                    continue
                index.setdefault(norm, []).append(cls)
        return index

    def _build_class_pair_property_map(self) -> Dict[tuple, str]:
        """Map (src_class_iri, dest_class_iri) -> property_iri from the metamodel."""
        pair_map: Dict[tuple, str] = {}
        for prop in self.metamodel.get("properties", []):
            src_class = prop.get("src_class")
            dest_class = prop.get("dest_class")
            iri = prop.get("iri")
            if src_class and dest_class and iri:
                pair_map[(src_class, dest_class)] = iri
        return pair_map

    def _extract_properties_from_structure(
        self,
        classes: List[Dict[str, Any]],
        extracted_structure: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        """Derive property candidates deterministically from structural edges.

        Each edge's src/dest labels are resolved to class instances (by label),
        then the class-pair is mapped to a metamodel property via src_class/
        dest_class. Direction is tried both ways to tolerate diagram direction
        uncertainty. Per-diagram scoping is not possible (class instances retain
        only the board-level file_id), so resolution is label-matching across the
        whole board; label collisions emit a candidate for every valid pair and
        are filtered downstream by the consolidator/validator.

        Returns (candidates, summary_counts).
        """
        label_index = self._build_label_index(classes)
        pair_map = self._build_class_pair_property_map()

        candidates: List[Dict[str, str]] = []
        seen = set()
        edges_parsed = len(extracted_structure)
        both_resolved = 0
        no_property = 0

        for edge in extracted_structure:
            src_instances = label_index.get(_normalize_label(edge.get("src_label", "")), [])
            dest_instances = label_index.get(_normalize_label(edge.get("dest_label", "")), [])
            if not src_instances or not dest_instances:
                continue
            both_resolved += 1

            matched_any = False
            for src in src_instances:
                for dest in dest_instances:
                    src_iri = src.get("class_iri", "")
                    dest_iri = dest.get("class_iri", "")

                    property_iri = pair_map.get((src_iri, dest_iri))
                    resolved_src, resolved_dest = src, dest
                    if property_iri is None:
                        # Try reversed direction; swap endpoints if only that matches.
                        property_iri = pair_map.get((dest_iri, src_iri))
                        if property_iri is not None:
                            resolved_src, resolved_dest = dest, src
                    if property_iri is None:
                        continue

                    matched_any = True
                    src_id = resolved_src.get("id", "")
                    dest_id = resolved_dest.get("id", "")
                    key = (src_id, property_iri, dest_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "src_id": src_id,
                            "property_iri": property_iri,
                            "dest_id": dest_id,
                        }
                    )

            if not matched_any:
                no_property += 1

        summary = {
            "edges_parsed": edges_parsed,
            "both_resolved": both_resolved,
            "no_metamodel_property": no_property,
            "candidates": len(candidates),
        }
        return candidates, summary

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Extract properties from PropertyChainInput and output PropertyChainOutput"""
        class_info = input["class_info"]
        classes = class_info.get("classes", [])
        extracted_structure = input.get("extracted_structure", []) or []

        # Execute property extraction
        extracted_properties = asyncio.run(self._extract_properties_from_classes(classes))

        # Derive deterministic candidates from structural edges (diagrams).
        # Structure candidates are merged FIRST so they win ties during
        # consolidation. They carry the frozen flags so the review loop trusts
        # them (skip LLM review) while still passing the validator.
        structure_candidates = []
        structure_summary = {}
        if extracted_structure:
            structure_candidates, structure_summary = self._extract_properties_from_structure(
                classes, extracted_structure
            )
            if self.progress:
                print(
                    f"[PropertyExtractor] Structure: {structure_summary['edges_parsed']} edges, "
                    f"{structure_summary['both_resolved']} resolved, "
                    f"{structure_summary['candidates']} candidates"
                )

        # Tag candidates with their origin and the flags that control downstream
        # strength/review handling.
        tagged_structure = [
            {
                **c,
                "_structure_derived": True,
                "strength": 2,
                "_frozen_from_existing_accept": True,
                "_frozen_strength": 2,
            }
            for c in structure_candidates
        ]
        tagged_llm = [{**c, "strength": 0} for c in extracted_properties]

        # Organize to match output format
        properties = []
        properties_with_labels = []
        for prop in tagged_structure + tagged_llm:
            prop_id = generate_property_id(prop["src_id"], prop["property_iri"], prop["dest_id"])

            # Schema-compliant property information (preserve control flags)
            property_item = {
                "id": prop_id,
                "src_id": prop["src_id"],
                "property_iri": prop["property_iri"],
                "dest_id": prop["dest_id"],
                "strength": prop.get("strength", 0),
            }
            for flag in ("_structure_derived", "_frozen_from_existing_accept", "_frozen_strength"):
                if flag in prop:
                    property_item[flag] = prop[flag]
            properties.append(property_item)

            # Add label information for log output
            src_class = next((c for c in classes if c["id"] == prop["src_id"]), None)
            dest_class = next((c for c in classes if c["id"] == prop["dest_id"]), None)

            src_label = src_class["label"] if src_class else prop["src_id"]
            dest_label = dest_class["label"] if dest_class else prop["dest_id"]

            properties_with_labels.append(
                {
                    **property_item,
                    "src_label": src_label,
                    "dest_label": dest_label,
                }
            )

        output: Dict[str, Any] = {"properties": properties}

        if self.progress:
            total_properties = len(properties)
            print(
                f"[PropertyExtractor] Completed: extracted {total_properties} properties "
                f"({len(tagged_structure)} from structure, {len(tagged_llm)} from LLM)"
            )

        # Save property chain output with Logger (with label information + structure summary)
        log_output = {"properties": properties_with_labels}
        if structure_summary:
            log_output["structure_summary"] = structure_summary
        self.logger.save_log(log_output, filename_prefix="property_extractor_output_")

        return output


if __name__ == "__main__":
    import glob

    # Create PropertyExtractor instance
    extractor = PropertyExtractor()

    # Load the latest ClassReviewer output JSON
    class_files = sorted(
        glob.glob("/workspace/app/log/class_consolidator/class_consolidator_output_*.json"), reverse=True
    )

    if not class_files:
        print("Error: No ClassReviewer output files found in log/class_consolidator/")
        exit(1)

    latest_file = class_files[0]
    print(f"Loading latest ClassReviewer output: {latest_file}")

    with open(latest_file, "r", encoding="utf-8") as f:
        class_info = json.load(f)

    # Prepare input data for PropertyExtractor
    sample_input = {"class_info": class_info, "metamodel": {}}

    print("=== PropertyExtractor Test ===")
    print(f"Input: {len(class_info.get('classes', []))} classes loaded")
    print("\n" + "=" * 50 + "\n")

    try:
        # Execute PropertyExtractor
        result = extractor.invoke(sample_input)

        print("Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\nSummary:")
        print(f"- Total properties extracted: {len(result['properties'])}")

        # Display each property relationship
        for i, prop in enumerate(result["properties"], 1):
            # Find source and destination class labels for display
            src_class = next((c for c in class_info["classes"] if c["id"] == prop["src_id"]), None)
            dest_class = next((c for c in class_info["classes"] if c["id"] == prop["dest_id"]), None)

            src_label = src_class["label"] if src_class else prop["src_id"]
            dest_label = dest_class["label"] if dest_class else prop["dest_id"]

            print(f"- Property {i}: {src_label} --[{prop['property_iri']}]--> {dest_label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
