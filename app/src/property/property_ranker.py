from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory


class PropertyRanker(Runnable):
    """
    Property relationship strength evaluation (Ranking)
    Input: PropertyChainOutput (properties with strength=0)
    Output: PropertyChainOutput (properties with strength=0-3)
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/property_ranker",
        batch_size: int = 20,
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.llm_schema = self._load_llm_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "property_ranker", "schema": self.llm_schema}})
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.progress = progress
        self.batch_size = batch_size

        self.prompt = self._load_prompt()
        self.custom_rules = self._load_custom_rules()
        self.metamodel = self._load_metamodel()
        self.logger = Logger(log_dir)

    # ===== Utilities =====
    def _load_llm_schema(self) -> Dict[str, Any]:
        """Load LLM output schema"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        schema_path = os.path.join(app_dir, "schemas", "property-ranker", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        prompt_path = os.path.join(app_dir, "prompts", "property_ranker.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_custom_rules(self) -> str:
        """Load custom rules file if it exists"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        custom_path = os.path.join(app_dir, "prompts", "custom", "property_ranker_custom.txt")

        # Return empty string if file doesn't exist
        if not os.path.exists(custom_path):
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return "\n" + content if content else ""

    def _load_metamodel(self) -> Dict[str, Any]:
        """Load metamodel"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
    def _index_classes_by_id(class_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Index classes array in class_info to be retrievable by id"""
        idx = {}
        for c in class_info.get("classes", []):
            cid = c.get("id")
            if cid:
                idx[cid] = c
        return idx

    def _group_by_property_iri(self, props: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group candidate properties by property_iri"""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in props:
            prop_iri = p.get("property_iri", "")
            if prop_iri not in groups:
                groups[prop_iri] = []
            groups[prop_iri].append(p)
        return groups

    def _get_class_definition(self, class_iri: str) -> tuple[str, str]:
        """Get class definition from metamodel (label, definition)"""
        for cls in self.metamodel.get("classes", []):
            if cls.get("iri") == class_iri:
                return cls.get("name", ""), cls.get("description", "")
        return "", ""

    def _get_property_definition(self, property_iri: str) -> tuple[str, str, str, str]:
        """Get property definition from metamodel (label, definition, src_class_iri, dest_class_iri)"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri") == property_iri:
                return (
                    prop.get("name", ""),
                    prop.get("description", ""),
                    prop.get("src_class", ""),
                    prop.get("dest_class", "")
                )
        return "", "", "", ""

    def _format_instances(self, instances: List[Dict[str, Any]]) -> str:
        """Format instance list to text format"""
        lines = []
        for inst in instances:
            lines.append(f"- ID: {inst.get('id', '')}, Label: {inst.get('label', '')}, Source: {inst.get('source', '')}")
        return "\n".join(lines)

    def _format_property_candidates(self, batch: List[Dict[str, Any]], class_index: Dict[str, Dict[str, Any]]) -> str:
        """Format property candidates for evaluation to text format"""
        lines = []
        for i, p in enumerate(batch, 1):
            src = class_index.get(p.get("src_id", ""), {})
            dest = class_index.get(p.get("dest_id", ""), {})
            lines.append(f"Candidate {i}:")
            lines.append(f"  src_id: {p.get('src_id', '')}")
            lines.append(f"    - src_label: {src.get('label', '')}")
            lines.append(f"    - src_source: {src.get('source', '')}")
            lines.append(f"  dest_id: {p.get('dest_id', '')}")
            lines.append(f"    - dest_label: {dest.get('label', '')}")
            lines.append(f"    - dest_source: {dest.get('source', '')}")
        return "\n".join(lines)

    async def _rank_batch(
        self,
        batch: List[Dict[str, Any]],
        class_index: Dict[str, Dict[str, Any]],
        property_iri: str
    ) -> List[Dict[str, Any]]:
        """Evaluate strength of one batch of candidates with LLM"""

        # Get property definition
        property_label, property_definition, src_class_iri, dest_class_iri = self._get_property_definition(property_iri)
        src_class_label, src_class_definition = self._get_class_definition(src_class_iri)
        dest_class_label, dest_class_definition = self._get_class_definition(dest_class_iri)

        # Get instance list
        src_instances = [class_index[p.get("src_id", "")] for p in batch if p.get("src_id") in class_index]
        dest_instances = [class_index[p.get("dest_id", "")] for p in batch if p.get("dest_id") in class_index]

        # Format prompt
        property_candidates_text = self._format_property_candidates(batch, class_index)
        src_instances_text = self._format_instances(src_instances)
        dest_instances_text = self._format_instances(dest_instances)

        prompt_filled = self.prompt.format(
            src_class_label=src_class_label,
            src_class_definition=src_class_definition,
            dest_class_label=dest_class_label,
            dest_class_definition=dest_class_definition,
            property_label=property_label,
            property_definition=property_definition,
            property_candidates=property_candidates_text,
            src_instances=src_instances_text,
            dest_instances=dest_instances_text,
            custom_rules=self.custom_rules
        )

        msg = HumanMessage(content=[{"type": "text", "text": prompt_filled}])

        # LLM invocation with retry functionality
        ai = await self._invoke_with_retry(msg)
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        results: List[Dict[str, Any]] = []
        for r in data.get("relations", []):
            results.append({
                "src_id": r.get("src_id", ""),
                "dest_id": r.get("dest_id", ""),
                "strength": r.get("strength", 0),
                "justification": r.get("justification", "")
            })

        return results

    async def _invoke_with_retry(self, msg: HumanMessage):
        """LLM invocation with retry functionality"""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.llm_json.ainvoke([msg])
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[PropertyRanker Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

    async def _rank_properties_async(
        self,
        properties: List[Dict[str, Any]],
        class_index: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute strength evaluation of all properties asynchronously"""

        # Group by property_iri
        groups = self._group_by_property_iri(properties)

        total = len(groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[PropertyRanker] Starting parallel processing: {total} property types (concurrency={self.max_concurrency})")

        # Create tasks by dividing each group into batches
        async def _wrapped_group(property_iri: str, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal counter
            batch_results = []
            for i in range(0, len(props), self.batch_size):
                batch = props[i:i + self.batch_size]
                result = await self._rank_batch(batch, class_index, property_iri)
                batch_results.extend(result)

            async with lock:
                counter += 1
                if self.progress:
                    print(f"[{counter}/{total}] PropertyRanker: {property_iri} processed ({len(props)} properties)")

            return batch_results

        # Execute in parallel
        all_results = await asyncio.gather(*[_wrapped_group(prop_iri, props) for prop_iri, props in groups.items()])

        # Convert results to dictionary with src_id+dest_id as key
        strength_map = {}
        for batch_results in all_results:
            for result in batch_results:
                key = f"{result['src_id']}:{result['dest_id']}"
                strength_map[key] = result

        return strength_map

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Accept PropertyChainOutput as input and output PropertyChainOutput with updated strength
        """
        property_candidates = input.get("property_candidates", {})
        class_info = input.get("class_info", {})

        properties = property_candidates.get("properties", [])
        classes = class_info.get("classes", [])

        if not properties:
            # If no properties, return as is
            output = {
                "id": property_candidates.get("id", str(uuid.uuid4())),
                "properties": []
            }
            self.logger.save_log(output, filename_prefix="property_ranker_output_")
            return output

        # Create class index
        class_index = self._index_classes_by_id(class_info)

        # Execute strength evaluation
        strength_map = asyncio.run(self._rank_properties_async(properties, class_index))

        # Set strength in properties
        ranked_properties = []
        ranked_properties_with_labels = []
        for prop in properties:
            key = f"{prop.get('src_id')}:{prop.get('dest_id')}"
            ranking_result = strength_map.get(key, {})

            strength = ranking_result.get("strength", 0)

            # Create property with updated strength and ranker_justification
            ranked_prop = {
                "id": prop.get("id", str(uuid.uuid4())),
                "src_id": prop.get("src_id", ""),
                "property_iri": prop.get("property_iri", ""),
                "dest_id": prop.get("dest_id", ""),
                "strength": strength,
                "ranker_justification": ranking_result.get("justification", "")
            }
            ranked_properties.append(ranked_prop)

            # Add label information and evaluation information for logging
            src_class = class_index.get(prop.get("src_id", ""), {})
            dest_class = class_index.get(prop.get("dest_id", ""), {})

            ranked_properties_with_labels.append({
                **ranked_prop,
                "src_label": src_class.get("label", ""),
                "dest_label": dest_class.get("label", ""),
                "justification": ranking_result.get("justification", "")
            })

        output = {
            "id": property_candidates.get("id", str(uuid.uuid4())),
            "properties": ranked_properties
        }

        if self.progress:
            # Display strength distribution
            strength_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for prop in ranked_properties:
                strength_counts[prop["strength"]] += 1
            print(f"[PropertyRanker] Completed: ranked {len(ranked_properties)} properties")
            print(f"[PropertyRanker] Strength distribution: 0={strength_counts[0]}, 1={strength_counts[1]}, 2={strength_counts[2]}, 3={strength_counts[3]}")

        # Log output (with label information and evaluation information)
        log_output = {
            "id": output["id"],
            "properties": ranked_properties_with_labels
        }
        self.logger.save_log(log_output, filename_prefix="property_ranker_output_")

        return output


if __name__ == "__main__":
    import glob

    # Create PropertyRanker instance
    ranker = PropertyRanker()

    # Load the latest PropertyExtractor output JSON
    property_files = sorted(glob.glob("/workspace/app/log/property_extractor/property_extractor_output_*.json"), reverse=True)

    if not property_files:
        print("Error: No PropertyExtractor output files found in log/property_extractor/")
        exit(1)

    latest_property_file = property_files[0]
    print(f"Loading latest PropertyExtractor output: {latest_property_file}")

    with open(latest_property_file, "r", encoding="utf-8") as f:
        property_candidates = json.load(f)

    # Load the latest ClassConsolidator output JSON
    class_files = sorted(glob.glob("/workspace/app/log/class_consolidator/class_consolidator_output_*.json"), reverse=True)

    if not class_files:
        print("Error: No ClassConsolidator output files found in log/class_consolidator/")
        exit(1)

    latest_class_file = class_files[0]
    print(f"Loading latest ClassConsolidator output: {latest_class_file}")

    with open(latest_class_file, "r", encoding="utf-8") as f:
        class_info = json.load(f)

    # Prepare input data for PropertyRanker
    sample_input = {
        "property_candidates": property_candidates,
        "class_info": class_info
    }

    print("=== PropertyRanker Test ===")
    print(f"Input: {len(property_candidates.get('properties', []))} properties to rank")
    print(f"Input: {len(class_info.get('classes', []))} classes loaded")
    print("\n" + "="*50 + "\n")

    try:
        # Execute PropertyRanker
        result = ranker.invoke(sample_input)

        print("Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print(f"\nSummary:")
        print(f"- Total properties ranked: {len(result['properties'])}")

        # Display strength distribution
        strength_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for prop in result["properties"]:
            strength_counts[prop["strength"]] += 1

        print(f"\nStrength Distribution:")
        print(f"  0 (None): {strength_counts[0]}")
        print(f"  1 (Weak): {strength_counts[1]}")
        print(f"  2 (Medium): {strength_counts[2]}")
        print(f"  3 (Strong): {strength_counts[3]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
