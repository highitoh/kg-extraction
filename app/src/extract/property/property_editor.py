import asyncio
import json
import os
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from src.utils.llm_factory import LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger
from src.utils.prompt_utils import fill_prompt_template


class PropertyEditor(Runnable):
    """
    Class that corrects strength or excludes properties judged as REVIEW

    Processing flow:
    1. Separate input into ACCEPT/REVIEW/REJECT
    2. Re-evaluate only REVIEW properties with LLM (parallel execution per property_iri)
    3. LLM output:
       - If revised_strength is 1-2: Update strength and keep judgment="REVIEW"
       - If revised_strength=0: Change judgment to "REJECT"
    4. Integrate ACCEPT/REJECT/corrected REVIEW and output
    """

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        log_dir: str = "/workspace/app/log/property_editor",
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = "low"
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # Load metamodel
        self.metamodel = self._load_metamodel()
        # Load prompt template
        self.prompt_template = self._load_prompt_template()

    def _create_llm(self) -> Any:
        return LLMFactory.create_llm(
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    def _build_schema(self, n: int) -> dict:
        """Load LLM schema file and set number of elements

        Args:
            n: Number of properties to be judged
        """
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        schema_path = os.path.join(app_dir, "schemas", "property-editor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Set number of correction array elements
        schema["properties"]["corrections"]["minItems"] = n
        schema["properties"]["corrections"]["maxItems"] = n

        return schema

    def _load_metamodel(self) -> dict:
        """Load metamodel file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_prohibit_rules(self, target_property_iri: str) -> str:
        """Get prohibit_rules for specified property from metamodel

        Args:
            target_property_iri: Target property IRI

        Returns:
            Formatted prohibit_rules string, or empty string if not found
        """
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri", "") == target_property_iri:
                rules = prop.get("prohibit_rules", [])
                if rules:
                    return "\n".join(f"- {rule}" for rule in rules)
                break
        return ""

    def _load_prompt_template(self) -> str:
        """Load prompt template file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        prompt_path = os.path.join(app_dir, "prompts", "property_editor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Main entry point for property strength correction processing

        Parameters
        ----------
        input: PropertyEditorInput (= PropertyReviewerOutput)
          - properties: Property list (includes judgment/justification)

        Returns
        -------
        output: PropertyEditorOutput (= PropertyReviewerOutput)
        """
        properties = input.get("properties", [])

        # Separate by judgment
        accept_properties = [p for p in properties if p.get("judgment") == "ACCEPT"]
        review_properties = [p for p in properties if p.get("judgment") == "REVIEW"]
        reject_properties = [p for p in properties if p.get("judgment") == "REJECT"]

        if self.progress:
            print(
                f"[PropertyEditor] Total {len(properties)} properties [ACCEPT: {len(accept_properties)}, "
                f"REVIEW: {len(review_properties)}, REJECT: {len(reject_properties)}]"
            )

        if not review_properties:
            # If no REVIEW properties, return as is
            if self.progress:
                print("[PropertyEditor] No REVIEW properties to process")
            return input

        # Group by property_iri
        review_groups = self._group_properties_by_property_iri(review_properties)

        if self.progress:
            print(f"[PropertyEditor] {len(review_groups)} property types found for review")

        # Execute correction in parallel for each REVIEW property
        llm = self._create_llm()
        corrected_properties = asyncio.run(self._process_properties_parallel(review_groups, llm))

        # Build output (ACCEPT + corrected REVIEW + REJECT)
        all_properties = accept_properties + corrected_properties + reject_properties

        output = input.copy()
        output["properties"] = all_properties

        if self.progress:
            # Aggregate correction results breakdown
            corrected_review_count = sum(1 for p in corrected_properties if p.get("judgment") == "REVIEW")
            corrected_reject_count = sum(1 for p in corrected_properties if p.get("judgment") == "REJECT")
            print(
                f"[PropertyEditor] Completed: Processed {len(review_properties)} REVIEW properties "
                f"[REVIEW: {corrected_review_count}, REJECT: {corrected_reject_count}]"
            )

        self.logger.save_log(output, filename_prefix="property_editor_output_")

        return output

    async def _process_properties_parallel(self, property_groups: Dict[str, list], llm: Any) -> list:
        """Process property groups in parallel"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(property_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(
                f"[PropertyEditor] Starting parallel processing: {total} property types "
                f"(concurrency={self.max_concurrency})"
            )

        async def _wrapped(property_iri: str, property_instances: list) -> list:
            nonlocal counter
            async with sem:
                property_name = self._get_property_name_from_iri(property_iri)

                corrected = await self._correct_properties_async(llm, property_instances, property_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        # Aggregate correction results breakdown
                        review_count = sum(1 for p in corrected if p.get("judgment") == "REVIEW")
                        reject_count = sum(1 for p in corrected if p.get("judgment") == "REJECT")
                        print(
                            f"[{counter}/{total}] PropertyEditor: {property_name} processed "
                            f"({len(property_instances)} instances) "
                            f"[REVIEW: {review_count}, REJECT: {reject_count}]"
                        )

                return corrected

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in property_groups.items()])

        # Flatten results
        all_corrected = []
        for result in results:
            all_corrected.extend(result)

        return all_corrected

    def _group_properties_by_property_iri(self, properties: list) -> Dict[str, list]:
        """Group by property_iri"""
        property_groups = {}

        for p in properties:
            property_iri = p.get("property_iri", "")
            if property_iri not in property_groups:
                property_groups[property_iri] = []
            property_groups[property_iri].append(p)

        return property_groups

    def _get_property_name_from_iri(self, property_iri: str) -> str:
        """Get property name from property_iri"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri", "") == property_iri:
                return prop.get("name", "unknown")
        return property_iri.split("#")[-1] if "#" in property_iri else property_iri

    async def _correct_properties_async(self, llm: Any, properties: list, property_iri: str) -> list:
        """Correct strength of specified property list with LLM (async version)

        Args:
            properties: Property instance list to be corrected (all judged as REVIEW)
            property_iri: Target property IRI (only property_rules of this property are used)
        """
        if not properties:
            return []

        schema = self._build_schema(len(properties))
        llm_json = llm.bind(
            response_format={"type": "json_schema", "json_schema": {"name": "property_editor", "schema": schema}}
        )

        # Build input data for LLM
        input_data = []
        for p in properties:
            # Use sources field if available, otherwise empty string
            source_text = ""
            if "sources" in p and p["sources"]:
                source_text = p["sources"][0]

            input_data.append(
                {
                    "src_label": p.get("src_label", ""),
                    "property_iri": p.get("property_iri", ""),
                    "dest_label": p.get("dest_label", ""),
                    "strength": p.get("strength", 0),
                    "source": source_text,
                    "justification": p.get("justification", ""),
                }
            )

        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)

        # Get rules for the target property
        prohibit_rules = self._get_prohibit_rules(property_iri)

        # Build prompt with rule sections and input data
        prompt = fill_prompt_template(self.prompt_template, prohibit_rules=prohibit_rules, input_data=input_json)

        # Execute strength correction with LLM (with retry)
        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            label="PropertyEditor",
        )

        # Extract text from response
        result_text = self._to_text(response.content)

        # Parse JSON
        try:
            parsed = json.loads(result_text)
            corrections = parsed["corrections"]

            # Apply correction results to each property
            corrected_properties = []
            for p, correction_obj in zip(properties, corrections, strict=True):
                p_corrected = p.copy()

                revised_strength = correction_obj.get("revised_strength", 0)
                revision_reason = correction_obj.get("revision_reason", "")

                if revised_strength == 0:
                    # If strength is 0, change to REJECT
                    p_corrected["judgment"] = "REJECT"
                    p_corrected["strength"] = 0
                    p_corrected["justification"] = revision_reason
                else:
                    # If strength is corrected, keep as REVIEW (will be re-evaluated by PropertyReviewer)
                    p_corrected["strength"] = revised_strength
                    # Keep judgment = "REVIEW"
                    p_corrected["justification"] = revision_reason

                corrected_properties.append(p_corrected)

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[PropertyEditor] JSON parse error: {e}, response: {result_text}")
            # On error, treat all as REJECT
            corrected_properties = []
            for p in properties:
                p_rejected = p.copy()
                p_rejected["judgment"] = "REJECT"
                p_rejected["strength"] = 0
                p_rejected["justification"] = f"JSON parse error: {e}"
                corrected_properties.append(p_rejected)

        return corrected_properties

    @staticmethod
    def _to_text(content: Any) -> str:
        """Extract text from LLM response"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return ""


if __name__ == "__main__":
    import glob

    # Get the latest property_reviewer log file from log directory
    log_dir = "/workspace/app/log/property_reviewer"
    log_files = glob.glob(os.path.join(log_dir, "property_reviewer_output_*.json"))

    if not log_files:
        print("Error: No property_reviewer log files found")
        exit(1)

    # Get the latest log file
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('properties', []))} properties")

    # Create and run the editor
    editor = PropertyEditor(progress=True, log_dir="/workspace/app/log/property_editor")
    result = editor.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('properties', []))} properties")
