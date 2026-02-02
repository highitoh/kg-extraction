from typing import Dict, Any, List
import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_utils import fill_prompt_template
from src.utils.llm_retry import call_with_retry


class ClassConsolidator(Runnable):
    """
    Class to consolidate similar class labels
    Judges same concept with LLM and integrates by selecting canonical label
    """

    def __init__(
        self,
        llm: Any = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        log_dir: str = "/workspace/app/log/class_consolidator"
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
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """Load prompt template file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts", "class_consolidator.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_custom_rules(self) -> str:
        """Load custom rules file if it exists"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        custom_path = os.path.join(app_dir, "prompts", "custom", "class_consolidator_custom.txt")

        # Return empty string if file doesn't exist
        if not os.path.exists(custom_path):
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return "\n" + content if content else ""

    def _build_schema(self) -> dict:
        """Load JSON schema for LLM"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas", "class-consolidator", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Main entry point for class consolidation processing

        Parameters
        ----------
        input: ClassConsolidatorInput (= ClassChainOutput)
          - id: Extraction ID
          - classes: List of class instances

        Returns
        -------
        output: ClassConsolidatorOutput (= ClassChainOutput)
        """
        classes = input.get("classes", [])

        if self.progress:
            print(f"[ClassConsolidator] Processing {len(classes)} classes")

        # Group by class_iri
        class_groups = self._group_classes_by_class_iri(classes)

        if self.progress:
            print(f"[ClassConsolidator] {len(class_groups)} class types found")

        # Execute consolidation in parallel for each class
        all_consolidated_classes = asyncio.run(self._process_classes_parallel(class_groups))

        # Build output
        output = input.copy()
        output["classes"] = all_consolidated_classes

        if self.progress:
            print(f"[ClassConsolidator] Completed: {len(classes)} -> {len(all_consolidated_classes)} classes")

        self.logger.save_log(output, filename_prefix="class_consolidator_output_")

        return output

    def _group_classes_by_class_iri(self, classes: list) -> Dict[str, list]:
        """Group by class_iri"""
        class_groups = {}

        for c in classes:
            class_iri = c.get("class_iri", "")
            if class_iri not in class_groups:
                class_groups[class_iri] = []
            class_groups[class_iri].append(c)

        return class_groups

    def _get_class_name_from_iri(self, class_iri: str) -> str:
        """Get class name from class_iri"""
        for cls in self.metamodel.get("classes", []):
            if cls.get("iri", "") == class_iri:
                return cls.get("name", "unknown")
        return class_iri.split("#")[-1] if "#" in class_iri else class_iri

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> list:
        """Process class groups in parallel"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassConsolidator] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})")

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                consolidated = await self._consolidate_classes_async(class_instances, class_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        print(f"[{counter}/{total}] ClassConsolidator: {class_name} processed ({len(class_instances)} -> {len(consolidated)} instances)")

                return consolidated

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # Flatten results
        all_consolidated = []
        for result in results:
            all_consolidated.extend(result)

        return all_consolidated

    async def _consolidate_classes_async(self, classes: list, class_iri: str) -> list:
        """
        Consolidate specified class list with LLM (async version)

        Args:
            classes: List of class instances to consolidate
            class_iri: Target class IRI

        Returns:
            Consolidated class instance list
        """
        if not classes:
            return []

        # No consolidation needed if only one instance
        if len(classes) == 1:
            return classes

        # Build schema
        schema = self._build_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_consolidator",
                    "schema": schema
                }
            }
        )

        # Create label list
        labels = [c.get("label", "") for c in classes]
        label_list_text = "\n".join([f"{i}: {label}" for i, label in enumerate(labels)])

        # Get class name and consolidation rules
        class_name = self._get_class_name_from_iri(class_iri)
        extraction_rules = self._build_extraction_rules(class_iri)

        # Build prompt
        prompt = fill_prompt_template(
            self.prompt_template,
            class_name=class_name,
            extraction_rules=extraction_rules,
            label_list=label_list_text,
            custom_rules=self.custom_rules
        )

        # Execute consolidation with LLM (with retry)
        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            progress=self.progress,
            label="ClassConsolidator",
        )

        # Extract text from response
        result_text = self._to_text(response.content)

        # Parse JSON
        try:
            parsed = json.loads(result_text)
            groups = parsed.get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print(f"[ClassConsolidator] JSON parse error: {result_text}")
            return classes

        # Generate consolidated class instances from group information
        consolidated_classes = []
        for group in groups:
            indices = group.get("indices", [])
            canonical_label = group.get("canonical_label", "")
            reason = group.get("reason", "")

            if not indices or not canonical_label:
                continue

            # Get class instances within group
            group_classes = [classes[i] for i in indices if i < len(classes)]
            if not group_classes:
                continue

            # Consolidation processing
            consolidated_class = self._merge_classes(group_classes, canonical_label, reason)
            consolidated_classes.append(consolidated_class)

        return consolidated_classes

    def _build_extraction_rules(self, class_iri: str) -> str:
        """
        Get extract_rules for corresponding class from metamodel and
        generate text of rules related to normalization

        Parameters
        ----------
        class_iri : str
            Target class IRI

        Returns
        -------
        str
            Consolidation rules string
        """
        extraction_rules = []
        for cls in self.metamodel.get("classes", []):
            if cls.get("iri") == class_iri:
                extraction_rules = cls.get("extract_rules", [])
                break

        return "\n".join(extraction_rules) if extraction_rules else []

    def _merge_classes(self, group_classes: List[Dict[str, Any]], canonical_label: str, reason: str = "") -> Dict[str, Any]:
        """
        Consolidate class instances within group into one

        Parameters
        ----------
        group_classes : List[Dict[str, Any]]
            List of class instances to consolidate
        canonical_label : str
            Canonical label
        reason : str
            Selection reason (for logging)

        Returns
        -------
        Dict[str, Any]
            Consolidated class instance
        """
        # Use first instance as base
        merged = group_classes[0].copy()

        # Update label to canonical label
        merged["label"] = canonical_label

        # Collect IDs and labels of consolidated classes (excluding the first one)
        consolidated_ids = []
        consolidated_labels = []

        for i, cls in enumerate(group_classes):
            if i == 0:
                # Skip the first class (the canonical one)
                continue
            cls_id = cls.get("id")
            if cls_id:
                consolidated_ids.append(cls_id)
                consolidated_labels.append(cls.get("label", ""))

        merged["consolidated_ids"] = consolidated_ids
        merged["consolidated_labels"] = consolidated_labels

        # Consolidate sources and file_ids
        all_source_file_pairs = []

        for cls in group_classes:
            # Get sources field (array or string)
            sources = cls.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]

            # Get file_ids field (array or string)
            file_ids = cls.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]

            # Verify that sources and file_ids have matching lengths
            # If lengths differ, align to shorter one (for backward compatibility)
            min_length = min(len(sources), len(file_ids))
            if len(sources) != len(file_ids):
                print(f"Warning: sources length ({len(sources)}) != file_ids length ({len(file_ids)}) in class {cls.get('id', 'unknown')}")
                sources = sources[:min_length]
                file_ids = file_ids[:min_length]

            # Collect as (source, file_id) pairs
            for source, file_id in zip(sources, file_ids):
                all_source_file_pairs.append((source, file_id))

        # Remove duplicates while preserving order
        unique_pairs = list(dict.fromkeys(all_source_file_pairs))

        # Split pairs back to sources and file_ids
        if unique_pairs:
            merged["sources"], merged["file_ids"] = zip(*unique_pairs)
            merged["sources"] = list(merged["sources"])
            merged["file_ids"] = list(merged["file_ids"])
        else:
            merged["sources"] = []
            merged["file_ids"] = []

        return merged


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

    # Get the latest class_reviewer log file from log directory
    log_dir = "/workspace/app/log/class_reviewer"
    log_files = glob.glob(os.path.join(log_dir, "class_reviewer_output_*.json"))

    if not log_files:
        print("Error: No class_reviewer log files found")
        exit(1)

    # Get the latest log file
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('classes', []))} classes")

    # Create and run the consolidator
    consolidator = ClassConsolidator(progress=True, log_dir="/workspace/app/log/class_consolidator")
    result = consolidator.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
