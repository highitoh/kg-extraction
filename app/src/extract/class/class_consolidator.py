from typing import Dict, Any, List
import asyncio
import json
import os

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_utils import fill_prompt_template
from src.utils.llm_retry import call_with_retry
from src.utils.id_generator import generate_class_id


class ClassConsolidator(Runnable):
    """
    Class to consolidate similar class labels
    Judges same concept with LLM and integrates by selecting canonical label
    """

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        log_dir: str = "/workspace/app/log/class_consolidator",
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = "minimal"
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # Load metamodel
        self.metamodel = self._load_metamodel()
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        # Load custom rules
        self.custom_rules = self._load_custom_rules()

    def _create_llm(self) -> Any:
        return LLMFactory.create_llm(
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

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
        llm = self._create_llm()
        all_consolidated_classes = asyncio.run(self._process_classes_parallel(class_groups, llm))

        # Build output
        output = input.copy()
        output["classes"] = all_consolidated_classes

        if self.progress:
            print(f"[ClassConsolidator] Completed: {len(classes)} -> {len(all_consolidated_classes)} classes")

        self.logger.save_log(output, filename_prefix="class_consolidator_output_")

        return output

    def invoke_stage2_match_with_frozen(self, new_classes: list, frozen_classes: list, config=None) -> Dict[str, Any]:
        """
        Stage 2: Match already-consolidated new classes against frozen existing classes.

        This is called after Stage 1 (consolidate new among themselves).
        Input new_classes are already consolidated, so typically small (5-10 classes).

        Parameters
        ----------
        new_classes: list
            Already-consolidated new class instances (output from Stage 1)
        frozen_classes: list
            Existing ACCEPT'ed class instances with _frozen_from_existing_accept flag

        Returns
        -------
        output: ClassConsolidatorOutput
            - classes: Merged list (frozen classes + unmatched new classes)
        """
        if not new_classes:
            # No new classes, return frozen as-is
            return {"classes": frozen_classes}

        if not frozen_classes:
            # No frozen classes, return new as-is
            return {"classes": new_classes}

        if self.progress:
            print(
                f"[ClassConsolidator Stage 2] Matching {len(new_classes)} new classes against {len(frozen_classes)} frozen classes"
            )

        # Group by class_iri (only process class_iris that appear in BOTH new and frozen)
        new_by_iri = self._group_classes_by_class_iri(new_classes)
        frozen_by_iri = self._group_classes_by_class_iri(frozen_classes)

        common_iris = set(new_by_iri.keys()) & set(frozen_by_iri.keys())

        # Process each common class_iri: match new against frozen
        all_classes = []

        # Process all common_iris in parallel
        llm = self._create_llm()
        matched_results = asyncio.run(self._process_stage2_parallel(common_iris, new_by_iri, frozen_by_iri, llm))

        for matched_result in matched_results:
            all_classes.extend(matched_result)

        # Add frozen classes for class_iris that have no new instances
        for class_iri, frozen_instances in frozen_by_iri.items():
            if class_iri not in common_iris:
                all_classes.extend(frozen_instances)

        # Add new classes for class_iris that have no frozen instances
        for class_iri, new_instances in new_by_iri.items():
            if class_iri not in common_iris:
                all_classes.extend(new_instances)

        output = {"classes": all_classes}

        if self.progress:
            print(
                f"[ClassConsolidator Stage 2] Completed: {len(new_classes)} new + {len(frozen_classes)} frozen → {len(all_classes)} final classes"
            )

        self.logger.save_log(output, filename_prefix="class_consolidator_stage2_output_")

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

    async def _process_classes_parallel(self, class_groups: Dict[str, list], llm: Any) -> list:
        """Process class groups in parallel"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(
                f"[ClassConsolidator] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})"
            )

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                consolidated = await self._consolidate_classes_async(llm, class_instances, class_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] ClassConsolidator: {class_name} processed ({len(class_instances)} -> {len(consolidated)} instances)"
                        )

                return consolidated

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # Flatten results
        all_consolidated = []
        for result in results:
            all_consolidated.extend(result)

        return all_consolidated

    async def _consolidate_classes_async(self, llm: Any, classes: list, class_iri: str) -> list:
        """
        Consolidate specified class list with LLM (async version)

        Args:
            llm: LLM instance to use for consolidation
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
        llm_json = llm.bind(
            response_format={"type": "json_schema", "json_schema": {"name": "class_consolidator", "schema": schema}}
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
            custom_rules=self.custom_rules,
        )

        # Execute consolidation with LLM (with retry)
        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
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

    async def _process_stage2_parallel(self, common_iris: set, new_by_iri: dict, frozen_by_iri: dict, llm: Any) -> list:
        """Process Stage 2 matching in parallel for each common class_iri"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(common_iris)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(
                f"[ClassConsolidator Stage 2] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})"
            )

        async def _wrapped(class_iri: str) -> list:
            nonlocal counter
            async with sem:
                new_instances = new_by_iri[class_iri]
                frozen_instances = frozen_by_iri[class_iri]

                class_name = self._get_class_name_from_iri(class_iri)

                matched_result = await self._match_new_against_frozen_async(
                    llm, new_instances, frozen_instances, class_iri
                )

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] ClassConsolidator Stage 2: {class_name} processed ({len(new_instances)} new vs {len(frozen_instances)} frozen → {len(matched_result)} final)"
                        )

                return matched_result

        results = await asyncio.gather(*[_wrapped(iri) for iri in common_iris])
        return results

    async def _match_new_against_frozen_async(
        self, llm: Any, new_classes: list, frozen_classes: list, class_iri: str
    ) -> list:
        """
        Stage 2 core logic: Match new classes against frozen classes.

        Task for LLM: For each new class, determine if it matches any frozen class.
        If match found, merge new into frozen. If not, keep new as standalone.

        Args:
            llm: LLM instance to use for matching
            new_classes: Consolidated new class instances (typically 1-5)
            frozen_classes: Existing frozen class instances (typically 5-50)
            class_iri: Target class IRI

        Returns:
            List of classes: frozen classes (with merged sources if matched) + unmatched new classes
        """
        # Build combined list: frozen first
        all_classes = frozen_classes + new_classes

        # Build schema
        schema = self._build_schema()
        llm_json = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "class_consolidator_stage2", "schema": schema},
            }
        )

        # Create label list with [EXISTING] markers
        labels = []
        for i, c in enumerate(all_classes):
            label = c.get("label", "")
            if i < len(frozen_classes):
                labels.append(f"{label} [EXISTING]")
            else:
                labels.append(f"{label} [NEW]")
        label_list_text = "\n".join([f"{i}: {label}" for i, label in enumerate(labels)])

        # Get class name and consolidation rules
        class_name = self._get_class_name_from_iri(class_iri)
        extraction_rules = self._build_extraction_rules(class_iri)

        # Build Stage 2 specific instructions
        stage2_rules = """

[STAGE 2: Matching New Classes Against Existing]
- Labels marked [EXISTING] are from previous extractions (frozen)
- Labels marked [NEW] are newly consolidated classes
- Task: Group each [NEW] label with a matching [EXISTING] label if they represent the same concept
- If a [NEW] label matches an [EXISTING] label, prefer the [EXISTING] label as canonical_label
- NEVER group two [EXISTING] labels together (already consolidated)
- If a [NEW] label doesn't match any [EXISTING], keep it as a standalone group
- Remove [EXISTING] and [NEW] markers from canonical_label in output
"""

        enhanced_custom_rules = self.custom_rules + stage2_rules

        # Build prompt
        prompt = fill_prompt_template(
            self.prompt_template,
            class_name=class_name,
            extraction_rules=extraction_rules,
            label_list=label_list_text,
            custom_rules=enhanced_custom_rules,
        )

        # Execute matching with LLM
        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            label="ClassConsolidator Stage 2",
        )

        # Parse response
        result_text = self._to_text(response.content)
        try:
            parsed = json.loads(result_text)
            groups = parsed.get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print("[ClassConsolidator Stage 2] JSON parse error, keeping classes separate")
            return all_classes

        # Process groups: merge new into frozen where matched
        result_classes = []
        processed_indices = set()

        for group in groups:
            indices = group.get("indices", [])
            canonical_label = group.get("canonical_label", "").replace(" [EXISTING]", "").replace(" [NEW]", "")
            reason = group.get("reason", "")

            if not indices or not canonical_label:
                continue

            # Check if group contains frozen classes
            frozen_indices = [i for i in indices if i < len(frozen_classes)]
            new_indices = [i for i in indices if i >= len(frozen_classes)]

            if len(frozen_indices) > 1:
                # Multiple frozen in one group - LLM error, keep separate
                for idx in frozen_indices:
                    result_classes.append(all_classes[idx])
                    processed_indices.add(idx)
                for idx in new_indices:
                    result_classes.append(all_classes[idx])
                    processed_indices.add(idx)
                continue

            # Normal case: merge group
            group_classes = [all_classes[i] for i in indices if i < len(all_classes)]

            # If frozen class in group, put it first (base for merge)
            if frozen_indices:
                frozen_idx = frozen_indices[0]
                group_classes_reordered = [all_classes[frozen_idx]] + [
                    all_classes[i] for i in indices if i != frozen_idx
                ]
                merged = self._merge_classes(group_classes_reordered, canonical_label, reason)
            else:
                merged = self._merge_classes(group_classes, canonical_label, reason)

            result_classes.append(merged)
            processed_indices.update(indices)

        # Add unprocessed classes
        for i, cls in enumerate(all_classes):
            if i not in processed_indices:
                result_classes.append(cls)

        return result_classes

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

    def _merge_classes(
        self, group_classes: List[Dict[str, Any]], canonical_label: str, reason: str = ""
    ) -> Dict[str, Any]:
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

        # Check if base class is frozen (from existing knowledge)
        is_frozen_base = merged.get("_frozen_from_existing_accept", False)

        # Update label to canonical label
        merged["label"] = canonical_label

        # Recompute ID based on the new canonical label and first source
        # UNLESS this is a frozen base class (preserve existing IDs for stability)
        if not is_frozen_base:
            first_source = group_classes[0].get("sources", [""])[0] if group_classes[0].get("sources") else ""
            merged["id"] = generate_class_id(merged["class_iri"], canonical_label, first_source)
        # else: preserve existing ID for frozen classes

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
                print(
                    f"Warning: sources length ({len(sources)}) != file_ids length ({len(file_ids)}) in class {cls.get('id', 'unknown')}"
                )
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
