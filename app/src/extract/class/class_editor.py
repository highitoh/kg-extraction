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


class ClassEditor(Runnable):
    """
    Class to correct or exclude class labels judged as REVIEW

    Processing flow:
    1. Separate input into ACCEPT/REVIEW/REJECT
    2. Re-evaluate only REVIEW classes with LLM (execute in parallel for each class_iri)
    3. LLM output:
       - If revised_label is provided: update label and keep judgment="REVIEW"
       - If should_reject=true: change judgment="REJECT"
    4. Consolidate ACCEPT/REJECT/corrected REVIEW for output
    """

    def __init__(self,
                 llm: Any = None,
                 model: str = "gpt-5-mini",
                 temperature: float = 0.0,
                 progress: bool = True,
                 max_concurrency: int = 8,
                 log_dir: str = "/workspace/app/log/class_editor"):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # Load metamodel
        self.metamodel = self._load_metamodel()
        # Load prompt template
        self.prompt_template = self._load_prompt_template()

    def _build_schema(self, n: int) -> dict:
        """Load LLM schema file and set element count

        Args:
            n: Number of classes to judge
        """
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-editor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Set array element count for corrections
        schema["properties"]["corrections"]["minItems"] = n
        schema["properties"]["corrections"]["maxItems"] = n

        return schema

    def _load_metamodel(self) -> dict:
        """Load metamodel file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        metamodel_path = os.path.join(app_dir, "metamodel",
                                      "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_extract_rules_section(self, target_class_iri: str) -> str:
        """Dynamically generate extract rules section for specified class

        Args:
            target_class_iri: Target class IRI
        """
        rules_lines = []

        for cls in self.metamodel.get("classes", []):
            if cls.get("iri", "") == target_class_iri:
                if "extract_rules" in cls and cls.get("extract_rules"):
                    for rule in cls["extract_rules"]:
                        rules_lines.append(f"- {rule}")
                break

        result = "\n".join(rules_lines)
        return result if result else "(No extraction rules)"

    def _load_prompt_template(self) -> str:
        """Load prompt template file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_editor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_duplicate_fix_schema(self) -> dict:
        """Load LLM schema file for duplicate fixing

        Returns:
            dict: JSON schema for duplicate fixing
        """
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-editor", "llm-duplicate.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_duplicate_fix_prompt_template(self) -> str:
        """Load prompt template file for duplicate fixing"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_editor_duplicate.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _merge_duplicate_classes(self, group_classes: list, canonical_label: str, merged_class_id: str) -> dict:
        """Merge duplicate classes into one

        Args:
            group_classes: List of class instances to merge
            canonical_label: Canonical label
            merged_class_id: ID of the class to be the representative

        Returns:
            dict: Merged class instance (including consolidated_ids and consolidated_labels)
        """
        # Find the class to be the representative
        base_class = None
        for cls in group_classes:
            if cls.get("id") == merged_class_id:
                base_class = cls.copy()
                break

        if not base_class:
            # Fallback: use the first class
            base_class = group_classes[0].copy()

        # Update label to canonical label
        base_class["label"] = canonical_label

        # Collect IDs and labels of merged classes (excluding canonical)
        consolidated_ids = []
        consolidated_labels = []

        for cls in group_classes:
            cls_id = cls.get("id")
            if cls_id != merged_class_id:
                consolidated_ids.append(cls_id)
                consolidated_labels.append(cls.get("label", ""))

        base_class["consolidated_ids"] = consolidated_ids
        base_class["consolidated_labels"] = consolidated_labels

        # Merge sources and file_ids (ClassConsolidator pattern)
        all_pairs = []
        for cls in group_classes:
            sources = cls.get("sources", [])
            file_ids = cls.get("file_ids", [])

            # Align lengths of sources and file_ids
            min_length = min(len(sources), len(file_ids))
            sources = sources[:min_length]
            file_ids = file_ids[:min_length]

            # Collect as (source, file_id) pairs
            for source, file_id in zip(sources, file_ids):
                all_pairs.append((source, file_id))

        # Remove duplicates while preserving order
        unique_pairs = list(dict.fromkeys(all_pairs))

        # Split pairs back to sources and file_ids
        if unique_pairs:
            base_class["sources"], base_class["file_ids"] = zip(*unique_pairs)
            base_class["sources"] = list(base_class["sources"])
            base_class["file_ids"] = list(base_class["file_ids"])
        else:
            base_class["sources"] = []
            base_class["file_ids"] = []

        return base_class

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Main entry point for class label correction processing

        Parameters
        ----------
        input: ClassLabelCorrectorInput (= ClassReviewerOutput)
          - id: Extraction ID
          - classes: List of class instances (including judgment/justification)

        Returns
        -------
        output: ClassLabelCorrectorOutput (= ClassReviewerOutput)
        """
        classes = input.get("classes", [])

        # Separate by judgment
        accept_classes = [c for c in classes if c.get("judgment") == "ACCEPT"]
        review_classes = [c for c in classes if c.get("judgment") == "REVIEW"]
        reject_classes = [c for c in classes if c.get("judgment") == "REJECT"]

        if self.progress:
            print(f"[ClassEditor] Total {len(classes)} classes "
                  f"[ACCEPT: {len(accept_classes)}, REVIEW: {len(review_classes)}, REJECT: {len(reject_classes)}]")

        if not review_classes:
            # Return as is if no REVIEW classes
            if self.progress:
                print("[ClassEditor] No REVIEW classes to process")
            return input

        # Group by class_iri
        review_groups = self._group_classes_by_class_iri(review_classes)

        if self.progress:
            print(f"[ClassEditor] {len(review_groups)} class types found for review")

        # Execute corrections in parallel for each REVIEW class
        corrected_classes = asyncio.run(self._process_classes_parallel(review_groups))

        # Build output (ACCEPT + corrected REVIEW + REJECT)
        all_classes = accept_classes + corrected_classes + reject_classes

        output = input.copy()
        output["classes"] = all_classes

        if self.progress:
            # Aggregate correction result breakdown
            corrected_review_count = sum(1 for c in corrected_classes if c.get("judgment") == "REVIEW")
            corrected_reject_count = sum(1 for c in corrected_classes if c.get("judgment") == "REJECT")
            print(f"[ClassEditor] Completed: Processed {len(review_classes)} REVIEW classes "
                  f"[REVIEW: {corrected_review_count}, REJECT: {corrected_reject_count}]")

        self.logger.save_log(output, filename_prefix="class_editor_output_")

        return output

    async def _fix_duplicates_async(self, classes: list, class_iri: str) -> dict:
        """Merge judgment and correction of duplicate classes (async version)

        Args:
            classes: List of class instances to correct
            class_iri: Target class IRI

        Returns:
            dict: Dictionary of duplicate correction results
                {
                    "dup_group_0": {
                        "should_merge": true,
                        "canonical_label": "User",
                        "merged_class_id": "cls_001",
                        "reason": "...",
                        "merged_class": {...}
                    },
                    ...
                }
        """
        # Extract only classes with duplicate_group_id
        duplicates = [c for c in classes if "duplicate_group_id" in c]

        if not duplicates:
            return {}

        # Group by duplicate_group_id
        groups = {}
        for cls in duplicates:
            group_id = cls.get("duplicate_group_id")
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(cls)

        if not groups:
            return {}

        # Build input data for LLM
        duplicate_groups_input = []
        for group_id, group_classes in groups.items():
            candidates = []
            for cls in group_classes:
                candidates.append({
                    "id": cls.get("id", ""),
                    "label": cls.get("label", ""),
                    "sources": cls.get("sources", [])
                })

            duplicate_groups_input.append({
                "duplicate_group_id": group_id,
                "candidates": candidates
            })

        duplicate_groups_json = json.dumps(duplicate_groups_input, ensure_ascii=False, indent=2)

        # Build schema
        schema = self._build_duplicate_fix_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_editor_duplicate",
                    "schema": schema
                }
            }
        )

        # Build prompt
        prompt_template = self._load_duplicate_fix_prompt_template()
        prompt = fill_prompt_template(
            prompt_template,
            duplicate_groups_json=duplicate_groups_json
        )

        # Execute duplicate fixing with LLM (with retry)
        try:
            response = await call_with_retry(
                lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
                timeout=90.0,
                label="ClassEditor-Duplicate",
            )

            # Extract text from response
            result_text = self._to_text(response.content)

            # Parse JSON
            parsed = json.loads(result_text)
            resolutions = parsed.get("resolutions", [])

            # Convert results to dictionary
            result = {}
            for resolution in resolutions:
                group_id = resolution.get("duplicate_group_id")
                should_merge = resolution.get("should_merge", False)
                reason = resolution.get("reason", "")

                result[group_id] = {
                    "should_merge": should_merge,
                    "reason": reason
                }

                if should_merge:
                    canonical_label = resolution.get("canonical_label", "")
                    merged_class_id = resolution.get("merged_class_id", "")

                    # canonical_label and merged_class_id are required
                    if not canonical_label or not merged_class_id:
                        # Fallback: do not merge
                        result[group_id]["should_merge"] = False
                        result[group_id]["reason"] = f"Missing required fields: {reason}"
                    else:
                        result[group_id]["canonical_label"] = canonical_label
                        result[group_id]["merged_class_id"] = merged_class_id

                        # Generate merged class
                        group_classes = groups.get(group_id, [])
                        if group_classes:
                            merged_class = self._merge_duplicate_classes(
                                group_classes,
                                canonical_label,
                                merged_class_id
                            )
                            result[group_id]["merged_class"] = merged_class

            return result

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassEditor] Duplicate fix JSON parse error: {e}")
            # Return empty dictionary on error (no merge)
            return {}
        except Exception as e:
            if self.progress:
                print(f"[ClassEditor] Duplicate fix error: {e}")
            return {}

    def _merge_fix_results(self, classes: list, prohibit_fixed: list, duplicate_resolved: dict) -> list:
        """Merge results of prohibit rule correction and duplicate correction

        Args:
            classes: Original class instance list
            prohibit_fixed: Class list after prohibit rule correction
            duplicate_resolved: Dictionary of duplicate correction results

        Returns:
            list: Merged class list
        """
        results = []
        processed_ids = set()

        # Map prohibit rule correction results
        prohibit_map = {c.get("id"): c for c in prohibit_fixed}

        for cls in classes:
            class_id = cls.get("id", "")

            if class_id in processed_ids:
                continue

            # Step 1: Apply prohibit rule correction
            corrected_cls = prohibit_map.get(class_id, cls)

            # Skip if REJECTED
            if corrected_cls.get("judgment") == "REJECT":
                results.append(corrected_cls)
                processed_ids.add(class_id)
                continue

            # Step 2: Check duplicate correction
            duplicate_group_id = cls.get("duplicate_group_id")

            if duplicate_group_id and duplicate_group_id in duplicate_resolved:
                resolution = duplicate_resolved[duplicate_group_id]
                should_merge = resolution.get("should_merge", False)

                if should_merge:
                    merged_class_id = resolution.get("merged_class_id")

                    if class_id == merged_class_id:
                        # Representative class: use merged class
                        merged_class = resolution.get("merged_class")
                        if merged_class:
                            # Preserve judgment and justification from prohibit rule correction
                            merged_class["judgment"] = corrected_cls.get("judgment", "REVIEW")
                            merged_class["justification"] = corrected_cls.get("justification", "")
                            results.append(merged_class)
                            processed_ids.add(class_id)
                        else:
                            # Use corrected class on merge failure
                            results.append(corrected_cls)
                            processed_ids.add(class_id)
                    else:
                        # Non-representative: skip as merged
                        processed_ids.add(class_id)
                else:
                    # should_merge=false: keep as is
                    results.append(corrected_cls)
                    processed_ids.add(class_id)
            else:
                # Not included in duplicate group: keep as is
                results.append(corrected_cls)
                processed_ids.add(class_id)

        # Clean up temporary duplicate_group_id field from all results
        for cls in results:
            if "duplicate_group_id" in cls:
                del cls["duplicate_group_id"]

        return results

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> list:
        """Process class groups in parallel (execute prohibit rule correction and duplicate correction concurrently)"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassEditor] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})")

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                # Execute prohibit rule correction and duplicate correction in parallel
                prohibit_fixed, duplicate_resolved = await asyncio.gather(
                    self._fix_prohibit_violations_async(class_instances, class_iri),
                    self._fix_duplicates_async(class_instances, class_iri)
                )

                # Merge results
                merged = self._merge_fix_results(class_instances, prohibit_fixed, duplicate_resolved)

                async with lock:
                    counter += 1
                    if self.progress:
                        # Aggregate correction result breakdown
                        review_count = sum(1 for c in merged if c.get("judgment") == "REVIEW")
                        reject_count = sum(1 for c in merged if c.get("judgment") == "REJECT")

                        # Count merges
                        merge_count = sum(
                            1 for res in duplicate_resolved.values()
                            if res.get("should_merge", False)
                        )
                        total_dup_groups = len(duplicate_resolved)

                        print(f"[{counter}/{total}] ClassEditor: {class_name} processed "
                              f"({len(class_instances)} -> {len(merged)} instances) "
                              f"[REVIEW: {review_count}, REJECT: {reject_count}] "
                              f"[Duplicates merged: {merge_count}/{total_dup_groups}]")

                return merged

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # Flatten results
        all_corrected = []
        for result in results:
            all_corrected.extend(result)

        return all_corrected

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

    async def _fix_prohibit_violations_async(self, classes: list, class_iri: str) -> list:
        """Fix prohibit rule violations in specified class list with LLM (async version)

        Args:
            classes: List of class instances to correct (all are REVIEW judgments)
            class_iri: Target class IRI (use only this class's extract_rules)
        """
        if not classes:
            return []

        schema = self._build_schema(len(classes))
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_editor",
                    "schema": schema
                }
            })

        # Build input data for LLM
        input_data = []
        for c in classes:
            input_data.append({
                "label": c.get("label", ""),
                "source": c.get("sources", [""])[0] if c.get("sources") else "",
                "justification": c.get("justification", "")
            })

        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)

        # Get only the target class's extract_rules
        extract_rules = self._build_extract_rules_section(class_iri)

        # Build prompt (embed extract rules specific to target class and input data)
        prompt = fill_prompt_template(
            self.prompt_template,
            extract_rules=extract_rules,
            input_data=input_json
        )

        # Execute label correction with LLM (with retry)
        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            label="ClassEditor",
        )

        # Extract text from response
        result_text = self._to_text(response.content)

        # Parse JSON
        try:
            parsed = json.loads(result_text)
            corrections = parsed["corrections"]

            # Apply correction results to each class
            corrected_classes = []
            for c, correction_obj in zip(classes, corrections):
                c_corrected = c.copy()

                should_reject = correction_obj.get("should_reject", False)
                revised_label = correction_obj.get("revised_label", "")
                revision_reason = correction_obj.get("revision_reason", "")

                if should_reject:
                    # Change to REJECT if excluding
                    c_corrected["judgment"] = "REJECT"
                    c_corrected["justification"] = revision_reason
                else:
                    # Keep as REVIEW if label corrected (will be re-evaluated by ClassReviewer)
                    if revised_label and revised_label != c.get("label", ""):
                        c_corrected["label"] = revised_label
                    # judgment = "REVIEW" unchanged
                    c_corrected["justification"] = revision_reason

                corrected_classes.append(c_corrected)

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassLabelCorrector] JSON parse error: {e}, response: {result_text}")
            # Treat all as REJECT on error
            corrected_classes = []
            for c in classes:
                c_rejected = c.copy()
                c_rejected["judgment"] = "REJECT"
                c_rejected["justification"] = f"JSON parse error: {e}"
                corrected_classes.append(c_rejected)

        return corrected_classes


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

    # Create and run the corrector
    corrector = ClassEditor(progress=True, log_dir="/workspace/app/log/class_editor")
    result = corrector.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
