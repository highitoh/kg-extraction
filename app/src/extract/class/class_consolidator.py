import asyncio
import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from src.utils.id_generator import generate_class_id
from src.utils.label_normalizer import normalize_label
from src.utils.llm_factory import DEFAULT_CONSOLIDATE_LLM_MODEL, LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger
from src.utils.prompt_utils import fill_prompt_template

EXISTING_MATCH_RULES = """

[Matching New Classes Against Existing]
- Labels marked [EXISTING] come from previous extractions; labels marked [NEW] are from this one
- Group a [NEW] label with an [EXISTING] label when they refer to the same thing
- NEVER group two [EXISTING] labels together; they were consolidated already
- A [NEW] label that matches nothing stays in a group of its own
- Leave the [EXISTING] and [NEW] markers out of canonical_label
"""


class ClassConsolidator(Runnable):
    """
    Class to consolidate similar class labels
    Judges same concept with LLM and integrates by selecting canonical label
    """

    # How much of each label's origin the verify pass sees
    MAX_SOURCES = 3
    MAX_SOURCE_CHARS = 300

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        log_dir: str = "/workspace/app/log/class_consolidator",
        metamodel_path: str = None,
    ):
        self.model = model or DEFAULT_CONSOLIDATE_LLM_MODEL
        self.temperature = temperature
        self.reasoning_effort = "low"
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # Load metamodel
        self.metamodel_path = metamodel_path
        self.metamodel = self._load_metamodel()
        # Load prompt templates
        self.candidate_template = self._load_prompt_template("class_consolidator_candidates.txt")
        self.verify_template = self._load_prompt_template("class_consolidator_verify.txt")
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
        metamodel_path = self.metamodel_path
        if not metamodel_path:
            # Go up 3 levels from src/extract/class/ to /workspace/app/
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self, filename: str) -> str:
        """Load prompt template file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts", filename)
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

    def _build_schema(self, filename: str) -> dict:
        """Load JSON schema for LLM"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas", "class-consolidator", filename)
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

    def match_against_existing(self, new_classes: list, frozen_classes: list, config=None) -> Dict[str, Any]:
        """
        Match already-consolidated new classes against the existing ones carried over.

        Runs after invoke(), which consolidates the new classes among themselves.
        Input new_classes are already consolidated, so typically small (5-10 classes).

        Parameters
        ----------
        new_classes: list
            Already-consolidated new class instances (output from invoke())
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
                f"[ClassConsolidator existing] Matching {len(new_classes)} new classes against "
                f"{len(frozen_classes)} frozen classes"
            )

        # Group by class_iri (only process class_iris that appear in BOTH new and frozen)
        new_by_iri = self._group_classes_by_class_iri(new_classes)
        frozen_by_iri = self._group_classes_by_class_iri(frozen_classes)

        common_iris = set(new_by_iri.keys()) & set(frozen_by_iri.keys())

        # Process each common class_iri: match new against frozen
        all_classes = []

        # Process all common_iris in parallel
        llm = self._create_llm()
        matched_results = asyncio.run(
            self._match_against_existing_parallel(common_iris, new_by_iri, frozen_by_iri, llm)
        )

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
                f"[ClassConsolidator existing] Completed: {len(new_classes)} new + {len(frozen_classes)} frozen "
                f"→ {len(all_classes)} final classes"
            )

        self.logger.save_log(output, filename_prefix="class_consolidator_existing_output_")

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
        verify_sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(
                f"[ClassConsolidator] Starting parallel processing: {total} class types "
                f"(concurrency={self.max_concurrency})"
            )

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                consolidated = await self._consolidate_classes_async(llm, class_instances, class_iri, verify_sem)

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] ClassConsolidator: {class_name} processed "
                            f"({len(class_instances)} -> {len(consolidated)} instances)"
                        )

                return consolidated

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # Flatten results
        all_consolidated = []
        for result in results:
            all_consolidated.extend(result)

        return all_consolidated

    async def _consolidate_classes_async(
        self, llm: Any, classes: list, class_iri: str, verify_sem: asyncio.Semaphore
    ) -> list:
        """
        Consolidate specified class list with LLM (async version)

        The candidate step proposes groups from labels alone; the verify step reads the
        source text behind each candidate and decides which members really are the same.

        Args:
            llm: LLM instance to use for consolidation
            classes: List of class instances to consolidate
            class_iri: Target class IRI
            verify_sem: Concurrency budget for the verify step

        Returns:
            Consolidated class instance list
        """
        if not classes:
            return []

        # No consolidation needed if only one instance
        if len(classes) == 1:
            return classes

        labels = [c.get("label", "") for c in classes]
        candidates = await self._propose_candidates(llm, labels, class_iri, self.custom_rules, "ClassConsolidator")
        if candidates is None:
            return classes

        async def _resolve(indices: List[int]) -> list:
            group_classes = [classes[i] for i in indices]
            if len(group_classes) == 1:
                return [self._merge_classes(group_classes, labels[indices[0]], "no other candidate")]

            async with verify_sem:
                verified = await self._verify_candidate(
                    llm, group_classes, class_iri, self.custom_rules, "ClassConsolidator verify"
                )
            if verified is None:
                # Keep the members apart rather than merging on an unreadable answer
                return [self._merge_classes([c], c.get("label", ""), "verify failed") for c in group_classes]

            return [
                self._merge_classes(members, canonical, reason)
                for members, canonical, reason in self._assemble_verified(group_classes, verified)
            ]

        results = await asyncio.gather(*[_resolve(indices) for indices in candidates])
        return [c for result in results for c in result]

    async def _propose_candidates(
        self, llm: Any, labels: List[str], class_iri: str, custom_rules: str, log_label: str
    ) -> List[List[int]] | None:
        """Candidate step: group labels that may refer to the same concept. Returns None on parse failure."""
        schema = self._build_schema("candidates.schema.json")
        llm_json = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "class_consolidator_candidates", "schema": schema},
            }
        )

        prompt = fill_prompt_template(
            self.candidate_template,
            class_name=self._get_class_name_from_iri(class_iri),
            extraction_rules=self._build_extraction_rules(class_iri),
            label_list="\n".join(f"{i}: {label}" for i, label in enumerate(labels)),
            custom_rules=custom_rules,
        )

        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            label=log_label,
        )

        result_text = self._to_text(response.content)
        try:
            groups = json.loads(result_text).get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print(f"[{log_label}] JSON parse error: {result_text}")
            return None

        return self._normalize_indices(groups, len(labels))

    async def _verify_candidate(
        self, llm: Any, group_classes: list, class_iri: str, custom_rules: str, log_label: str, labels: List[str] = None
    ) -> list | None:
        """Verify step: with the source text in view, split a candidate group and name each part."""
        schema = self._build_schema("llm.schema.json")
        llm_json = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "class_consolidator_verify", "schema": schema},
            }
        )

        prompt = fill_prompt_template(
            self.verify_template,
            class_name=self._get_class_name_from_iri(class_iri),
            extraction_rules=self._build_extraction_rules(class_iri),
            member_list=self._build_member_list(group_classes, labels),
            custom_rules=custom_rules,
        )

        response = await call_with_retry(
            lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            label=log_label,
        )

        result_text = self._to_text(response.content)
        try:
            return json.loads(result_text).get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print(f"[{log_label}] JSON parse error: {result_text}")
            return None

    def _build_member_list(self, classes: list, labels: List[str] = None) -> str:
        """Render one candidate group as index, label and the text each label came from."""
        lines = []
        for i, cls in enumerate(classes):
            lines.append(f"{i}: {labels[i] if labels else cls.get('label', '')}")
            sources = cls.get("sources") or []
            if isinstance(sources, str):
                sources = [sources]
            if not sources:
                lines.append("    source: (none recorded)")
            for source in sources[: self.MAX_SOURCES]:
                lines.append(f"    source: {str(source)[: self.MAX_SOURCE_CHARS]}")
        return "\n".join(lines)

    @staticmethod
    def _take_unseen(raw: Any, count: int, seen: set) -> List[int]:
        """Indices from one LLM group: in range, not already used, no repeats within the group."""
        indices = []
        for i in raw if isinstance(raw, list) else []:
            if isinstance(i, int) and 0 <= i < count and i not in seen:
                seen.add(i)
                indices.append(i)
        return indices

    @classmethod
    def _normalize_indices(cls, groups: list, count: int) -> List[List[int]]:
        """Keep every index exactly once. Indices the LLM dropped become groups of their own,
        so a label can never disappear because the model forgot to mention it."""
        seen = set()
        result = []
        for group in groups:
            indices = cls._take_unseen(group.get("indices"), count, seen)
            if indices:
                result.append(indices)
        result.extend([i] for i in range(count) if i not in seen)
        return result

    def _assemble_verified(self, group_classes: list, verified: list, labels: List[str] = None) -> list:
        """Map a verify result back onto class instances as (members, canonical_label, reason)."""
        member_labels = labels or [c.get("label", "") for c in group_classes]
        out = []
        seen = set()
        for group in verified:
            indices = self._take_unseen(group.get("indices"), len(group_classes), seen)
            if not indices:
                continue
            canonical = self._strip_markers(group.get("canonical_label", ""))
            choices = [self._strip_markers(member_labels[i]) for i in indices]
            if canonical not in choices:
                canonical = choices[0]
            out.append(([group_classes[i] for i in indices], canonical, group.get("reason", "")))
        for i in range(len(group_classes)):
            if i not in seen:
                out.append(([group_classes[i]], self._strip_markers(member_labels[i]), "not mentioned by verify"))
        return out

    @staticmethod
    def _strip_markers(label: str) -> str:
        return label.replace(" [EXISTING]", "").replace(" [NEW]", "")

    def _merge_classes_keeping_frozen(self, members: list, canonical_label: str, reason: str) -> list:
        """Merge a matched group so the carried-over class stays the merge base and keeps its ID."""
        frozen = [m for m in members if m.get("_frozen_from_existing_accept")]
        if len(frozen) > 1:
            # Two already-consolidated classes in one group is an LLM error; leave them alone
            return list(members)
        if frozen:
            ordered = frozen + [m for m in members if m is not frozen[0]]
            return [self._merge_classes(ordered, canonical_label, reason)]
        return [self._merge_classes(members, canonical_label, reason)]

    async def _match_against_existing_parallel(
        self, common_iris: set, new_by_iri: dict, frozen_by_iri: dict, llm: Any
    ) -> list:
        """Match against existing classes in parallel for each common class_iri"""
        sem = asyncio.Semaphore(self.max_concurrency)
        verify_sem = asyncio.Semaphore(self.max_concurrency)
        total = len(common_iris)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(
                f"[ClassConsolidator existing] Starting parallel processing: {total} class types "
                f"(concurrency={self.max_concurrency})"
            )

        async def _wrapped(class_iri: str) -> list:
            nonlocal counter
            async with sem:
                new_instances = new_by_iri[class_iri]
                frozen_instances = frozen_by_iri[class_iri]

                class_name = self._get_class_name_from_iri(class_iri)

                matched_result = await self._match_against_existing_async(
                    llm, new_instances, frozen_instances, class_iri, verify_sem
                )

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] ClassConsolidator existing: {class_name} processed "
                            f"({len(new_instances)} new vs {len(frozen_instances)} frozen → "
                            f"{len(matched_result)} final)"
                        )

                return matched_result

        results = await asyncio.gather(*[_wrapped(iri) for iri in common_iris])
        return results

    async def _match_against_existing_async(
        self, llm: Any, new_classes: list, frozen_classes: list, class_iri: str, verify_sem: asyncio.Semaphore
    ) -> list:
        """
        Match new classes against the ones carried over from previous extractions.

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

        # Create label list with [EXISTING] markers
        labels = [
            f"{c.get('label', '')} [{'EXISTING' if i < len(frozen_classes) else 'NEW'}]"
            for i, c in enumerate(all_classes)
        ]

        existing_match_rules = self.custom_rules + EXISTING_MATCH_RULES

        candidates = await self._propose_candidates(
            llm, labels, class_iri, existing_match_rules, "ClassConsolidator existing"
        )
        if candidates is None:
            return all_classes

        async def _resolve(indices: List[int]) -> list:
            group_classes = [all_classes[i] for i in indices]
            group_labels = [labels[i] for i in indices]
            if len(group_classes) == 1:
                return group_classes

            async with verify_sem:
                verified = await self._verify_candidate(
                    llm,
                    group_classes,
                    class_iri,
                    existing_match_rules,
                    "ClassConsolidator existing verify",
                    labels=group_labels,
                )
            if verified is None:
                return group_classes

            resolved = []
            for members, canonical, reason in self._assemble_verified(group_classes, verified, group_labels):
                resolved.extend(self._merge_classes_keeping_frozen(members, canonical, reason))
            return resolved

        results = await asyncio.gather(*[_resolve(indices) for indices in candidates])
        result_classes = [c for result in results for c in result]

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
        canonical_label = normalize_label(canonical_label)
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
                    f"Warning: sources length ({len(sources)}) != file_ids length ({len(file_ids)}) "
                    f"in class {cls.get('id', 'unknown')}"
                )
                sources = sources[:min_length]
                file_ids = file_ids[:min_length]

            # Collect as (source, file_id) pairs
            for source, file_id in zip(sources, file_ids, strict=True):
                all_source_file_pairs.append((source, file_id))

        # Remove duplicates while preserving order
        unique_pairs = list(dict.fromkeys(all_source_file_pairs))

        # Split pairs back to sources and file_ids
        if unique_pairs:
            merged["sources"], merged["file_ids"] = zip(*unique_pairs, strict=True)
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
