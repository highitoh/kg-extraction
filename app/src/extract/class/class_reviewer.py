from typing import Dict, Any
import asyncio
import json
import os
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_utils import fill_prompt_template
from src.utils.llm_retry import call_with_retry


class ClassReviewer(Runnable):
    """
    Filter and process extracted classes using LLM
    """

    def __init__(self,
                 llm: Any = None,
                 model: str = "gemini-3.1-flash-lite-preview",
                 temperature: float = 0.0,
                 progress: bool = True,
                 max_concurrency: int = 8,
                 log_dir: str = "/workspace/app/log/class_reviewer"):
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
        # Load custom rules
        self.custom_rules = self._load_custom_rules()

    def _build_schema(self, n: int) -> dict:
        """Load LLM schema file and set element count

        Args:
            n: Number of classes to judge
        """
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-reviewer", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Set array element count for judgments
        schema["properties"]["judgments"]["minItems"] = n
        schema["properties"]["judgments"]["maxItems"] = n

        return schema

    def _build_duplicate_schema(self) -> dict:
        """Load LLM schema file for duplicate detection

        Returns:
            dict: JSON schema for duplicate detection
        """
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-reviewer", "llm-duplicate.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metamodel(self) -> dict:
        """Load metamodel file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        metamodel_path = os.path.join(app_dir, "metamodel",
                                      "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_prohibit_rules_section(self, target_class_iri: str) -> str:
        """Dynamically generate prohibit rules section for specified class

        Args:
            target_class_iri: Target class IRI
        """
        rules_lines = []

        for cls in self.metamodel.get("classes", []):
            if cls.get("iri", "") == target_class_iri:
                if "prohibit_rules" in cls and cls.get("prohibit_rules"):
                    for rule in cls["prohibit_rules"]:
                        rules_lines.append(f"- {rule}")
                break

        result = "\n".join(rules_lines)
        return result if result else "(No prohibition rules)"

    def _load_prompt_template(self) -> str:
        """Load prompt template file and dynamically generate prohibit rules"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_reviewer.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template

    def _load_custom_rules(self) -> str:
        """Load custom rules file if it exists."""
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        custom_path = os.path.join(app_dir, "prompts", "custom", "class_reviewer_custom.txt")

        if not os.path.exists(custom_path):
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return "\n" + content if content else ""

    def _load_duplicate_prompt_template(self) -> str:
        """Load prompt template file for duplicate detection"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_reviewer_duplicate.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _get_sources(cls: Dict[str, Any], max_sources: int = 2) -> list[str]:
        """Get class sources (up to max_sources) with backward compatibility."""
        sources = cls.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        if not sources and cls.get("source"):
            sources = [cls.get("source", "")]
        return [s for s in sources[:max_sources] if isinstance(s, str)]

    def _format_class_instances(self, classes: list) -> str:
        """Format class instances with source excerpts for reviewer prompts."""
        lines = []
        for cls in classes:
            lines.append(f"- ID: {cls.get('id', '')}, Label: {cls.get('label', '')}")
            sources = self._get_sources(cls)
            if sources:
                for i, source in enumerate(sources):
                    lines.append(f"  - source[{i}]: {source[:200]}")
            else:
                lines.append("  - source[0]: ")
        return "\n".join(lines)

    def _convert_duplicate_response_to_dict(self, parsed_response: dict) -> dict:
        """Convert LLM response to internal format dictionary

        Args:
            parsed_response: JSON response from LLM (containing duplicate_groups)

        Returns:
            dict: Dictionary with group IDs as keys
                {
                    "dup_group_0": {
                        "class_ids": [...],
                        "labels": [...],
                        "reason": "..."
                    },
                    ...
                }
        """
        result = {}
        duplicate_groups = parsed_response.get("duplicate_groups", [])

        for i, group in enumerate(duplicate_groups):
            group_id = f"dup_group_{i}"
            result[group_id] = {
                "class_ids": group.get("class_ids", []),
                "labels": group.get("labels", []),
                "reason": group.get("reason", "")
            }

        return result

    async def _detect_duplicates_async(self, classes: list, class_iri: str) -> dict:
        """Detect duplicates in specified class list (async version)

        Args:
            classes: List of class instances to detect duplicates
            class_iri: Target class IRI

        Returns:
            dict: Dictionary of duplicate groups
                {
                    "dup_group_0": {
                        "class_ids": ["cls_001", "cls_003"],
                        "labels": ["User", "System User"],
                        "reason": "..."
                    },
                    ...
                }
        """
        if not classes or len(classes) < 2:
            # No duplicates if 1 or fewer
            return {}

        # Build schema
        schema = self._build_duplicate_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_reviewer_duplicate",
                    "schema": schema
                }
            }
        )

        # Extract class IDs and labels
        class_data = [
            {"id": c.get("id", ""), "label": c.get("label", "")}
            for c in classes
        ]
        class_data_json = json.dumps(class_data, ensure_ascii=False, indent=2)

        # Load prompt template
        prompt_template = self._load_duplicate_prompt_template()
        prompt = f"{prompt_template}\n{class_data_json}"

        # Execute duplicate detection with LLM (with retry)
        try:
            response = await call_with_retry(
                lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
                timeout=90.0,
                label="ClassReviewer-Duplicate",
            )

            # Extract text from response
            result_text = self._to_text(response.content)

            # Parse JSON
            parsed = json.loads(result_text)

            # Convert to internal format
            duplicate_groups = self._convert_duplicate_response_to_dict(parsed)

            return duplicate_groups

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassReviewer] Duplicate detection JSON parse error: {e}")
            # Treat as no duplicates on error
            return {}
        except Exception as e:
            if self.progress:
                print(f"[ClassReviewer] Duplicate detection error: {e}")
            return {}

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        classes = input.get("classes", [])

        if self.progress:
            print(f"[ClassReviewer] Processing {len(classes)} classes")

        # Group by class_iri
        class_groups = self._group_classes_by_class_iri(classes)

        if self.progress:
            print(f"[ClassReviewer] {len(class_groups)} class types found")

        # Execute filtering in parallel for each class
        all_classes_with_judgment, all_filtered_classes = asyncio.run(
            self._process_classes_parallel(class_groups)
        )

        # Build output
        output = input.copy()
        output["classes"] = all_filtered_classes

        if self.progress:
            print(
                f"[ClassReviewer] Completed: {len(classes)} -> {len(all_filtered_classes)} classes"
            )

        log_output = input.copy()
        log_output["classes"] = all_classes_with_judgment
        self.logger.save_log(log_output, filename_prefix="class_reviewer_output_")

        return output

    def _merge_review_results(self, classes: list, prohibit_judgments: list, duplicate_groups: dict) -> tuple[list, list]:
        """Merge results of prohibit rule review and duplicate detection

        Args:
            classes: Original class instance list
            prohibit_judgments: List of judgment results from prohibit rule review
            duplicate_groups: Dictionary of duplicate groups

        Returns:
            tuple:
              - all class results including REJECT
              - filtered class list excluding REJECT
        """
        all_results = []
        filtered_results = []

        for i, cls in enumerate(classes):
            cls_result = cls.copy()
            class_id = cls.get("id", "")

            # Get results from prohibit rule review
            judgment = prohibit_judgments[i].get("judgment", "REJECT")
            justification = prohibit_judgments[i].get("justification", "")

            # Check if included in duplicate group
            found_in_duplicate = False
            for group_id, group_info in duplicate_groups.items():
                if class_id in group_info.get("class_ids", []):
                    found_in_duplicate = True

                    # Process when duplicate is found
                    if judgment != "REVIEW":
                        # If not REVIEW in prohibit rules, change to REVIEW with duplicate reason
                        judgment = "REVIEW"
                        justification = f"Duplicate detected: {group_info.get('reason', '')}"
                    # else: If REVIEW in prohibit rules, prioritize that justification

                    # Add duplicate fields
                    # duplicate_group_id: Temporary field for ClassEditor to identify duplicate group
                    cls_result["duplicate_group_id"] = group_id
                    # consolidated_ids/consolidated_labels are empty initially
                    # ClassEditor sets actual merged IDs and labels during merge
                    cls_result["consolidated_ids"] = []
                    cls_result["consolidated_labels"] = []
                    break

            cls_result["judgment"] = judgment
            cls_result["justification"] = justification
            all_results.append(cls_result)

            # Include only non-REJECT classes in filtered results
            if judgment != "REJECT":
                filtered_results.append(cls_result)

        return all_results, filtered_results

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> tuple[list, list]:
        """Process class groups in parallel (execute prohibit rule review and duplicate detection concurrently)"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassReviewer] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})")

        async def _wrapped(class_iri: str, class_instances: list) -> tuple[list, list]:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                # Run LLM-backed review steps serially to avoid reusing async client
                # resources concurrently within the same reviewer instance.
                prohibit_judgments = await self._review_prohibit_rules_async(
                    class_instances, class_iri
                )
                duplicate_groups = await self._detect_duplicates_async(
                    class_instances, class_iri
                )

                # Merge results
                all_results, filtered = self._merge_review_results(
                    class_instances, prohibit_judgments, duplicate_groups
                )

                async with lock:
                    counter += 1
                    if self.progress:
                        # Aggregate judgment result breakdown
                        accept_count = sum(1 for c in filtered if c.get("judgment") == "ACCEPT")
                        review_count = sum(1 for c in filtered if c.get("judgment") == "REVIEW")
                        reject_count = len(class_instances) - len(filtered)
                        duplicate_count = len(duplicate_groups)

                        print(f"[{counter}/{total}] ClassReviewer: {class_name} processed "
                              f"({len(class_instances)} -> {len(filtered)} instances) "
                              f"[ACCEPT: {accept_count}, REVIEW: {review_count}, REJECT: {reject_count}] "
                              f"[Duplicate groups: {duplicate_count}]")

                return all_results, filtered

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # Flatten results
        all_with_judgment = []
        all_filtered = []
        for all_group, filtered_group in results:
            all_with_judgment.extend(all_group)
            all_filtered.extend(filtered_group)

        return all_with_judgment, all_filtered

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

    async def _review_prohibit_rules_async(self, classes: list, class_iri: str) -> list:
        """Review specified class list for prohibit rules with LLM (async version)

        Args:
            classes: List of class instances to review
            class_iri: Target class IRI (use only this class's prohibit_rules)

        Returns:
            list: List of judgment results [{"judgment": "ACCEPT", "justification": "..."}, ...]
        """
        if not classes:
            return []

        schema = self._build_schema(len(classes))
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_reviewer",
                    "schema": schema
                }
            })

        # Format class instances with source excerpts to give reviewer context.
        class_instances_text = self._format_class_instances(classes)

        # Get only the target class's prohibit_rules
        prohibit_rules = self._build_prohibit_rules_section(class_iri)

        # Build prompt (embed prohibit rules and class labels specific to target class)
        prompt = fill_prompt_template(
            self.prompt_template,
            prohibit_rules=prohibit_rules,
            class_instances=class_instances_text,
            custom_rules=self.custom_rules,
        )

        try:
            # Execute filtering with LLM (with retry + timeout)
            response = await call_with_retry(
                lambda: llm_json.ainvoke([HumanMessage(content=prompt)]),
                timeout=90.0,
                label="ClassReviewer-Prohibit",
            )

            # Extract text from response
            result_text = self._to_text(response.content)

            # Parse JSON
            parsed = json.loads(result_text)
            judgments = parsed["judgments"]
            return judgments

        except asyncio.TimeoutError:
            if self.progress:
                print(f"[ClassReviewer] Prohibit rules review timeout for class_iri={class_iri}")
            return [{"judgment": "REJECT", "justification": "LLM call timed out"} for _ in classes]
        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassReviewer] Prohibit rules review JSON parse error: {e}, response: {result_text}")
            # Treat all as REJECT on error
            return [{"judgment": "REJECT", "justification": "Parse error"} for _ in classes]
        except Exception as e:
            if self.progress:
                print(f"[ClassReviewer] Prohibit rules review error: {e}")
            return [{"judgment": "REJECT", "justification": "LLM call failed"} for _ in classes]

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
    import json
    import glob
    import os

    # Find the latest ClassExtractor log file
    log_pattern = "/workspace/app/log/class_extractor/class_extractor_output_*.json"
    log_files = glob.glob(log_pattern)

    if not log_files:
        print(f"No log files found matching pattern: {log_pattern}")
        exit(1)

    # Get the most recent file
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading test input from: {latest_log}")

    # Load the JSON data
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('classes', []))} classes")

    # Create and run the filter
    class_reviewer = ClassReviewer(progress=True, log_dir="/workspace/app/log/class_reviewer")
    result = class_reviewer.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
