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


def async_retry_with_backoff(max_retries: int = 5, initial_delay: float = 1.0, max_delay: float = 60.0):
    """
    Retry decorator for async functions (with exponential backoff)

    Args:
        max_retries: Maximum number of retries
        initial_delay: Wait time before first retry (seconds)
        max_delay: Maximum wait time (seconds)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (APIConnectionError, RateLimitError, APIError) as e:
                    last_exception = e

                    if attempt == max_retries:
                        raise

                    # Wait before retrying
                    wait_time = min(delay, max_delay)
                    print(f"[Retry {attempt + 1}/{max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

                    # Exponential backoff
                    delay *= 2

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class ClassReviewer(Runnable):
    """
    Filter and process extracted classes using LLM
    """

    def __init__(self,
                 llm: Any = None,
                 model: str = "gpt-5-mini",
                 temperature: float = 0.0,
                 progress: bool = True,
                 max_concurrency: int = 8,
                 max_retries: int = 5,
                 log_dir: str = "/workspace/app/log/class_reviewer"):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
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
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-reviewer", "llm-duplicate.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metamodel(self) -> dict:
        """Load metamodel file"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
        return result if result else "(禁止ルールなし)"

    def _load_prompt_template(self) -> str:
        """Load prompt template file and dynamically generate prohibit rules"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_reviewer.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template

    def _load_duplicate_prompt_template(self) -> str:
        """Load prompt template file for duplicate detection"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_reviewer_duplicate.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

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
            response = await self._invoke_with_retry(llm_json, prompt)

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
        all_filtered_classes = asyncio.run(self._process_classes_parallel(class_groups))

        # Build output
        output = input.copy()
        output["classes"] = all_filtered_classes

        if self.progress:
            print(
                f"[ClassReviewer] Completed: {len(classes)} -> {len(all_filtered_classes)} classes"
            )

        self.logger.save_log(output, filename_prefix="class_reviewer_output_")

        return output

    def _merge_review_results(self, classes: list, prohibit_judgments: list, duplicate_groups: dict) -> list:
        """Merge results of prohibit rule review and duplicate detection

        Args:
            classes: Original class instance list
            prohibit_judgments: List of judgment results from prohibit rule review
            duplicate_groups: Dictionary of duplicate groups

        Returns:
            list: Merged class list (excluding REJECT)
        """
        results = []

        for i, cls in enumerate(classes):
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
                    cls["duplicate_group_id"] = group_id
                    # consolidated_ids/consolidated_labels are empty initially
                    # ClassEditor sets actual merged IDs and labels during merge
                    cls["consolidated_ids"] = []
                    cls["consolidated_labels"] = []
                    break

            # Include only non-REJECT classes in results
            if judgment != "REJECT":
                cls["judgment"] = judgment
                cls["justification"] = justification
                results.append(cls)

        return results

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> list:
        """Process class groups in parallel (execute prohibit rule review and duplicate detection concurrently)"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassReviewer] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})")

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                # Execute prohibit rule review and duplicate detection in parallel
                prohibit_judgments, duplicate_groups = await asyncio.gather(
                    self._review_prohibit_rules_async(class_instances, class_iri),
                    self._detect_duplicates_async(class_instances, class_iri)
                )

                # Merge results
                merged = self._merge_review_results(class_instances, prohibit_judgments, duplicate_groups)

                async with lock:
                    counter += 1
                    if self.progress:
                        # Aggregate judgment result breakdown
                        accept_count = sum(1 for c in merged if c.get("judgment") == "ACCEPT")
                        review_count = sum(1 for c in merged if c.get("judgment") == "REVIEW")
                        reject_count = len(class_instances) - len(merged)
                        duplicate_count = len(duplicate_groups)

                        print(f"[{counter}/{total}] ClassReviewer: {class_name} processed "
                              f"({len(class_instances)} -> {len(merged)} instances) "
                              f"[ACCEPT: {accept_count}, REVIEW: {review_count}, REJECT: {reject_count}] "
                              f"[Duplicate groups: {duplicate_count}]")

                return merged

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # Flatten results
        all_filtered = []
        for result in results:
            all_filtered.extend(result)

        return all_filtered

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

        # Extract only labels to pass to prompt
        labels = [c.get("label", "") for c in classes]
        labels_json = json.dumps(labels, ensure_ascii=False, indent=2)

        # Get only the target class's prohibit_rules
        prohibit_rules = self._build_prohibit_rules_section(class_iri)

        # Build prompt (embed prohibit rules specific to target class)
        prompt_with_rules = self.prompt_template.replace("{PROHIBIT_RULES}", prohibit_rules)
        prompt = f"{prompt_with_rules}\n{labels_json}"

        # Execute filtering with LLM (with retry)
        response = await self._invoke_with_retry(llm_json, prompt)

        # Extract text from response
        result_text = self._to_text(response.content)

        # Parse JSON
        try:
            parsed = json.loads(result_text)
            judgments = parsed["judgments"]
            return judgments

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassReviewer] Prohibit rules review JSON parse error: {e}, response: {result_text}")
            # Treat all as REJECT on error
            return [{"judgment": "REJECT", "justification": "Parse error"} for _ in classes]

    async def _invoke_with_retry(self, llm_json, prompt: str):
        """LLM invocation with retry functionality"""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await llm_json.ainvoke([HumanMessage(content=prompt)])
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[ClassReviewer Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

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
