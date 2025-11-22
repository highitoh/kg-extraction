from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory

VIEW_TYPES = [
    "value_analysis",
    "business_concept",
    "business_requirement",
    "system_requirement",
    "system_component",
    "data_analysis",
]


@dataclass
class ExtractedClass:
    """One element of ClassExtractOutput.classes"""
    id: str              # Required: Class instance ID
    class_iri: str       # Required: URI format
    label: str           # Required: Extracted class (display label)
    sources: List[str]   # Required: Source text (array)
    file_ids: List[str]  # Required: Source file IDs (array)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClassExtractor(Runnable):
    """
    Class for extracting class candidates from view descriptions
    """

    def __init__(
        self,
        llm: Any = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        max_retries: int = 5,
        log_dir: str = "/workspace/app/log/class_extractor"
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.logger = Logger(log_dir)

        # Load JSON schema
        schema = self._load_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "class_extractor", "schema": schema}})

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

    def _load_schema(self) -> dict:
        """Load JSON schema file"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        schema_path = os.path.join(app_dir, "schemas", "class-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """Load prompt template file"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts", "class_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def invoke(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parameters
        ----------
        input_data: Conforms to ClassExtractorInput
          - view_info: View extraction results (assuming texts etc. in each view)
          - metamodel: Metamodel (assumed to be used for class IRI determination, detailed implementation later)
        Returns
        -------
        output: Conforms to ClassCExtractorOutput
        """
        view_info = self._get_view_info(input_data)
        metamodel = self._get_metamodel(input_data)

        # Execute in parallel for each view
        view_results = asyncio.run(self._process_views_parallel(view_info, metamodel))

        # Add class_iri and format schema
        classes: List[ExtractedClass] = []
        for result in view_results:
            view_type = result["view_type"]
            label_items = result["label_items"]

            for li in label_items:
                c = li["class"]
                label = li["label"]
                source = li["source"]
                file_id = li["file_id"]
                class_iri = self._resolve_class_iri(view_type, c, metamodel)  # Implementation to return URI

                # Ensure 1:1 correspondence between sources and file_ids
                sources = [source]
                file_ids = [file_id]

                classes.append(ExtractedClass(
                    id=str(uuid.uuid4()),
                    class_iri=class_iri,
                    label=label,
                    sources=sources,
                    file_ids=file_ids,
                ))

        # ---- Output (conforms to ClassChainOutput) -----------------------------------------
        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "classes": [c.to_dict() for c in classes],  # Schema: meets minItems=1 after implementation
        }

        if self.progress:
            print(f"[ClassExtractor] Completed: extracted {len(classes)} classes")

        # Log ClassExtractorOutput using Logger
        self.logger.save_log(output, filename_prefix="class_extractor_output_")

        return output

    def _get_view_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "view_info" not in input_data or not isinstance(input_data["view_info"], dict):
            raise ValueError("ClassChainInput: 'view_info' is missing.")
        return input_data["view_info"]

    def _get_metamodel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "metamodel" not in input_data or not isinstance(input_data["metamodel"], dict):
            raise ValueError("ClassChainInput: 'metamodel' is missing.")
        return input_data["metamodel"]

    def _get_texts_for_view(self, view_info: Dict[str, Any], view_type: str) -> List[Dict[str, Any]]:
        """
        Template to get texts for a specific view_type from view_info as a flat list.
        """
        views = view_info.get("views", [])
        for v in views:
            if v.get("type") == view_type:
                return v.get("texts", []) or []
        return []

    async def _process_views_parallel(self, view_info: Dict[str, Any], metamodel: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute class extraction in parallel for each view"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(VIEW_TYPES)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassExtractor] Starting parallel processing: {total} views (concurrency={self.max_concurrency})")

        async def _wrapped(view_type: str) -> Dict[str, Any]:
            nonlocal counter
            async with sem:
                texts = self._get_texts_for_view(view_info, view_type)

                label_items = []
                for attempt in range(5):
                    try:
                        label_items = await self._extract_labels_from_view_async(view_type, texts, metamodel)
                        break
                    except Exception as e:
                        if attempt == 4:  # Last attempt
                            if self.progress:
                                print(f"[ClassExtractor] Error in view={view_type}: {e}")
                        await asyncio.sleep(min(2 ** attempt, 10))

                async with lock:
                    counter += 1
                    if self.progress:
                        print(f"[{counter}/{total}] ClassExtractor: {view_type} processed (extracted {len(label_items)} items)")

                return {"view_type": view_type, "label_items": label_items}

        results = await asyncio.gather(*[_wrapped(vt) for vt in VIEW_TYPES])
        return results

    async def _extract_labels_from_view_async(
        self,
        view_type: str,
        texts: List[Dict[str, Any]],
        metamodel: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Process to extract class candidates from target view descriptions (async version).
        Returns:
          [{"class": "<class>", "label": "<instance>", "source": "<source>", "file_id": "<source file ID>"} , ...]
        """
        return await self._extract_labels_async(view_type, texts, metamodel)

    async def _extract_labels_async(self, view_type: str, texts: List[Dict[str, Any]], metamodel: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Extract class instances from specified view type using LLM (async version)
        """
        if not texts:
            return []

        # Get text of view description
        chunk_lines = []
        for idx, t in enumerate(texts):
            chunk_lines.append(f"{t.get('text', '')}")
        chunks_text = "\n".join(chunk_lines)

        # Generate class definition and policy strings from metamodel
        class_definitions = self._get_class_definitions(metamodel, view_type)
        class_policies = self._get_class_policies(metamodel, view_type)

        # Embed values into prompt template
        prompt = self.prompt_template.format(
            class_definitions=class_definitions,
            class_policies=class_policies
        )
        prompt = f"{prompt}\n\n【テキスト】\n{chunks_text}\n"

        response = await self._invoke_with_retry(HumanMessage(content=prompt))

        # Extract text from response.content
        result_text = self._to_text(response.content)

        # Parse JSON
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            if self.progress:
                print(f"[ClassExtractor] JSON parse error: {result_text}")
            return []

        # Convert to label candidate list
        labels = []
        value_items = result.get("classes", [])
        for item in value_items:
            if "class" in item and "label" in item and "source" in item:
                labels.append({
                    "class": item["class"],
                    "label": item["label"],
                    "source": item["source"],
                    "file_id": texts[0].get("file_id", "unknown")  # Use first file ID for convenience
                })

        return labels

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
                    print(f"[ClassExtractor Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

    def _get_class_definitions(self, metamodel: Dict[str, Any], view_type: str) -> str:
        """
        Generate class definition string for specified view_type from metamodel

        Parameters
        ----------
        metamodel : Dict[str, Any]
            Metamodel definition
        view_type : str
            Target view type

        Returns
        -------
        str
            Class definition string (e.g., "- Stakeholder: description\n- Value: description")
        """
        if not metamodel or "classes" not in metamodel:
            return ""

        classes = metamodel.get("classes", [])
        view_classes = [cls for cls in classes if cls.get("view_type") == view_type]

        # Build class definitions
        definitions = []
        for cls in view_classes:
            name = cls.get("name", "")
            description = cls.get("description", "")
            definitions.append(f"- {name}: {description}")

        return "\n".join(definitions)

    def _get_class_policies(self, metamodel: Dict[str, Any], view_type: str) -> str:
        """
        Generate class policy string for specified view_type from metamodel

        Parameters
        ----------
        metamodel : Dict[str, Any]
            Metamodel definition
        view_type : str
            Target view type

        Returns
        -------
        str
            Class policy string
        """
        if not metamodel or "classes" not in metamodel:
            return ""

        classes = metamodel.get("classes", [])
        view_classes = [cls for cls in classes if cls.get("view_type") == view_type]

        # Build policies
        policies = []
        for cls in view_classes:
            name = cls.get("name", "")
            policy_items = []

            # Token count constraints
            min_len = cls.get("min_length")
            max_len = cls.get("max_length")
            if min_len is not None and max_len is not None:
                policy_items.append(f"  - Token count: {min_len}-{max_len} tokens")

            # Format guidelines (array support)
            extract_rules = cls.get("extract_rules", [])
            for guideline in extract_rules:
                policy_items.append(f"  - {guideline}")

            if policy_items:
                policies.append(f"- {name}")
                policies.extend(policy_items)

        return "\n".join(policies)

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

    def _resolve_class_iri(
        self,
        view_type: str,
        class_name: str,
        metamodel: Dict[str, Any],
    ) -> str:
        """
        Process to get class_iri (URI) corresponding to extracted class
        """
        # Search for class_iri mapping from metamodel
        classes = metamodel.get("classes", [])
        for cls in classes:
            if cls.get("view_type") == view_type and cls.get("name") == class_name:
                return cls.get("iri", "")

        # Return empty string if not found
        return ""

if __name__ == "__main__":
    import glob
    import os

    # Get the latest view_filter log file from log directory
    log_dir = "/workspace/app/log/view_filter"
    log_files = glob.glob(os.path.join(log_dir, "view_filter_output_*.json"))

    if not log_files:
        print("Error: No view_filter log files found")
        exit(1)

    # Get the latest log file
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")
    with open(latest_log, "r", encoding="utf-8") as f:
        view_info = json.load(f)

    # Get metamodel
    # Go up 2 levels from src/XXX/ to /workspace/app/

    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
    with open(metamodel_path, "r", encoding="utf-8") as f:
        metamodel = json.load(f)

    sample_input = {
        "view_info": view_info,
        "metamodel": metamodel,
    }

    extractor = ClassExtractor(progress=True)
    out = extractor.invoke(sample_input)
    print(out)
