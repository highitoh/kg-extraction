from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
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
from src.utils.id_generator import generate_class_id

VIEW_TYPES = [
    "value_analysis",
    "business_concept",
    "business_operation",
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
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        log_dir: str = "/workspace/app/log/class_extractor"
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # Load JSON schema
        schema = self._load_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "class_extractor", "schema": schema}})

        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        # Load custom rules
        self.custom_rules = self._load_custom_rules()

    def _load_schema(self) -> dict:
        """Load JSON schema file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_path = os.path.join(app_dir, "schemas", "class-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """Load prompt template file"""
        # Go up 3 levels from src/extract/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        prompt_path = os.path.join(app_dir, "prompts", "class_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_custom_rules(self) -> str:
        """Load custom rules file if it exists"""
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        custom_path = os.path.join(app_dir, "prompts", "custom", "class_extractor_custom.txt")

        if not os.path.exists(custom_path):
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return "\n" + content if content else ""

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
                    id=generate_class_id(class_iri, label, source),
                    class_iri=class_iri,
                    label=label,
                    sources=sources,
                    file_ids=file_ids,
                ))

        # ---- Output (conforms to ClassChainOutput) -----------------------------------------
        output: Dict[str, Any] = {
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

    def _format_texts_with_ids(self, texts: List[Dict[str, Any]]) -> tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Format texts with IDs for LLM prompt in CSV format.

        Args:
            texts: List of {text, file_id} dicts from ViewChainOutput

        Returns:
            Tuple of (csv_string, text_map)
            - csv_string: CSV format with "ID,Text" header and rows
            - text_map: {text_id: {"text": str, "file_id": str}}
        """
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Write CSV header
        writer.writerow(['ID', 'Text'])

        # Create text_map to store full text and file_id for each ID
        text_map = {}

        for idx, t in enumerate(texts):
            text_id = f"t{idx}"
            text_content = t.get('text', '')
            file_id = t.get('file_id', 'unknown')

            writer.writerow([text_id, text_content])
            text_map[text_id] = {
                "text": text_content,
                "file_id": file_id
            }

        return output.getvalue(), text_map

    async def _extract_labels_async(self, view_type: str, texts: List[Dict[str, Any]], metamodel: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Extract class instances from specified view type using LLM (async version)
        """
        if not texts:
            return []

        # Format texts with IDs and create mapping
        texts_csv, text_map = self._format_texts_with_ids(texts)

        # Generate class definition and policy strings from metamodel
        class_definitions = self._get_class_definitions(metamodel, view_type)
        class_policies = self._get_class_policies(metamodel, view_type)

        # Embed values into prompt template
        prompt = fill_prompt_template(
            self.prompt_template,
            class_definitions=class_definitions,
            class_policies=class_policies,
            custom_rules=self.custom_rules,
        )
        prompt = f"{prompt}\n\n[Text]\n{texts_csv}\n"

        response = await call_with_retry(
            lambda: self.llm_json.ainvoke([HumanMessage(content=prompt)]),
            timeout=90.0,
            label="ClassExtractor",
        )

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
            # Validate required fields (note: source_id replaces source)
            if not all(k in item for k in ["class", "label", "source_id"]):
                if self.progress:
                    print(f"[ClassExtractor] Warning: Missing required field in response item: {item}")
                continue

            text_id = item["source_id"]

            # Resolve text_id to full text and file_id
            if text_id in text_map:
                text_info = text_map[text_id]
                labels.append({
                    "class": item["class"],
                    "label": item["label"],
                    "source": text_info["text"],      # Resolved full text (backward compatible)
                    "file_id": text_info["file_id"],  # Correct file_id (fixes existing bug!)
                })
            else:
                # Handle invalid text ID from LLM
                if self.progress:
                    print(f"[ClassExtractor] Warning: Invalid text ID '{text_id}' not found in text_map")

        return labels


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
    # Go up 3 levels from src/extract/XXX/ to /workspace/app/

    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
