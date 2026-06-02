from typing import Dict, Any, List, Union, Tuple
import asyncio
import json
import os
import uuid
import csv
import io

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_utils import fill_prompt_template
from src.utils.llm_retry import call_with_retry
from src.utils.chunk_creator import ChunkCreator


class ViewExtractor(Runnable):
    """
    Task to extract view descriptions
    Input: ViewChainInput
    Output: ViewChainOutput
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        max_spans_per_label: int = 3,
        log_dir: str = "/workspace/app/log/view_extractor",
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )

        # Load JSON schema
        schema = self._load_schema()
        self.llm_json = self.llm.bind(
            response_format={"type": "json_schema", "json_schema": {"name": "view_extractor", "schema": schema}}
        )
        self.max_concurrency = max_concurrency
        self.progress = progress
        self.max_spans_per_label = max_spans_per_label

        # Load prompt file
        self.prompt = self._load_prompt()

        # Target views
        self.views = [
            "value_analysis",
            "business_concept",
            "business_operation",
            "business_requirement",
            "system_requirement",
            "system_component",
            "data_analysis",
        ]

        # Label code mapping
        self.label_codes = {
            "VA": "value_analysis",
            "BC": "business_concept",
            "BO": "business_operation",
            "BR": "business_requirement",
            "SR": "system_requirement",
            "SC": "system_component",
            "DA": "data_analysis",
        }

        # Initialize Logger
        self.logger = Logger(log_dir)

    def _load_schema(self) -> dict:
        """Load JSON schema file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        schema_path = os.path.join(app_dir, "schemas", "view-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        prompt_path = os.path.join(app_dir, "prompts", "view_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """Convert LLM message to text"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    def _segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences using ChunkCreator utility"""
        cc = ChunkCreator()
        return cc._split_into_sentences(text)

    def _format_as_csv(self, sentences: List[str], chunk_index: int) -> Tuple[str, Dict[str, str]]:
        """
        Format sentences as CSV with id,text structure

        Returns:
            Tuple of (csv_string, sentence_map) where sentence_map is {id: text}
        """
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(["id", "text"])

        # Track sentence ID to text mapping
        sentence_map = {}

        # Write sentences with IDs like c{chunk_idx}_s{sent_idx}
        for sent_idx, sentence in enumerate(sentences):
            sentence_id = f"c{chunk_index}_s{sent_idx}"
            writer.writerow([sentence_id, sentence])
            sentence_map[sentence_id] = sentence

        return output.getvalue(), sentence_map

    async def _extract_spans_for_chunk(self, chunk: str, chunk_index: int) -> Dict[str, List[str]]:
        """Extract view-specific spans for one chunk using CSV classification"""
        # Segment chunk into sentences
        sentences = self._segment_sentences(chunk)

        # Handle empty chunks
        if not sentences:
            return {k: [] for k in self.views}

        # Format as CSV and get sentence mapping
        csv_input, sentence_map = self._format_as_csv(sentences, chunk_index)

        # Format the prompt with the CSV input
        prompt_text = fill_prompt_template(self.prompt, csv_input=csv_input)

        msg = HumanMessage(content=[{"type": "text", "text": prompt_text}])
        ai = await call_with_retry(
            lambda: self.llm_json.ainvoke([msg]),
            timeout=90.0,
            label="ViewExtractor",
        )
        raw = self._to_text(ai.content)
        classification_results: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                classification_results = data.get("classifications", [])
            elif isinstance(data, list):
                # Backward compatibility if the model returns a bare array
                classification_results = data
        except Exception:
            classification_results = []

        if not isinstance(classification_results, list):
            classification_results = []

        # Parse classification results and reconstruct spans by view type
        spans: Dict[str, List[str]] = {k: [] for k in self.views}

        for item in classification_results:
            if not isinstance(item, dict):
                continue

            sentence_id = item.get("id", "")
            labels_str = item.get("labels", "")

            # Skip if sentence ID not found or labels invalid
            if sentence_id not in sentence_map or not labels_str:
                continue

            # Skip "NO" (none_of_the_above)
            if labels_str == "NO":
                continue

            # Split pipe-delimited labels
            label_codes = labels_str.split("|")

            # Get original sentence text
            sentence_text = sentence_map[sentence_id]

            # Map label codes to full view names and add sentence
            for code in label_codes:
                code = code.strip()
                if code in self.label_codes:
                    view_name = self.label_codes[code]
                    # Add sentence if not already present and within limit
                    if sentence_text not in spans[view_name]:
                        if len(spans[view_name]) < self.max_spans_per_label:
                            spans[view_name].append(sentence_text)

        return spans

    async def _process_chunks_parallel(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Extract views from chunks using parallel processing"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(chunks)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ViewExtractor] Starting parallel processing: {total} chunks (concurrency={self.max_concurrency})")

        async def _wrapped(i: int, c: str) -> Dict[str, Any]:
            nonlocal counter
            async with sem:
                spans = {k: [] for k in self.views}
                for attempt in range(5):
                    try:
                        spans = await self._extract_spans_for_chunk(c, i)
                        break
                    except Exception:
                        await asyncio.sleep(min(2**attempt, 10))
                async with lock:
                    counter += 1
                    total_spans = sum(len(v) for v in spans.values())
                    if self.progress:
                        print(f"[{counter}/{total}] ViewExtractor: chunk {i} processed (total_spans={total_spans})")
                return {"index": i, "spans": spans, "chunk": c}

        results = await asyncio.gather(*[_wrapped(i, c) for i, c in enumerate(chunks)])
        return results

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        target = input["target"]
        chunks_data: List[Dict[str, Any]] = target["chunks"]
        file_id = target.get("file_id", str(uuid.uuid4()))

        # Get list of chunk texts
        chunks = [chunk["text"] for chunk in chunks_data]

        # Extract views with parallel execution
        chunk_results = asyncio.run(self._process_chunks_parallel(chunks))

        # Convert results to ViewChainOutput format
        views = []
        for view_type in self.views:
            texts = []
            for result in chunk_results:
                spans = result["spans"].get(view_type, [])
                for span in spans:
                    texts.append(
                        {
                            "file_id": file_id,
                            "text": span,
                        }
                    )

            if texts:  # Add view only if extracted texts exist
                views.append(
                    {
                        "type": view_type,
                        "texts": texts,
                    }
                )

        output: Dict[str, Any] = {"views": views}

        # Log ViewChainOutput using Logger
        self.logger.save_log(output, filename_prefix="view_extractor_output_")

        return output


if __name__ == "__main__":
    # Sample data for testing
    sample_input = {
        "target": {
            "file_id": "sample_file",
            "chunks": [
                {
                    "text": "The system needs to manage customer information. The customer database stores name, address, and phone number."
                },
                {"text": "The project goal is to increase sales. The API server is built with RESTful design."},
            ],
        },
        "metamodel": {"version": "1.0", "domain": "business"},
    }

    # Create ViewExtractor instance
    print("ViewExtractor test starting...")
    extractor = ViewExtractor(
        model="gpt-5-nano",
        temperature=0.0,
        max_concurrency=2,  # Limit concurrency for testing
        max_spans_per_label=2,  # Limit for testing
    )

    print(f"Target views: {extractor.views}")

    try:
        # Run test
        result = extractor.invoke(sample_input)

        print("\n=== Test Results ===")
        print(f"Extracted Views: {len(result['views'])}")

        for view in result["views"]:
            print(f"\nView Type: {view['type']}")
            print(f"Extracted Texts: {len(view['texts'])}")
            for i, text in enumerate(view["texts"][:2]):  # Show only first 2 items
                print(f"  [{i + 1}] {text['text']}")

        print("\n✅ ViewExtractor test completed")

    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        import traceback

        traceback.print_exc()
