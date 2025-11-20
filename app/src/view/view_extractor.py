from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory

class ViewExtractor(Runnable):
    """
    Task to extract view descriptions
    Input: ViewChainInput
    Output: ViewChainOutput
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
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
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "view_extractor", "schema": schema}})
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.progress = progress
        self.max_spans_per_label = max_spans_per_label

        # Load prompt file
        self.prompt = self._load_prompt()

        # Target views
        self.views = [
            "value_analysis",
            "business_concept",
            "business_requirement",
            "system_requirement",
            "system_component",
            "data_analysis",
        ]

        # Initialize Logger
        self.logger = Logger(log_dir)

    def _load_schema(self) -> dict:
        """Load JSON schema file"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        schema_path = os.path.join(app_dir, "schemas", "view-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

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

    async def _extract_spans_for_chunk(self, chunk: str) -> Dict[str, List[str]]:
        """Extract view-specific spans for one chunk"""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "text", "text": f"# 対象テキスト\n{chunk}"},
            ]
        )
        ai = await self._invoke_with_retry(msg)
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        # Normalize (only known keys, remove duplicates, limit maximum count)
        spans: Dict[str, List[str]] = {k: [] for k in self.views}
        for k in self.views:
            v = data.get(k, [])
            if not isinstance(v, list):
                continue
            seen = set()
            out = []
            for s in v:
                if not isinstance(s, str):
                    continue
                s2 = s.strip()
                if not s2 or s2 in seen:
                    continue
                seen.add(s2)
                out.append(s2)
                if len(out) >= self.max_spans_per_label:
                    break
            spans[k] = out
        return spans

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
                    print(f"[ViewExtractor Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

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
                        spans = await self._extract_spans_for_chunk(c)
                        break
                    except Exception:
                        await asyncio.sleep(min(2 ** attempt, 10))
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
                    texts.append({
                        "file_id": file_id,
                        "text": span,
                    })

            if texts:  # Add view only if extracted texts exist
                views.append({
                    "type": view_type,
                    "texts": texts,
                })

        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "views": views,
        }

        # Log ViewChainOutput using Logger
        self.logger.save_log(output, filename_prefix="view_extractor_output_")

        return output


if __name__ == "__main__":
    # Sample data for testing
    sample_input = {
        "target": {
            "id": "test_001",
            "file_id": "sample_file",
            "chunks": [
                {
                    "text": "The system needs to manage customer information. The customer database stores name, address, and phone number."
                },
                {
                    "text": "The project goal is to increase sales. The API server is built with RESTful design."
                }
            ]
        },
        "metamodel": {
            "version": "1.0",
            "domain": "business"
        }
    }

    # Create ViewExtractor instance
    print("ViewExtractor test starting...")
    extractor = ViewExtractor(
        model="gpt-5-nano",
        temperature=0.0,
        max_concurrency=2,  # Limit concurrency for testing
        progress=True,
        max_spans_per_label=2  # Limit for testing
    )

    print(f"Target views: {extractor.views}")

    try:
        # Run test
        result = extractor.invoke(sample_input)

        print("\n=== Test Results ===")
        print(f"Output ID: {result['id']}")
        print(f"Extracted Views: {len(result['views'])}")

        for view in result['views']:
            print(f"\nView Type: {view['type']}")
            print(f"Extracted Texts: {len(view['texts'])}")
            for i, text in enumerate(view['texts'][:2]):  # Show only first 2 items
                print(f"  [{i+1}] {text['text']}")

        print("\n✅ ViewExtractor test completed")

    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()

