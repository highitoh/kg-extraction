"""Image extraction from PDFs using vision LLM.

3-stage pipeline:
  Stage 1 (rule-based): Filter by size/aspect ratio (NO LLM cost)
  Stage 2 (LLM, parallel): Relevance classification (cheap)
  Stage 3 (LLM, parallel): Description generation for high-relevance images
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import base64
import json
import os
import uuid

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from src.utils.image_utils import (
    extract_image_bytes,
    passes_basic_filters,
    resize_image_base64,
)
from src.utils.llm_factory import LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger


class ImageExtractor(Runnable):
    """Extract knowledge-rich image descriptions from PDF files."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model: str = "gemini-3.1-flash-lite-preview",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 5,
        relevance_threshold: float = 0.5,
        verbose: bool = False,
        log_dir: str = "/workspace/app/log/image_extractor",
    ):
        self.model = model
        self.temperature = temperature
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.relevance_threshold = relevance_threshold
        self.verbose = verbose

        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )

        # Resolve app_dir once (go up 4 levels from src/extract/image/)
        self.app_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )

        # Load prompts and schemas (verbose vs non-verbose)
        self.relevance_prompt = self._load_relevance_prompt()
        self.description_prompt = self._load_description_prompt()

        relevance_schema = (
            self._load_schema("image_relevance_filter_verbose.schema.json")
            if self.verbose
            else self._load_schema("image_relevance_filter.schema.json")
        )
        description_schema = self._load_schema("image_description_extractor.schema.json")

        self.llm_relevance = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "image_relevance_filter",
                    "schema": relevance_schema,
                    "strict": True,
                },
            }
        )
        self.llm_description = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "image_description_extractor",
                    "schema": description_schema,
                    "strict": True,
                },
            }
        )

        self.logger = Logger(log_dir)

    # ---------- Loading helpers ----------

    def _load_schema(self, filename: str) -> dict:
        path = os.path.join(self.app_dir, "schemas", "image-chain", filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_relevance_prompt(self) -> str:
        filename = "image_relevance_filter_verbose.txt" if self.verbose else "image_relevance_filter.txt"
        path = os.path.join(self.app_dir, "prompts", filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_description_prompt(self) -> str:
        path = os.path.join(self.app_dir, "prompts", "image_description_extractor.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ---------- Parsing helpers ----------

    @staticmethod
    def _to_text(content: Union[str, List[Any]]) -> str:
        if isinstance(content, str):
            return content
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    # ---------- Stage 1: Rule-based filtering ----------

    def _discover_candidates(self, file_path: str) -> List[Dict[str, Any]]:
        """Open PDF and apply rule-based filters; return candidate images with bytes."""
        candidates: List[Dict[str, Any]] = []
        total_seen = 0
        rule_filtered = 0
        byte_extraction_failed = 0
        page_count = 0

        if self.progress:
            print(f"[ImageExtractor] Starting Stage 1 (rule-based filter): {file_path}")

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                images = page.images or []
                for image_dict in images:
                    total_seen += 1
                    if not passes_basic_filters(image_dict):
                        rule_filtered += 1
                        continue
                    img_bytes = extract_image_bytes(page, image_dict)
                    if not img_bytes:
                        byte_extraction_failed += 1
                        continue
                    candidates.append({
                        "id": f"image-{uuid.uuid4()}",
                        "page_number": page_num,
                        "width": image_dict.get("width"),
                        "height": image_dict.get("height"),
                        "bytes": img_bytes,
                    })

        if self.progress:
            print(
                f"[ImageExtractor] Stage 1 done: pages={page_count} total_images={total_seen} "
                f"rule_filtered={rule_filtered} byte_extraction_failed={byte_extraction_failed} "
                f"candidates={len(candidates)}"
            )
        return candidates

    # ---------- Stage 2: Relevance classification (parallel) ----------

    async def _classify_relevance_parallel(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(candidates)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ImageExtractor] Starting Stage 2 (relevance): {total} candidates (concurrency={self.max_concurrency})")

        async def _wrapped(i: int, candidate: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal counter
            async with sem:
                base64_img = resize_image_base64(candidate["bytes"], max_width=512)
                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": self.relevance_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        },
                    ]
                )
                try:
                    response = await call_with_retry(
                        lambda: self.llm_relevance.ainvoke([msg]),
                        timeout=60.0,
                        label="ImageExtractor.relevance",
                    )
                    data = json.loads(self._to_text(response.content))
                    score = float(data.get("relevance_score", 0.0))
                    reason = data.get("reason")
                    status = "ok"
                except Exception as exc:
                    score = 0.0
                    reason = None
                    status = f"failed ({exc})"

                result = {**candidate, "relevance_score": score}
                if reason is not None:
                    result["filter_reason"] = reason

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] ImageExtractor Stage 2: image {i} (page={candidate['page_number']}) "
                            f"score={score:.2f} status={status}"
                        )
                return result

        return await asyncio.gather(*[_wrapped(i, c) for i, c in enumerate(candidates)])

    # ---------- Stage 3: Description generation (parallel) ----------

    async def _generate_descriptions_parallel(
        self,
        high_relevance_images: List[Dict[str, Any]],
        file_id: str,
    ) -> List[Dict[str, Any]]:
        if not high_relevance_images:
            return []
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(high_relevance_images)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ImageExtractor] Starting Stage 3 (description): {total} images (concurrency={self.max_concurrency})")

        async def _wrapped(i: int, image_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            nonlocal counter
            async with sem:
                base64_img = resize_image_base64(image_data["bytes"], max_width=1024)
                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": self.description_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        },
                    ]
                )
                chunk: Optional[Dict[str, Any]] = None
                status = "ok"
                text_len = 0
                has_structural = False
                try:
                    response = await call_with_retry(
                        lambda: self.llm_description.ainvoke([msg]),
                        timeout=90.0,
                        label="ImageExtractor.description",
                    )
                    data = json.loads(self._to_text(response.content))

                    description = data.get("description", "") or ""
                    drawio = data.get("drawio_description", "") or ""
                    has_structural = bool(data.get("has_structural_content", False))

                    # Use drawio format as primary text when structural; otherwise the natural description
                    text_content = drawio if (has_structural and drawio) else description

                    if not text_content:
                        status = "empty"
                    else:
                        text_len = len(text_content)
                        chunk = {
                            "text": text_content,
                            "file_id": file_id,
                            "page_number": image_data["page_number"],
                            "provenance": {
                                "source": "image",
                                "image_id": image_data["id"],
                                "relevance_score": image_data["relevance_score"],
                                "has_structural_content": has_structural,
                                "image_description": description,
                                "drawio_description": drawio if drawio else None,
                            },
                        }
                except Exception as exc:
                    status = f"failed ({exc})"

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] ImageExtractor Stage 3: image {i} (page={image_data['page_number']}) "
                            f"structural={has_structural} text_len={text_len} status={status}"
                        )
                return chunk

        results = await asyncio.gather(
            *[_wrapped(i, img) for i, img in enumerate(high_relevance_images)]
        )
        return [r for r in results if r is not None]

    # ---------- Orchestration ----------

    async def _run_pipeline(self, file_path: str, file_id: str) -> Dict[str, Any]:
        candidates = self._discover_candidates(file_path)
        total_found_after_rule = len(candidates)

        classified = await self._classify_relevance_parallel(candidates)
        high_relevance = [
            c for c in classified if c.get("relevance_score", 0.0) >= self.relevance_threshold
        ]

        image_chunks = await self._generate_descriptions_parallel(high_relevance, file_id)

        return {
            "image_chunks": image_chunks,
            "statistics": {
                "after_rule_filter": total_found_after_rule,
                "after_relevance_filter": len(high_relevance),
                "chunks_generated": len(image_chunks),
                "relevance_threshold": self.relevance_threshold,
            },
        }

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Args:
            input: Dict with at least {"files": [{"path": ...}]} and optionally "file_id".

        Returns:
            Dict with "image_chunks" (list) and "image_statistics" fields.
        """
        file_path = input["files"][0]["path"]
        file_id = input.get("file_id") or input.get("target", {}).get("file_id") or str(uuid.uuid4())

        result = asyncio.run(self._run_pipeline(file_path, file_id))

        output: Dict[str, Any] = {
            "image_chunks": result["image_chunks"],
            "image_statistics": result["statistics"],
        }

        self.logger.save_log(
            {"image_chunks": result["image_chunks"]},
            filename_prefix="image_extractor_output_",
        )

        if self.progress:
            stats = result["statistics"]
            print(
                f"[ImageExtractor] done: rule_filtered={stats['after_rule_filter']} "
                f"relevance_filtered={stats['after_relevance_filter']} "
                f"chunks_generated={stats['chunks_generated']}"
            )

        return output


if __name__ == "__main__":
    sample_input = {
        "files": [{"path": "../doc/sample.pdf"}],
        "file_id": "sample_file",
    }

    print("ImageExtractor test starting...")
    extractor = ImageExtractor(
        model="gpt-5-nano",
        temperature=0.0,
        max_concurrency=2,
        relevance_threshold=0.5,
        verbose=True,
    )

    try:
        result = extractor.invoke(sample_input)
        print("\n=== Test Results ===")
        print(f"Image chunks: {len(result['image_chunks'])}")
        print(f"Statistics: {result['image_statistics']}")
        for i, chunk in enumerate(result["image_chunks"][:3]):
            print(f"\n[Chunk {i+1}] page={chunk['page_number']}")
            print(f"  text: {chunk['text'][:200]}...")
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
