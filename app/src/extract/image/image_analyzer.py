"""LLM-based image analysis shared by every image source.

Images reach this module as ImageSubject records; where they came from (PDF
XObjects, rendered PDF pages, ...) is the source's concern, not the analyzer's.

Pipeline:
  Stage 1 (optional, LLM): relevance screening, skipped when relevance_threshold is None
  Stage 2 (LLM, parallel): description generation
"""

import asyncio
import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image

from src.utils.llm_factory import LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger
from src.utils.mermaid_parser import is_structural_diagram


def build_chunk_text(structural: str, has_structural: bool) -> str:
    """Build the chunk text that downstream extraction reads from an image.

    Only the structural (Mermaid) form is returned. The prose description the
    vision model writes is a summary it composed, not wording the document
    carries, so a class extracted from it traces back to generated text rather
    than to the source. It is therefore dropped rather than carried downstream.

    A non-structural image yields no text, and so no chunk: prose was the only
    thing such an image contributed.
    """
    structural = (structural or "").strip()
    if has_structural and structural and is_structural_diagram(structural):
        return structural
    return ""


@dataclass
class ImageSubject:
    """One image plus the static metadata its source knows about it."""

    load: Callable[[], Image.Image]  # Deferred so screened-out images are never decoded
    file_id: str
    page_number: Optional[int] = None
    source: Dict[str, Any] = field(default_factory=dict)


class ImageAnalyzer:
    """Turn images into chunks by describing them with a vision LLM.

    Screening is a parameter, not a subclass: pass relevance_threshold to drop
    low-value images before the (more expensive) description call, or leave it
    as None to describe every subject.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_concurrency: int = 5,
        relevance_threshold: float | None = None,
        screen_width: int | None = None,
        describe_width: int | None = None,
        label: str = "ImageAnalyzer",
        verbose: bool = False,
        progress: bool = True,
        log_dir: str | None = None,
        log_prefix: str = "image_analyzer_output_",
    ):
        self.model = model
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self.relevance_threshold = relevance_threshold
        self.screen_width = screen_width
        self.describe_width = describe_width
        self.label = label
        self.verbose = verbose
        self.progress = progress
        self.log_prefix = log_prefix

        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )

        # Resolve app_dir (go up 4 levels from src/extract/image/)
        self.app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        self.description_prompt = self._load_prompt("image_description_extractor.txt")
        self.llm_description = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "image_description_extractor",
                    "schema": self._load_schema("image_description_extractor.schema.json"),
                    "strict": True,
                },
            }
        )

        if self.relevance_threshold is not None:
            suffix = "_verbose" if verbose else ""
            self.relevance_prompt = self._load_prompt(f"image_relevance_filter{suffix}.txt")
            self.llm_relevance = self.llm.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "image_relevance_filter",
                        "schema": self._load_schema(f"image_relevance_filter{suffix}.schema.json"),
                        "strict": True,
                    },
                }
            )

        self.logger = Logger(log_dir) if log_dir else None

    # ---------- Loading helpers ----------

    def _load_schema(self, filename: str) -> dict:
        path = os.path.join(self.app_dir, "schemas", "image-chain", filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.app_dir, "prompts", filename)
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

    @staticmethod
    def _describe_source(source: Dict[str, Any]) -> str:
        """Human-readable identity for progress logs, derived from the source data."""
        parts = [f"{key}={value}" for key, value in source.items() if key != "kind"]
        return " ".join(parts) or "-"

    # ---------- Image helpers ----------

    @staticmethod
    def _to_base64(image: Image.Image, max_width: int | None) -> str:
        """Encode as PNG, optionally downscaling to cap token cost."""
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        if max_width and image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _message(self, prompt: str, base64_img: str) -> HumanMessage:
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            ]
        )

    # ---------- Stage 1: relevance screening ----------

    async def _screen(self, subjects: List[ImageSubject]) -> List[Tuple[ImageSubject, float]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        lock = asyncio.Lock()
        total = len(subjects)
        counter = 0

        if self.progress:
            print(f"[{self.label}] Screening {total} images (concurrency={self.max_concurrency})")

        async def _one(subject: ImageSubject) -> Tuple[ImageSubject, float]:
            nonlocal counter
            async with semaphore:
                score = 0.0
                try:
                    base64_img = self._to_base64(subject.load(), self.screen_width)
                except Exception as exc:
                    status = f"failed ({exc})"
                else:
                    # A failed call fails the run rather than dropping the image
                    response = await call_with_retry(
                        lambda: self.llm_relevance.ainvoke([self._message(self.relevance_prompt, base64_img)]),
                        timeout=60.0,
                        label=f"{self.label}.relevance",
                    )
                    try:
                        data = json.loads(self._to_text(response.content))
                        score = float(data.get("relevance_score", 0.0))
                        status = "ok"
                    except Exception as exc:
                        status = f"failed ({exc})"

                async with lock:
                    counter += 1
                    if self.progress:
                        print(
                            f"[{counter}/{total}] {self.label} screening: "
                            f"{self._describe_source(subject.source)} score={score:.2f} status={status}"
                        )
                return subject, score

        return await asyncio.gather(*[_one(s) for s in subjects])

    # ---------- Stage 2: description generation ----------

    async def _describe(self, scored: List[Tuple[ImageSubject, Optional[float]]]) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        lock = asyncio.Lock()
        total = len(scored)
        counter = 0

        if self.progress:
            print(f"[{self.label}] Describing {total} images (concurrency={self.max_concurrency})")

        async def _one(subject: ImageSubject, score: Optional[float]) -> Optional[Dict[str, Any]]:
            nonlocal counter
            chunk = None
            status = "ok"
            text_len = 0
            has_structural = False
            response = None
            async with semaphore:
                try:
                    base64_img = self._to_base64(subject.load(), self.describe_width)
                except Exception as exc:
                    status = f"failed ({exc})"
                else:
                    # A failed call fails the run rather than dropping the image
                    response = await call_with_retry(
                        lambda: self.llm_description.ainvoke([self._message(self.description_prompt, base64_img)]),
                        timeout=90.0,
                        label=f"{self.label}.description",
                    )
            if response is not None:
                try:
                    data = json.loads(self._to_text(response.content))

                    structural = data.get("structural_description", "") or ""
                    has_structural = bool(data.get("has_structural_content", False))
                    text = build_chunk_text(structural, has_structural)

                    if not text:
                        status = "empty"
                    else:
                        text_len = len(text)
                        chunk = self._build_chunk(subject, score, text, structural, has_structural)
                except Exception as exc:
                    status = f"failed ({exc})"

            async with lock:
                counter += 1
                if self.progress:
                    print(
                        f"[{counter}/{total}] {self.label}: {self._describe_source(subject.source)} "
                        f"structural={has_structural} text_len={text_len} status={status}"
                    )
            return chunk

        results = await asyncio.gather(*[_one(subject, score) for subject, score in scored])
        return [r for r in results if r is not None]

    def _build_chunk(
        self,
        subject: ImageSubject,
        score: Optional[float],
        text: str,
        structural: str,
        has_structural: bool,
    ) -> Dict[str, Any]:
        """Assemble a chunk: origin under "source", generated findings under "analysis"."""
        source = dict(subject.source)
        if subject.page_number is not None:
            source.setdefault("page_number", subject.page_number)

        analysis: Dict[str, Any] = {
            "has_structural_content": has_structural,
            "structural_description": structural or None,
        }
        if score is not None:
            analysis["relevance_score"] = score

        return {"text": text, "file_id": subject.file_id, "source": source, "analysis": analysis}

    # ---------- Orchestration ----------

    async def _run(self, subjects: List[ImageSubject]) -> Dict[str, Any]:
        if self.relevance_threshold is None:
            # No screening: describe every subject
            scored: List[Tuple[ImageSubject, Optional[float]]] = [(s, None) for s in subjects]
        else:
            # Screening: score every subject, then keep only those worth the description call
            screened = await self._screen(subjects)
            scored = [(s, score) for s, score in screened if score >= self.relevance_threshold]

        chunks = await self._describe(scored)

        statistics: Dict[str, Any] = {
            "candidates": len(subjects),
            "analyzed": len(scored),
            "chunks_generated": len(chunks),
        }
        if self.relevance_threshold is not None:
            statistics["relevance_threshold"] = self.relevance_threshold

        return {"image_chunks": chunks, "image_statistics": statistics}

    def analyze(self, subjects: List[ImageSubject]) -> Dict[str, Any]:
        """Describe every subject and return chunks plus pipeline statistics."""
        if not subjects:
            if self.progress:
                print(f"[{self.label}] No images to analyze")
            return {
                "image_chunks": [],
                "image_statistics": {"candidates": 0, "analyzed": 0, "chunks_generated": 0},
            }

        result = asyncio.run(self._run(subjects))

        if self.logger:
            self.logger.save_log({"image_chunks": result["image_chunks"]}, filename_prefix=self.log_prefix)

        if self.progress:
            stats = result["image_statistics"]
            print(
                f"[{self.label}] done: candidates={stats['candidates']} "
                f"analyzed={stats['analyzed']} chunks_generated={stats['chunks_generated']}"
            )

        return result
