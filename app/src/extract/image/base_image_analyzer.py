"""Base class for LLM-based image analysis (shared by PDF and Miro sources)."""

import asyncio
import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from PIL import Image

from src.utils.llm_factory import LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger


class BaseImageAnalyzer(Runnable):
    """Base class for LLM-based image analysis.

    Provides shared functionality for:
    - Loading prompts and schemas
    - LLM description generation (async, parallel)
    - Image resizing
    - Text parsing from LLM responses

    Subclasses:
    - PdfImageExtractor: Extract and analyze images from PDF XObjects
    - MiroBoardImageExtractor: Analyze PDF pages as images for Miro Frames/Diagrams
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        max_concurrency: int = 5,
        log_dir: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_concurrency = max_concurrency

        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )

        # Resolve app_dir (go up 4 levels from src/extract/image/)
        self.app_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )

        # Load description prompt and schema
        self.description_prompt = self._load_description_prompt()
        description_schema = self._load_schema("image_description_extractor.schema.json")

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

        self.logger = Logger(log_dir) if log_dir else None

    # ========== Loading helpers ==========

    def _load_schema(self, filename: str) -> dict:
        """Load JSON schema from schemas/image-chain/ directory."""
        path = os.path.join(self.app_dir, "schemas", "image-chain", filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_description_prompt(self) -> str:
        """Load image description prompt."""
        path = os.path.join(self.app_dir, "prompts", "image_description_extractor.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ========== Parsing helpers ==========

    @staticmethod
    def _to_text(content: Union[str, List[Any]]) -> str:
        """Extract text from LLM response content."""
        if isinstance(content, str):
            return content
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    # ========== Image utilities ==========

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64-encoded PNG string
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # ========== LLM description generation (async) ==========

    async def _generate_description_async(
        self,
        image: Image.Image,
        image_id: str,
        semaphore: asyncio.Semaphore,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Generate LLM description for a single image (async).

        Common logic used by both PDF and Miro extractors.

        Args:
            image: PIL Image object
            image_id: Unique identifier for this image
            semaphore: Async semaphore for concurrency control
            progress_callback: Optional callback(status, data) for progress reporting

        Returns:
            {
                "image_id": str,
                "description": str,
                "structural_description": str | None,
                "has_structural_content": bool,
                "status": "success" | "failed" | "empty"
            }
        """
        async with semaphore:
            # Convert image to base64 (caller should resize before calling if needed)
            base64_img = self._image_to_base64(image)

            # Send to LLM
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": self.description_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                    },
                ]
            )

            try:
                response = await call_with_retry(
                    lambda: self.llm_description.ainvoke([msg]),
                    timeout=90.0,
                    label="BaseImageAnalyzer.description",
                )
                data = json.loads(self._to_text(response.content))

                description = data.get("description", "") or ""
                structural = data.get("structural_description", "") or ""
                has_structural = bool(data.get("has_structural_content", False))

                if not description:
                    status = "empty"
                else:
                    status = "success"

                result = {
                    "image_id": image_id,
                    "description": description,
                    "structural_description": structural if structural else None,
                    "has_structural_content": has_structural,
                    "status": status
                }

                if progress_callback:
                    progress_callback("success", result)

                return result

            except Exception as exc:
                result = {
                    "image_id": image_id,
                    "description": "",
                    "structural_description": None,
                    "has_structural_content": False,
                    "status": f"failed ({exc})"
                }

                if progress_callback:
                    progress_callback("failed", result)

                return result

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement invoke()")
