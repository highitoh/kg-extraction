"""Image extraction from Miro board PDF exports (Frame/Diagram analysis)."""

import asyncio
from typing import Any, Dict, List

from src.extract.image.base_image_analyzer import BaseImageAnalyzer
from src.utils.pdf_page_to_image import convert_pdf_page_to_image


class MiroBoardImageExtractor(BaseImageAnalyzer):
    """Extract and analyze Frame/Diagram images from Miro board PDF.

    This class:
    1. Takes PDF pages matched to Frames/Diagrams (from text similarity)
    2. Converts each matched page to an image
    3. Sends images to LLM for description (inherits from BaseImageAnalyzer)
    4. Returns chunks with Miro-specific metadata

    Note: ALL matched Frames/Diagrams are analyzed (no screening phase).
    """

    def __init__(
        self,
        llm=None,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        max_concurrency: int = 5,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/miro_board_image_extractor",
    ):
        super().__init__(
            llm=llm,
            model=model,
            temperature=temperature,
            max_concurrency=max_concurrency,
            log_dir=log_dir,
        )
        self.progress = progress

    async def _run_pipeline(
        self,
        pdf_path: str,
        item_page_mappings: List[Dict[str, Any]],
        file_id: str,
    ) -> Dict[str, Any]:
        """Process ALL matched Frames/Diagrams (no screening).

        Args:
            pdf_path: Path to PDF file
            item_page_mappings: Matched pages to items (from text similarity)
            file_id: File ID for chunks

        Returns:
            {
                "image_chunks": [...],
                "image_statistics": {...}
            }
        """
        if not item_page_mappings:
            if self.progress:
                print("[MiroBoardImageExtractor] No page-to-item mappings, skipping image analysis")
            return {
                "image_chunks": [],
                "image_statistics": {
                    "total_mappings": 0,
                    "images_processed": 0,
                    "images_generated": 0
                }
            }

        if self.progress:
            print(f"[MiroBoardImageExtractor] Processing {len(item_page_mappings)} matched pages")

        # Convert PDF pages to images and generate descriptions
        semaphore = asyncio.Semaphore(self.max_concurrency)
        counter = 0
        lock = asyncio.Lock()
        total = len(item_page_mappings)

        async def _process_page(mapping: Dict[str, Any], index: int):
            nonlocal counter
            page_number = mapping["page_number"]
            item_id = mapping["item_id"]
            item_type = mapping["item_type"]

            # Convert PDF page to image
            try:
                image = convert_pdf_page_to_image(
                    pdf_path,
                    page_number,
                    max_pixels=2073600  # Full HD
                )
            except Exception as e:
                if self.progress:
                    async with lock:
                        counter += 1
                        print(f"[{counter}/{total}] Page {page_number} → {item_type} {item_id}: failed to convert ({e})")
                return None

            # Generate description (uses base class async method)
            result = await self._generate_description_async(
                image,
                item_id,
                semaphore,
                progress_callback=None
            )

            async with lock:
                counter += 1
                if self.progress:
                    has_structural = result.get("has_structural_content", False)
                    text_len = len(result.get("description", ""))
                    print(f"[{counter}/{total}] Page {page_number} → {item_type} {item_id}: "
                          f"structural={has_structural} text_len={text_len} status={result['status']}")

            # Skip if failed or empty
            if result["status"] not in ("success",):
                return None

            # Build chunk with Miro-specific metadata
            description = result["description"]
            structural_description = result["structural_description"]
            has_structural = result["has_structural_content"]

            # Use structural format as primary text when structural; otherwise natural description
            text_content = structural_description if (has_structural and structural_description) else description

            if not text_content:
                return None

            chunk = {
                "text": text_content,
                "file_id": f"{file_id}:{item_id}",
                "page_number": page_number,
                "provenance": {
                    "source": "image",
                    "item_id": item_id,
                    "item_type": item_type,
                    "has_structural_content": has_structural,
                    "image_description": description,
                    "structural_description": structural_description
                }
            }

            return chunk

        # Process all pages in parallel
        results = await asyncio.gather(
            *[_process_page(mapping, i) for i, mapping in enumerate(item_page_mappings)]
        )

        image_chunks = [r for r in results if r is not None]

        if self.progress:
            print(f"[MiroBoardImageExtractor] Generated {len(image_chunks)} image chunks")

        return {
            "image_chunks": image_chunks,
            "image_statistics": {
                "total_mappings": len(item_page_mappings),
                "images_processed": len(item_page_mappings),
                "images_generated": len(image_chunks)
            }
        }

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Extract image descriptions from Miro board PDF pages.

        Args:
            input: {
                "pdf_path": str,
                "item_page_mappings": List[Dict],  # From text similarity matching
                "file_id": str,
            }

        Returns:
            {
                "image_chunks": [...],
                "image_statistics": {...}
            }
        """
        pdf_path = input["pdf_path"]
        item_page_mappings = input["item_page_mappings"]
        file_id = input["file_id"]

        result = asyncio.run(self._run_pipeline(
            pdf_path,
            item_page_mappings,
            file_id,
        ))

        # Log results
        if self.logger:
            self.logger.save_log(
                {"image_chunks": result["image_chunks"]},
                filename_prefix="miro_board_image_extractor_output_"
            )

        return result
