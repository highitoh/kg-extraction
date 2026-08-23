"""Image sources for ImageAnalyzer.

Each function turns one origin (PDF XObjects, rendered PDF pages, ...) into
ImageSubject records. Anything origin-specific -- rule-based filtering, page
rasterisation, provenance fields -- belongs here rather than in the analyzer.
"""

import io
import uuid
from typing import Any, Dict, List

import pdfplumber
from PIL import Image

from src.extract.image.image_analyzer import ImageSubject
from src.utils.image_utils import extract_image_bytes, passes_basic_filters
from src.utils.pdf_page_to_image import convert_pdf_page_to_image


def pdf_embedded_images(file_path: str, file_id: str, progress: bool = True) -> List[ImageSubject]:
    """Collect images embedded in a PDF, skipping tiny and decorative ones.

    Args:
        file_path: Path to the PDF
        file_id: File ID carried into every chunk
        progress: Print a summary of what was filtered out

    Returns:
        One ImageSubject per image that passed the rule-based filters
    """
    subjects: List[ImageSubject] = []
    total_seen = 0
    rule_filtered = 0
    byte_extraction_failed = 0

    if progress:
        print(f"[pdf_embedded_images] Scanning {file_path}")

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            for image_dict in page.images or []:
                total_seen += 1
                if not passes_basic_filters(image_dict):
                    rule_filtered += 1
                    continue
                image_bytes = extract_image_bytes(page, image_dict)
                if not image_bytes:
                    byte_extraction_failed += 1
                    continue

                image_id = f"image-{uuid.uuid4()}"
                subjects.append(
                    ImageSubject(
                        load=_bytes_loader(image_bytes),
                        file_id=file_id,
                        page_number=page_number,
                        source={"kind": "image", "page_number": page_number, "image_id": image_id},
                    )
                )

    if progress:
        print(
            f"[pdf_embedded_images] pages={page_count} total_images={total_seen} "
            f"rule_filtered={rule_filtered} byte_extraction_failed={byte_extraction_failed} "
            f"candidates={len(subjects)}"
        )
    return subjects


def pdf_page_images(
    pdf_path: str,
    item_page_mappings: List[Dict[str, Any]],
    file_id: str,
    max_pixels: int = 2073600,
) -> List[ImageSubject]:
    """Rasterise the PDF pages that were matched to Miro Frames/Diagrams.

    Args:
        pdf_path: Path to the PDF exported from Miro
        item_page_mappings: Page-to-item matches, each {"page_number", "item_id", "item_type"}
        file_id: File ID carried into every chunk
        max_pixels: Pixel cap per rendered page

    Returns:
        One ImageSubject per mapped page
    """
    subjects: List[ImageSubject] = []

    for mapping in item_page_mappings:
        page_number = mapping["page_number"]
        item_id = mapping["item_id"]
        subjects.append(
            ImageSubject(
                load=_page_loader(pdf_path, page_number, max_pixels),
                file_id=f"{file_id}:{item_id}",
                page_number=page_number,
                source={
                    "kind": "image",
                    "page_number": page_number,
                    "item_id": item_id,
                    "item_type": mapping["item_type"],
                },
            )
        )

    return subjects


def _bytes_loader(image_bytes: bytes):
    def load() -> Image.Image:
        return Image.open(io.BytesIO(image_bytes))

    return load


def _page_loader(pdf_path: str, page_number: int, max_pixels: int):
    def load() -> Image.Image:
        return convert_pdf_page_to_image(pdf_path, page_number, max_pixels=max_pixels)

    return load
