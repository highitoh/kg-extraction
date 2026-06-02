"""PDF page to image conversion utility using PyMuPDF."""

import io
import math

import fitz  # PyMuPDF
from PIL import Image


def convert_pdf_page_to_image(
    pdf_path: str,
    page_number: int,
    max_pixels: int = 2073600,  # Full HD equivalent (1920x1080)
) -> Image.Image:
    """Convert a single PDF page to PIL Image with pixel count limit.

    Args:
        pdf_path: Path to PDF file
        page_number: Page number (0-indexed)
        max_pixels: Maximum total pixels (width * height), default: 2,073,600 (Full HD)
                    Maintains aspect ratio while keeping total pixels <= max_pixels

    Returns:
        PIL Image object (width * height <= max_pixels)

    Example:
        >>> img = convert_pdf_page_to_image("board.pdf", 0)
        >>> img.width * img.height <= 2073600
        True
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number)

        # Get original page dimensions (in points)
        page_width = page.rect.width
        page_height = page.rect.height
        original_pixels = page_width * page_height

        # Calculate zoom to achieve target pixel count while maintaining aspect ratio
        if original_pixels > max_pixels:
            zoom = math.sqrt(max_pixels / original_pixels)
        else:
            zoom = 1.0  # Don't upscale if already smaller

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    finally:
        doc.close()
