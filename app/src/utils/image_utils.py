"""Image utility functions for PDF image extraction and processing."""

import base64
import io
from typing import Any, Dict, Optional

from PIL import Image


def passes_basic_filters(image_dict: Dict[str, Any]) -> bool:
    """Rule-based filter: skip tiny/decorative images by size and aspect ratio."""
    width = image_dict.get("width", 0) or 0
    height = image_dict.get("height", 0) or 0

    if width < 50 or height < 50:
        return False

    if width * height < 10000:
        return False

    if height == 0:
        return False
    aspect_ratio = width / height
    if aspect_ratio > 10 or aspect_ratio < 0.1:
        return False

    return True


def extract_image_bytes(page: Any, image_dict: Dict[str, Any]) -> Optional[bytes]:
    """Extract image bytes from a pdfplumber page by cropping to the image bbox.

    Uses the page.crop().to_image() approach which is robust across image types.
    Returns PNG bytes, or None on failure.
    """
    try:
        x0 = image_dict.get("x0")
        y0 = image_dict.get("top")
        x1 = image_dict.get("x1")
        y1 = image_dict.get("bottom")
        if None in (x0, y0, x1, y1):
            return None

        page_width = page.width
        page_height = page.height
        x0 = max(0, min(x0, page_width))
        x1 = max(0, min(x1, page_width))
        y0 = max(0, min(y0, page_height))
        y1 = max(0, min(y1, page_height))
        if x1 <= x0 or y1 <= y0:
            return None

        cropped = page.crop((x0, y0, x1, y1))
        pil_image = cropped.to_image(resolution=150).original

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


def image_to_base64(image_bytes: bytes, format: str = "PNG") -> str:
    """Convert image bytes to base64 string (re-encoded to ensure valid format)."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(image_bytes).decode("ascii")


def resize_image_base64(image_bytes: bytes, max_width: int = 1024) -> str:
    """Resize image to reduce token cost, return base64 PNG."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")
        if pil_image.width > max_width:
            ratio = max_width / pil_image.width
            new_size = (max_width, int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(image_bytes).decode("ascii")
