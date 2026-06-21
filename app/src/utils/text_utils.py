"""Text cleaning and normalization utilities."""

import html
import re

from charset_normalizer import from_path


def clean_html_text(text: str) -> str:
    """Strip HTML markup from rich-text content.

    Rich-text fields (e.g. from Miro) contain HTML such as <p>, <ol>, <li>,
    <strong> along with HTML entities and zero-width spaces. This converts
    block-level tags to spaces so adjacent items don't merge, removes the
    remaining tags, unescapes entities, and collapses whitespace.

    Args:
        text: Raw HTML text

    Returns:
        Plain text with markup removed
    """
    if not text:
        return ""

    # Convert block-level boundaries to spaces to preserve word separation
    text = re.sub(r"</(p|li|ol|ul|div|br|h[1-6])\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Unescape HTML entities (e.g. &#xff08; -> （, &amp; -> &)
    text = html.unescape(text)

    # Remove zero-width spaces and collapse whitespace
    text = text.replace("​", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_file_encoding(file_path: str) -> str:
    """
    Detect file encoding using charset-normalizer

    Args:
        file_path: Path to the file to analyze

    Returns:
        Detected encoding name (e.g., 'utf-8', 'shift-jis', 'iso-8859-1')
        or 'utf-8' as fallback if detection fails
    """
    try:
        result = from_path(file_path).best()
        if result and result.encoding:
            return result.encoding
    except Exception:
        pass
    return "utf-8"
