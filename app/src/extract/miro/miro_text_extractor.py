"""Text extraction from Miro boards (parallel to PDF text extraction)."""

import re

from src.extract.miro.miro_board_renderer import attach_container_children, group_items_by_region
from src.extract.miro.miro_item_analyzer import MiroItemAnalyzer
from src.utils.miro_client import MiroClient


def parse_board_id_from_url(miro_url: str) -> str:
    """Extract board ID from Miro URL.

    Supported formats:
        - https://miro.com/app/board/o9J_abc123
        - https://miro.com/app/board/o9J_abc123/
        - miro:o9J_abc123

    Args:
        miro_url: Miro board URL or identifier

    Returns:
        Board ID (e.g., "o9J_abc123")

    Raises:
        ValueError: If URL format is invalid
    """
    # Handle direct board ID format
    if miro_url.startswith("miro:"):
        return miro_url.replace("miro:", "")

    # Extract from URL
    match = re.search(r"miro\.com/app/board/([^/?]+)", miro_url)
    if match:
        return match.group(1)

    raise ValueError(
        f"Invalid Miro URL format: {miro_url}. Expected format: https://miro.com/app/board/BOARD_ID or miro:BOARD_ID"
    )


def extract_text_from_miro_board(miro_url: str) -> list[dict]:
    """Extract one text entry per item from a Miro board.

    This function is parallel to extract_text_from_document() for PDFs. The
    entries are sentence-level; they are grouped by region and split into chunks
    further down the pipeline.

    Args:
        miro_url: Miro board URL (e.g., "https://miro.com/app/board/o9J_abc123")

    Returns:
        One entry per item, with metadata:
        [
            {
                "text": str,
                "file_id": str,  # "miro:{board_id}:{region_id}"
                "metadata": {
                    "item_id": str,
                    "item_type": str,
                    "region_id": str,
                    "position": {"x": float, "y": float}
                },
                "_miro_raw_data": dict  # Attached to first entry for snapshot storage
            },
            ...
        ]

    Example:
        >>> item_texts = extract_text_from_miro_board("https://miro.com/app/board/o9J_abc123")
        >>> print(len(item_texts))
        15
        >>> print(item_texts[0]["text"])
        "Customer needs faster service"
    """
    # 1. Parse board ID from URL
    board_id = parse_board_id_from_url(miro_url)

    # 2. Fetch via MiroClient
    client = MiroClient()
    board_data = client.get_board(board_id)
    board_items = client.get_board_items(board_id)
    connectors = client.get_connectors(board_id)

    # Diagram children come back with parent=null, so read the relationship
    # back per container - the renderer needs it to group them
    attach_container_children(client, board_id, board_items)

    # 3. Transform via MiroItemAnalyzer
    analyzer = MiroItemAnalyzer()
    analyzed_items = analyzer.analyze({"items": board_items})
    item_texts = analyzer.to_item_texts(analyzed_items)

    if not item_texts:
        # Return an empty entry to maintain flow
        item_texts = [{"text": "", "metadata": {}}]

    # 4. Generate region-specific file_ids
    board_file_id_prefix = f"miro:{board_id}"
    region_of = group_items_by_region(board_items)

    # 5. Attach region-specific file_id to each entry
    for item_text in item_texts:
        item_id = item_text["metadata"].get("item_id", "unknown")
        region_id = region_of.get(item_id, item_id)
        item_text["metadata"]["region_id"] = region_id
        item_text["file_id"] = f"{board_file_id_prefix}:{region_id}"

    # 6. Store raw data for snapshot (attached to first entry)
    item_texts[0]["_miro_raw_data"] = {
        "id": board_id,
        "name": board_data.get("name", "Unnamed Board"),
        "items": board_items,
        "connectors": connectors,
        "modifiedAt": board_data.get("modifiedAt"),
    }

    return item_texts
