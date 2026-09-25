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
    """Extract text chunks from Miro board.

    This function is parallel to extract_text_from_document() for PDFs.

    Args:
        miro_url: Miro board URL (e.g., "https://miro.com/app/board/o9J_abc123")

    Returns:
        List of text chunks with metadata:
        [
            {
                "text": str,
                "metadata": {
                    "item_id": str,
                    "item_type": str,
                    "position": {"x": float, "y": float}
                },
                "_miro_raw_data": dict  # Attached to first chunk for snapshot storage
            },
            ...
        ]

    Example:
        >>> chunks = extract_text_from_miro_board("https://miro.com/app/board/o9J_abc123")
        >>> print(len(chunks))
        15
        >>> print(chunks[0]["text"])
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
    chunks = analyzer.to_text_chunks(analyzed_items)

    if not chunks:
        # Return empty chunk to maintain flow
        chunks = [{"text": "", "metadata": {}}]

    # 4. Generate region-specific file_ids
    board_file_id_prefix = f"miro:{board_id}"
    region_of = group_items_by_region(board_items)

    # 5. Attach region-specific file_id to each chunk
    for chunk in chunks:
        item_id = chunk["metadata"].get("item_id", "unknown")
        region_id = region_of.get(item_id, item_id)
        chunk["metadata"]["region_id"] = region_id
        chunk["file_id"] = f"{board_file_id_prefix}:{region_id}"

    # 6. Store raw data for snapshot (attached to first chunk)
    chunks[0]["_miro_raw_data"] = {
        "id": board_id,
        "name": board_data.get("name", "Unnamed Board"),
        "items": board_items,
        "connectors": connectors,
        "modifiedAt": board_data.get("modifiedAt"),
    }

    return chunks
