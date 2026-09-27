"""Text extraction from Miro boards (parallel to PDF text extraction)."""

import re
from datetime import datetime, timedelta, timezone

from src.extract.miro.miro_board_renderer import MiroBoardRenderer, attach_container_children, group_items_by_region
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


def _filter_board_as_of(items: list[dict], connectors: list[dict], as_of: str) -> tuple[list[dict], list[dict]]:
    """Keep what had been created on the board by 23:59:59 UTC on the as_of date.

    The API keeps no history, so kept items carry their current content and position.

    Args:
        items: Raw Miro API items, with container children already attached
        connectors: Raw Miro API connectors
        as_of: Date in YYYYMMDD format

    Returns:
        (items, connectors) that existed at the cutoff
    """
    try:
        cutoff = datetime.strptime(as_of, "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {as_of}. Expected YYYYMMDD (e.g., 20230115)") from e

    undated = 0

    def existed(obj: dict) -> bool:
        nonlocal undated
        created_at = obj.get("createdAt")
        if not created_at:
            undated += 1
            return True
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return created < cutoff

    by_id = {item.get("id"): item for item in items}
    kept = [item for item in items if existed(item)]
    dropped_ids = set(by_id) - {item.get("id") for item in kept}

    # A child's position is relative to its parent, so resolve it before the parent goes
    renderer = MiroBoardRenderer()
    for item in kept:
        if (item.get("parent") or {}).get("id") in dropped_ids:
            left, top = renderer._absolute_topleft(item, by_id)
            geometry = item.get("geometry") or {}
            item["position"] = {
                **(item.get("position") or {}),
                "x": left + float(geometry.get("width") or 0.0) / 2,
                "y": top + float(geometry.get("height") or 0.0) / 2,
                "relativeTo": "canvas_center",
            }
            item["parent"] = None

    kept_connectors = [
        connector
        for connector in connectors
        if existed(connector)
        and not {(connector.get("startItem") or {}).get("id"), (connector.get("endItem") or {}).get("id")} & dropped_ids
    ]

    print(
        f"[MiroExtractor] As of {as_of} 23:59:59 UTC: kept {len(kept)}/{len(items)} items, "
        f"{len(kept_connectors)}/{len(connectors)} connectors"
    )

    return kept, kept_connectors


def extract_text_from_miro_board(miro_url: str, as_of: str | None = None) -> list[dict]:
    """Extract one text entry per item from a Miro board.

    This function is parallel to extract_text_from_document() for PDFs. The
    entries are sentence-level; they are grouped by region and split into chunks
    further down the pipeline.

    Args:
        miro_url: Miro board URL (e.g., "https://miro.com/app/board/o9J_abc123")
        as_of: Optional YYYYMMDD; only items created by 23:59:59 UTC that day are used

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

    if as_of:
        board_items, connectors = _filter_board_as_of(board_items, connectors, as_of)

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
