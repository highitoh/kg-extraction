"""Text extraction from Miro boards with PDF page matching."""

import re
from typing import Any, Dict, List

import pdfplumber

from src.extract.miro.miro_item_analyzer import MiroItemAnalyzer
from src.utils.miro_client import MiroClient
from src.utils.text_similarity import match_pages_to_items


def _process_visual_item(item: Dict[str, Any], item_type: str, parent_to_children: Dict[str, List], analyzer: MiroItemAnalyzer) -> tuple:
    """Process a Frame or Diagram item and extract text from its children.

    Args:
        item: Miro item (Frame or Diagram)
        item_type: "frame" or "diagram"
        parent_to_children: Map of parent IDs to child items
        analyzer: MiroItemAnalyzer instance

    Returns:
        Tuple of (visual_item_dict, analyzed_children_list)
    """
    item_id = item["id"]
    children = parent_to_children.get(item_id, [])

    # Analyze child items
    analyzed_children = analyzer.analyze({"items": children})
    child_text = " ".join([c["content"] for c in analyzed_children if c.get("content")])

    # If no children, use item's own text (common for Diagrams)
    if not child_text:
        item_analyzed = analyzer.analyze({"items": [item]})
        if item_analyzed:
            child_text = item_analyzed[0].get("content", "")

    # Count child item types
    child_item_types = {}
    for child in children:
        child_type = child.get("type")
        if child_type:
            child_item_types[child_type] = child_item_types.get(child_type, 0) + 1

    visual_item = {
        "item_id": item_id,
        "item_type": item_type,
        "child_items": children,
        "item_text": child_text,
        "child_item_types": child_item_types
    }

    return visual_item, analyzed_children


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
        f"Invalid Miro URL format: {miro_url}. "
        "Expected format: https://miro.com/app/board/BOARD_ID or miro:BOARD_ID"
    )


def extract_text_from_miro_board(pdf_path: str, miro_board_url: str) -> Dict[str, Any]:
    """Extract text from Miro board (PDF + API) and match pages to Frames/Diagrams.

    This function combines two data sources:
    - Miro API: Text content from all items, parent-child relationships
    - PDF file: Page text for matching to Frames/Diagrams

    Args:
        pdf_path: Path to PDF file (manually exported from Miro UI)
        miro_board_url: Miro board URL (e.g., "https://miro.com/app/board/o9J_abc123")

    Returns:
        {
            "sentences": [...],  # Text chunks from Miro API (for knowledge extraction)
            "miro_visual_items": [...],  # Frame + Diagram items for image analysis
            "item_page_mappings": [...],  # Page matches (page_num → item_id, item_type)
            "miro_board_id": str,
            "miro_raw_data": {...}  # For snapshot storage
        }

    Note:
        - sentences come from Miro API, NOT PDF
        - PDF text is used only for matching pages to Frames/Diagrams
        - Visual items include ONLY Frame and Diagram types (not Document/Table)
    """
    # 1. Parse board ID from URL
    board_id = parse_board_id_from_url(miro_board_url)
    print(f"Extracting from Miro board: {board_id}")

    # 2. Fetch Miro API data
    client = MiroClient()
    board_data = client.get_board(board_id)
    print(f"Board name: {board_data.get('name', 'Unnamed')}")

    # Fetch items by type to reduce API calls
    all_items = client.get_board_items(board_id)
    print(f"Total items fetched: {len(all_items)}")

    # 3. Separate items by type
    frames = [item for item in all_items if item.get("type") == "frame"]
    diagrams = [item for item in all_items if item.get("type") == "diagram"]
    documents = [item for item in all_items if item.get("type") == "document_format"]
    tables = [item for item in all_items if item.get("type") == "data_table_format"]

    print(f"  - Frames: {len(frames)}")
    print(f"  - Diagrams: {len(diagrams)}")
    print(f"  - Documents: {len(documents)}")
    print(f"  - Tables: {len(tables)}")

    # 4. Build parent-to-children map
    parent_to_children = {}
    for item in all_items:
        parent = item.get("parent")
        if parent and parent.get("id"):
            parent_id = parent["id"]
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(item)

    # 5. Extract text from all relevant items
    analyzer = MiroItemAnalyzer()

    # Process Frames and Diagrams using common logic
    visual_items = []  # For image analysis (Frames + Diagrams only)
    text_chunks = []  # Text chunks (one per Frame/Diagram)

    for item, item_type in [(f, "frame") for f in frames] + [(d, "diagram") for d in diagrams]:
        visual_item, analyzed_children = _process_visual_item(item, item_type, parent_to_children, analyzer)
        visual_items.append(visual_item)

        # Create a single chunk for all items within this Frame/Diagram
        if visual_item["item_text"]:  # Only create chunk if there's text
            text_chunks.append({
                "text": visual_item["item_text"],
                "file_id": f"miro_board:{board_id}:{visual_item['item_id']}",
                "metadata": {
                    "item_id": visual_item["item_id"],
                    "item_type": visual_item["item_type"],
                    "child_item_types": visual_item["child_item_types"]
                }
            })

    # Process standalone Documents and Tables (text extraction only, NO image analysis)
    for item in documents + tables:
        analyzed_items = analyzer.analyze({"items": [item]})
        for analyzed in analyzed_items:
            if analyzed.get("content"):
                text_chunks.append({
                    "text": analyzed["content"],
                    "file_id": f"miro_board:{board_id}:{analyzed['item_id']}",
                    "metadata": {
                        "item_id": analyzed["item_id"],
                        "item_type": analyzed["item_type"],
                        "position": analyzed.get("position")
                    }
                })

    print(f"Visual items (Frames + Diagrams): {len(visual_items)}")
    print(f"Text chunks created: {len(text_chunks)}")

    # 6. Prepare sentences for knowledge extraction
    sentences = [{"text": chunk["text"], "file_id": chunk["file_id"]} for chunk in text_chunks]

    # 7. Extract PDF page text for matching
    print(f"Extracting PDF page text from: {pdf_path}")
    pdf_page_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pdf_page_texts.append(page_text)
    except Exception as e:
        print(f"Warning: Failed to extract PDF text: {e}")
        pdf_page_texts = []

    print(f"PDF pages: {len(pdf_page_texts)}")

    # 8. Match PDF pages to Frames/Diagrams via text similarity
    item_page_mappings = []
    if pdf_page_texts and visual_items:
        item_page_mappings = match_pages_to_items(
            pdf_page_texts,
            visual_items,
            threshold=0.9
        )
        print(f"Matched {len(item_page_mappings)} pages to Frames/Diagrams")

    # 9. Prepare return data
    return {
        "sentences": sentences,
        "miro_visual_items": visual_items,
        "item_page_mappings": item_page_mappings,
        "miro_board_id": board_id,
        "miro_raw_data": {
            "id": board_id,
            "name": board_data.get("name", "Unnamed Board"),
            "items": all_items,
            "modifiedAt": board_data.get("modifiedAt"),
        }
    }
