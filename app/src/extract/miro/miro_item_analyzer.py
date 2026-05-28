"""Analyzer for transforming Miro board items into structured text chunks."""

from typing import Optional


class MiroItemAnalyzer:
    """Transform raw Miro board data into structured text chunks for extraction.

    Supported Miro Item Types (API v2):
        - sticky_note: Extract text content
        - card: Extract title + description
        - shape: Extract text label if present
        - text: Extract text content
        - connector: Extract structure (start/end, label) - for future relationship inference
        - frame: Extract title as context
        - document: Extract title + plain text content
        - diagram: Extract embedded text/labels from Smart Diagram
        - document_format: Extract title and text content from Miro Doc
        - data_table_format: Extract table content from Miro Table

    Skipped Types:
        - image: Skip initially (future: OCR)
        - app_card, embed: Skip (third-party/external content)
    """

    SUPPORTED_ITEM_TYPES = {
        "sticky_note",
        "card",
        "shape",
        "text",
        "connector",
        "frame",
        "document",
        "diagram",
        "document_format",
        "data_table_format",
    }

    def analyze(self, raw_board_data: dict) -> list[dict]:
        """Transform Miro board items into structured chunks.

        Args:
            raw_board_data: Raw board data from MiroClient (includes "items" array)

        Returns:
            List of item dictionaries:
            {
                "item_id": str,
                "item_type": str,
                "content": str,
                "position": {"x": float, "y": float},
                "metadata": dict (optional)
            }
        """
        items = raw_board_data.get("items", [])
        analyzed_items = []

        for item in items:
            item_type = item.get("type")
            if item_type not in self.SUPPORTED_ITEM_TYPES:
                continue

            analyzed_item = self._analyze_item(item)
            if analyzed_item and analyzed_item.get("content"):
                analyzed_items.append(analyzed_item)

        return analyzed_items

    def _analyze_item(self, item: dict) -> Optional[dict]:
        """Analyze a single Miro item.

        Args:
            item: Raw item data from Miro API

        Returns:
            Structured item dict or None if no content
        """
        item_id = item.get("id")
        item_type = item.get("type")
        data = item.get("data", {})
        position = item.get("position", {})

        content = None

        if item_type == "sticky_note":
            content = data.get("content", "").strip()

        elif item_type == "card":
            title = data.get("title", "").strip()
            description = data.get("description", "").strip()
            content = f"{title}\n{description}".strip() if title or description else None

        elif item_type == "shape":
            content = data.get("content", "").strip()

        elif item_type == "text":
            content = data.get("content", "").strip()

        elif item_type == "frame":
            content = data.get("title", "").strip()

        elif item_type == "document":
            title = data.get("title", "").strip()
            document_url = data.get("documentUrl", "")
            # For now, just extract title. Future: fetch document content
            content = title

        elif item_type == "diagram":
            # Extract any embedded text/labels from diagram
            content = data.get("content", "").strip() or data.get("title", "").strip()

        elif item_type == "document_format":
            # Extract title and any accessible text content from Miro Doc
            title = data.get("title", "").strip()
            # Placeholder: API may provide doc content in future
            content = title

        elif item_type == "data_table_format":
            # Extract table content if available in API response
            # Placeholder: May need to iterate cells if API provides structured data
            content = data.get("title", "").strip()

        elif item_type == "connector":
            # Store connector structure for future relationship inference
            start_item = item.get("startItem", {}).get("id")
            end_item = item.get("endItem", {}).get("id")
            label = data.get("shape", {}).get("text", "").strip()

            if start_item and end_item:
                return {
                    "item_id": item_id,
                    "item_type": "connector",
                    "content": label or "",
                    "position": position,
                    "metadata": {
                        "start_item": start_item,
                        "end_item": end_item,
                    },
                }

        if not content:
            return None

        return {
            "item_id": item_id,
            "item_type": item_type,
            "content": content,
            "position": position,
        }

    def to_text_chunks(self, analyzed_items: list[dict]) -> list[dict]:
        """Convert analyzed items to text chunks for LLM extraction.

        Args:
            analyzed_items: Output from analyze()

        Returns:
            List of text chunks compatible with existing extraction pipeline:
            {
                "text": str,
                "metadata": {
                    "item_id": str,
                    "item_type": str,
                    "position": dict
                }
            }
        """
        chunks = []

        for item in analyzed_items:
            # Skip connectors in text extraction (they're metadata for relationships)
            if item["item_type"] == "connector":
                continue

            chunk = {
                "text": item["content"],
                "metadata": {
                    "item_id": item["item_id"],
                    "item_type": item["item_type"],
                    "position": item["position"],
                },
            }
            chunks.append(chunk)

        return chunks
