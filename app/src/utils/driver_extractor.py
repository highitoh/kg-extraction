"""LLM-based driver extraction from source changes."""

import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.utils.llm_factory import LLMFactory


def cluster_and_extract_drivers(source_changes: dict, kg_changes: dict, model: str = None) -> List[dict]:
    """Use LLM to cluster source changes into business drivers.

    Args:
        source_changes: {new_items, modified_items, deleted_items, source_type}
        kg_changes: {added_classes, updated_classes, deleted_classes, added_properties, ...}
        model: Optional model override (recommended: gemini-3.0-flash-lite)

    Returns:
        List of driver objects:
        [
            {
                "driverId": str,
                "type": str,
                "description": str,
                "category": str,
                "source_items": [str],
                "affected_entities": {"classes": [...], "properties": [...]}
            },
            ...
        ]

    Example:
        >>> drivers = cluster_and_extract_drivers(source_changes, kg_changes)
        >>> print(drivers[0]["type"])
        "New GDPR compliance requirement"
    """
    # Skip if no source changes
    if not source_changes.get("new_items") and not source_changes.get("modified_items"):
        return []

    # Create LLM instance (uses default model or override)
    # Recommended: gemini-3.0-flash-lite for fast inference
    llm = LLMFactory.create_llm(model=model) if model else LLMFactory.create_llm()

    # Load prompt template
    app_dir = Path(__file__).parent.parent.parent
    prompt_path = app_dir / "prompts" / "change_driver_extractor.txt"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Summarize KG changes for prompt
    kg_summary = _summarize_kg_changes(kg_changes)

    # Prepare prompt
    prompt = prompt_template.format(
        source_changes=json.dumps(source_changes, indent=2, ensure_ascii=False), kg_changes_summary=kg_summary
    )

    # Invoke LLM
    response = llm.invoke(prompt)

    # Parse response
    drivers = _parse_driver_response(response)

    return drivers


def _summarize_kg_changes(kg_changes: dict) -> str:
    """Summarize KG changes for LLM prompt.

    Args:
        kg_changes: KG diff result from change_diff_engine

    Returns:
        Human-readable summary string
    """
    lines = []

    added_classes = kg_changes.get("added_classes", [])
    updated_classes = kg_changes.get("updated_classes", [])
    deleted_classes = kg_changes.get("deleted_classes", [])
    added_properties = kg_changes.get("added_properties", [])
    deleted_properties = kg_changes.get("deleted_properties", [])

    if added_classes:
        lines.append(f"Added {len(added_classes)} classes:")
        for cls in added_classes[:10]:  # Limit to 10 for brevity
            lines.append(f"  - {cls.get('label', 'Unknown')} ({cls.get('class_iri', 'Unknown')})")

    if updated_classes:
        lines.append(f"\nUpdated {len(updated_classes)} classes:")
        for cls in updated_classes[:10]:
            lines.append(f"  - {cls.get('label', 'Unknown')}")

    if deleted_classes:
        lines.append(f"\nDeleted {len(deleted_classes)} classes:")
        for cls in deleted_classes[:10]:
            lines.append(f"  - {cls.get('label', 'Unknown')}")

    if added_properties:
        lines.append(f"\nAdded {len(added_properties)} properties:")
        for prop in added_properties[:10]:
            lines.append(
                f"  - {prop.get('src_label', 'Unknown')} → {prop.get('dest_label', 'Unknown')} "
                f"({prop.get('property_iri', 'Unknown')})"
            )

    if deleted_properties:
        lines.append(f"\nDeleted {len(deleted_properties)} properties:")
        for prop in deleted_properties[:10]:
            lines.append(f"  - {prop.get('src_label', 'Unknown')} → {prop.get('dest_label', 'Unknown')}")

    return "\n".join(lines) if lines else "No significant changes detected"


def _parse_driver_response(response) -> List[dict]:
    """Parse LLM response into driver objects.

    Args:
        response: LLM response object

    Returns:
        List of driver dicts
    """
    # Extract text from response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            # Extract text from content blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content = block.get("text", "")
                    break
        response_text = content if isinstance(content, str) else str(content)
    else:
        response_text = str(response)

    try:
        # Parse JSON response
        result = json.loads(response_text)
        drivers = result.get("drivers", [])

        # Validate driver structure
        validated_drivers = []
        for driver in drivers:
            if isinstance(driver, dict) and "type" in driver:
                validated_drivers.append(driver)

        return validated_drivers

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response text: {response_text}")
        return []
