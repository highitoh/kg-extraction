"""
Knowledge Filtering Utility

Provides filtering logic for accumulated knowledge.
Extracts only ACCEPT-judged classes and properties with strength > 0.
"""

from typing import List, Dict, Any, Tuple


def filter_accumulated_knowledge(
    classes: List[Dict[str, Any]],
    properties: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter valid classes and properties as accumulated knowledge.

    Filtering conditions:
    - Class: judgment == "ACCEPT"
    - Property: strength > 0

    Args:
        classes: List of classes (ClassChainOutput.classes format)
        properties: List of properties (PropertyChainOutput.properties format)

    Returns:
        Tuple of (filtered_classes, filtered_properties)

    Examples:
        >>> classes = [
        ...     {"id": "c1", "label": "Class1", "judgment": "ACCEPT"},
        ...     {"id": "c2", "label": "Class2", "judgment": "REJECT"},
        ... ]
        >>> properties = [
        ...     {"id": "p1", "strength": 2},
        ...     {"id": "p2", "strength": 0},
        ... ]
        >>> filtered_classes, filtered_props = filter_accumulated_knowledge(classes, properties)
        >>> len(filtered_classes)
        1
        >>> len(filtered_props)
        1
    """
    # Extract only ACCEPT-judged classes
    filtered_classes = [
        cls for cls in classes
        if cls.get("judgment") == "ACCEPT"
    ]

    # Extract only properties with strength greater than 0
    filtered_properties = [
        prop for prop in properties
        if prop.get("strength", 0) > 0
    ]

    return filtered_classes, filtered_properties
