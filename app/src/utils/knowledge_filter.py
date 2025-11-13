"""
Knowledge Filtering Utility

累積知識のフィルタリングロジックを提供する。
ACCEPT判定のクラスと強度>0のプロパティのみを抽出する。
"""

from typing import List, Dict, Any, Tuple


def filter_accumulated_knowledge(
    classes: List[Dict[str, Any]],
    properties: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    累積知識として有効なクラスとプロパティをフィルタリングする。

    フィルタリング条件:
    - クラス: judgment == "ACCEPT"
    - プロパティ: strength > 0

    Args:
        classes: クラスのリスト (ClassChainOutput.classes形式)
        properties: プロパティのリスト (PropertyChainOutput.properties形式)

    Returns:
        (filtered_classes, filtered_properties) のタプル

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
    # ACCEPT判定のクラスのみを抽出
    filtered_classes = [
        cls for cls in classes
        if cls.get("judgment") == "ACCEPT"
    ]

    # 強度が0より大きいプロパティのみを抽出
    filtered_properties = [
        prop for prop in properties
        if prop.get("strength", 0) > 0
    ]

    return filtered_classes, filtered_properties
