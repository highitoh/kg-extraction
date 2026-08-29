"""Normalization rules for extracted class labels.

Every rule in this module must hold to the following contract:

- Labels only. Body text legitimately carries spacing these rules destroy
  (chapter numbers, table columns), so never apply them upstream of extraction.
- Idempotent. Applying a rule twice equals applying it once.
- Latin-safe. Spacing that separates Latin words carries meaning and must survive.

To add a rule: write a `_rule_*` function with an example in its docstring,
then append it to `_RULES` at the position it should run.

Rules that can only detect a defect, not repair it, do not belong here. A label
carrying a control character, for example, is corrupted beyond what a rewrite can
recover, and routing it to review is the appropriate response.
"""

import re

# Hiragana, katakana, CJK ideographs, the iteration mark and the prolonged sound mark.
_JAPANESE = r"[ぁ-んァ-ヶ一-龠々ー]"
_WRAPPED = re.compile(f"({_JAPANESE})[ 　]+({_JAPANESE})")


def _rule_join_wrapped_japanese(label: str) -> str:
    """Join Japanese text that a line wrap split into two words.

    Japanese is written without word separators, so a space between two Japanese
    characters is layout, not meaning. A space next to a Latin character is left
    alone because there it separates words.

    Example: "オープンな標準的技 術" -> "オープンな標準的技術"
             "接触確認 API による接続" -> unchanged
    """
    return _WRAPPED.sub(r"\1\2", label)


def _rule_collapse_whitespace(label: str) -> str:
    """Collapse runs of whitespace into one space and trim the ends.

    Example: "  接触確認  アプリ " -> "接触確認 アプリ"
    """
    return " ".join(label.split())


_RULES = (
    # Join before collapsing: the join rule consumes runs of spacing itself.
    _rule_join_wrapped_japanese,
    _rule_collapse_whitespace,
)


def normalize_label(label: str) -> str:
    """Apply every label rule in order.

    Args:
        label: Raw label as produced by an LLM or a document parser

    Returns:
        Normalized label
    """
    if not label:
        return ""

    for rule in _RULES:
        label = rule(label)
    return label
