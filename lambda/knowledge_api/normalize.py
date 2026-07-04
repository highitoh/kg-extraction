"""Class name normalization for knowledge graph keys.

DynamoDB identity keys (Classes.className, the class-name segments of
Properties.projectSourceClass / relationship / projectTargetClass, and
Instances.entityKey) store the normalized form. The original spelling is
preserved in the displayName attribute. Write and read paths must apply the
same normalization (batch_runner.py mirrors this rule when writing items).
"""

import re

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_class_name(name: str) -> str:
    """Normalize a class name into its DynamoDB key form.

    Rules:
    - "#" is removed because it is the composite-key delimiter
    - whitespace runs collapse to a single space, ends trimmed
    - casefolded so identity is case-insensitive
    """
    without_delimiter = name.replace("#", " ")
    collapsed = _WHITESPACE_RUN.sub(" ", without_delimiter).strip()
    return collapsed.casefold()


def class_entity_key(project: str, class_name: str) -> str:
    """Build the Instances table entityKey for a class."""
    return f"{project}#CLASS#{normalize_class_name(class_name)}"


def property_entity_key(
    project: str, source_class: str, property_name: str, target_class: str
) -> str:
    """Build the Instances table entityKey for a property."""
    source = normalize_class_name(source_class)
    target = normalize_class_name(target_class)
    return f"{project}#PROP#{source}#{property_name}#{target}"
