"""Build the inputs the LLM extraction chains consume for a changeset.

The class/property chains take the same shapes as in the document pipeline
(``view_info`` with text spans, ``class_info`` with class instances), so
producing those shapes is all it takes to run them against the changeset
metamodel unmodified.

The substantial part is choosing which text to hand over. Drivers are
extracted while the source text of the extraction is still available, but
rather than passing the whole document the context is narrowed to the
passages the changed knowledge came from, then widened to their surroundings
so that sentences stating *why* something changed are included even when they
did not themselves become knowledge.
"""

from typing import Any, Dict, List, Set


def build_change_context(
    diff: Dict[str, Any],
    sentences: List[Dict[str, Any]],
    file_id: str = "unknown",
    window: int = 1,
) -> List[Dict[str, str]]:
    """Build the text spans that explain a knowledge change.

    Args:
        diff: change_diff_engine.compute_diff() result
        sentences: Sentences of the extraction that produced the later snapshot
            ({"text", "section"} entries)
        file_id: Provenance ID recorded on every selected span
        window: How many neighbouring sentences to keep around a match

    Returns:
        List of {"file_id", "text"} spans in document order
    """
    excerpts = _changed_excerpts(diff)
    if not excerpts or not sentences:
        return []

    matched = _matching_indices(sentences, excerpts)
    if not matched:
        return []

    selected = _widen(matched, sentences, window)
    return [{"file_id": file_id, "text": sentences[i]["text"]} for i in sorted(selected) if sentences[i].get("text")]


def as_view_info(texts: List[Dict[str, str]], view_types: List[str]) -> Dict[str, Any]:
    """Present the text spans as the view_info the class chain expects.

    The view-extraction stage is bypassed: every LLM view type declared by the
    metamodel receives the full text list, so newly added metamodel classes
    are extracted without code changes.

    Args:
        texts: Text spans from build_change_context()
        view_types: View types derived from the changeset metamodel

    Returns:
        {"views": [{"type": ..., "texts": [...]}, ...]}
    """
    return {"views": [{"type": view_type, "texts": texts} for view_type in view_types]}


def as_class_info(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Present the instances as the class_info the property chain expects.

    Every instance is marked ``_is_new`` so the property extractor considers
    all pairs (it skips pairs without at least one new instance).

    Args:
        instances: Instance-graph class entries (derived + LLM-extracted)

    Returns:
        {"classes": [...]}
    """
    return {"classes": [{**inst, "_is_new": True} for inst in instances]}


def _changed_excerpts(diff: Dict[str, Any]) -> List[str]:
    """Collect the source excerpts of the objects this extraction changed.

    Only added and updated classes are considered: their excerpts come from
    the text being extracted now. Deleted objects were justified by an earlier
    document, so their excerpts would never match.
    """
    changed = list(diff.get("added_classes", [])) + [u["after"] for u in diff.get("updated_classes", [])]

    excerpts = []
    for cls in changed:
        for source in cls.get("sources", []) or []:
            normalized = _normalize(source)
            if normalized:
                excerpts.append(normalized)
    return excerpts


def _matching_indices(sentences: List[Dict[str, Any]], excerpts: List[str]) -> Set[int]:
    """Find the sentences an excerpt was taken from.

    Containment is checked both ways: an excerpt may span several sentences,
    or a sentence may hold more text than the excerpt kept.
    """
    matched = set()
    for index, sentence in enumerate(sentences):
        text = _normalize(sentence.get("text", ""))
        if not text:
            continue
        if any(text in excerpt or excerpt in text for excerpt in excerpts):
            matched.add(index)
    return matched


def _widen(matched: Set[int], sentences: List[Dict[str, Any]], window: int) -> Set[int]:
    """Widen the matches to their surrounding context.

    This is what brings in sentences that give the reason for a change without
    being knowledge themselves. A section is the passage a reader would read
    together, so it is kept whole; the positional window only applies to
    matches outside any section, and never crosses a section boundary.
    """
    sections = {sentences[i].get("section") for i in matched if sentences[i].get("section")}

    selected = set(matched)
    for index, sentence in enumerate(sentences):
        if sentence.get("section") and sentence["section"] in sections:
            selected.add(index)

    for index in matched:
        if sentences[index].get("section"):
            continue  # Already covered by its section
        for offset in range(-window, window + 1):
            neighbour = index + offset
            if 0 <= neighbour < len(sentences) and not sentences[neighbour].get("section"):
                selected.add(neighbour)

    return selected


def _normalize(text: str) -> str:
    """Collapse whitespace so excerpts and sentences compare reliably."""
    return " ".join(str(text or "").split())
