"""Parse Mermaid diagrams into format-agnostic structural edge records.

Scoped to the Mermaid subset that the image description LLM actually emits
(flowchart / graph / classDiagram). The parser is edge-pattern-driven: any line
that does not match a known node-token + edge-operator pattern is skipped rather
than raising, so unsupported diagram types degrade gracefully to "no edges".
"""

import re
from typing import Any, Dict, List

# Diagram-type headers and other non-edge lines to skip outright.
_DIRECTIVE_RE = re.compile(
    r"^\s*(flowchart|graph|classDiagram|sequenceDiagram|stateDiagram(-v2)?|"
    r"erDiagram|gantt|pie|journey|subgraph|end|class\s|classDef|style|click|%%)",
    re.IGNORECASE,
)

# Node shapes: ID[label], ID(label), ID((label)), ID{label}.
# Capture the node id and its display label.
_NODE_DECL_RE = re.compile(r"([A-Za-z0-9_\-./]+)\s*(?:\(\(([^)]*)\)\)|\[([^\]]*)\]|\(([^)]*)\)|\{([^}]*)\})")

# Flowchart edge operators (longest first so e.g. '-.->' wins over '-->').
_FLOW_OPS = ["-.->", "-.-", "==>", "===", "-->", "---", "--o", "--x"]

# classDiagram relation operators.
_CLASS_OPS = ["<|--", "--|>", "o--", "--o", "*--", "--*", "<|..", "..|>", "-->", "<--", "..>", "<..", "--", ".."]

# Operators whose arrowhead points left, so src/dest must be swapped.
_REVERSED_OPS = {"<--", "<|--", "<|..", "<..", "..<"}

_NODE_TOKEN = r"[A-Za-z0-9_\-./]+(?:\(\([^)]*\)\)|\[[^\]]*\]|\([^)]*\)|\{[^}]*\})?"


def _clean_label(text: str) -> str:
    """Strip wrapping quotes/backticks and <br> tags, collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _node_label(token: str, label_map: Dict[str, str]) -> str:
    """Resolve a node token to its display label.

    token may be a bare id ('A') or a shaped declaration ('A[label]').
    Prefer an inline label; else the label_map; else the bare id.
    """
    m = _NODE_DECL_RE.match(token.strip())
    if m:
        node_id = m.group(1)
        inline = next((g for g in m.groups()[1:] if g is not None), "")
        inline = _clean_label(inline)
        if inline:
            return inline
        return label_map.get(node_id, node_id)
    node_id = token.strip()
    return label_map.get(node_id, node_id)


def _build_label_map(text: str) -> Dict[str, str]:
    """Map node id -> display label from all shape declarations in the text."""
    label_map: Dict[str, str] = {}
    for m in _NODE_DECL_RE.finditer(text):
        node_id = m.group(1)
        label = next((g for g in m.groups()[1:] if g is not None), "")
        label = _clean_label(label)
        if label:
            label_map[node_id] = label
    return label_map


def _split_statements(mermaid_text: str) -> List[str]:
    """Split into statements on newlines and semicolons."""
    parts: List[str] = []
    for line in mermaid_text.splitlines():
        for stmt in line.split(";"):
            stmt = stmt.strip()
            if stmt:
                parts.append(stmt)
    return parts


def _match_edge(stmt: str, ops: List[str], label_map: Dict[str, str]) -> Dict[str, str] | None:
    """Match a single 'src <op> dest' statement, with optional edge label.

    Supports labeled flowchart edges in both forms:
      A -->|label| B   and   A -- label --> B
    """
    for op in ops:
        # Form: A -->|label| B  (label pipe immediately after the operator)
        piped = re.match(
            rf"^({_NODE_TOKEN})\s*{re.escape(op)}\s*\|([^|]*)\|\s*({_NODE_TOKEN})\s*$",
            stmt,
        )
        if piped:
            left, edge_label, right = piped.group(1), piped.group(2), piped.group(3)
            return _make_edge(left, right, edge_label, op, label_map)

        # Plain: A <op> B  (optionally with trailing ': label' for classDiagram)
        plain = re.match(
            rf"^({_NODE_TOKEN})\s*{re.escape(op)}\s*({_NODE_TOKEN})\s*(?::\s*(.*))?$",
            stmt,
        )
        if plain:
            left, right, edge_label = plain.group(1), plain.group(2), plain.group(3) or ""
            return _make_edge(left, right, edge_label, op, label_map)

    # Form: A -- label --> B (label between dashes and arrow)
    dashed = re.match(
        rf"^({_NODE_TOKEN})\s*--\s*([^>|]+?)\s*-->\s*({_NODE_TOKEN})\s*$",
        stmt,
    )
    if dashed:
        left, edge_label, right = dashed.group(1), dashed.group(2), dashed.group(3)
        return _make_edge(left, right, edge_label, "-->", label_map)

    return None


def _make_edge(left: str, right: str, edge_label: str, op: str, label_map: Dict[str, str]) -> Dict[str, str]:
    """Build an edge record, normalizing direction so arrowhead -> dest."""
    src_token, dest_token = (right, left) if op in _REVERSED_OPS else (left, right)
    return {
        "src_label": _node_label(src_token, label_map),
        "dest_label": _node_label(dest_token, label_map),
        "edge_label": _clean_label(edge_label),
    }


def parse_mermaid_edges(mermaid_text: str) -> List[Dict[str, str]]:
    """Parse Mermaid flowchart/graph/classDiagram text into edge records.

    Args:
        mermaid_text: Raw Mermaid diagram string.

    Returns:
        List of {"src_label", "dest_label", "edge_label"} records (deduplicated).
        Returns [] for empty input or unsupported diagram types.
    """
    if not mermaid_text or not mermaid_text.strip():
        return []

    label_map = _build_label_map(mermaid_text)
    # Try classDiagram ops first when the diagram declares classDiagram, else flowchart ops.
    is_class = bool(re.search(r"classDiagram", mermaid_text, re.IGNORECASE))
    ops = _CLASS_OPS if is_class else _FLOW_OPS

    edges: List[Dict[str, str]] = []
    seen = set()
    for stmt in _split_statements(mermaid_text):
        if _DIRECTIVE_RE.match(stmt):
            continue
        edge = _match_edge(stmt, ops, label_map)
        # Fall back to the other operator set for mixed content.
        if edge is None:
            other = _FLOW_OPS if is_class else _CLASS_OPS
            edge = _match_edge(stmt, other, label_map)
        if edge is None:
            continue
        if not edge["src_label"] or not edge["dest_label"]:
            continue
        key = (edge["src_label"], edge["dest_label"], edge["edge_label"])
        if key in seen:
            continue
        seen.add(key)
        edges.append(edge)

    return edges


def build_extracted_structure(image_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build format-agnostic structure records from structural image chunks.

    Args:
        image_chunks: Image chunks with a "provenance" dict that may carry
            "has_structural_content" and "structural_description" (Mermaid).

    Returns:
        List of {"src_label", "dest_label", "edge_label", "file_id"} records.
    """
    structure: List[Dict[str, str]] = []
    for chunk in image_chunks:
        provenance = chunk.get("provenance", {}) or {}
        if not provenance.get("has_structural_content"):
            continue
        mermaid = provenance.get("structural_description")
        if not mermaid:
            continue
        file_id = chunk.get("file_id", "")
        for edge in parse_mermaid_edges(mermaid):
            structure.append({**edge, "file_id": file_id})
    return structure


if __name__ == "__main__":
    samples = [
        "flowchart TD\n  A[クラス抽出] --> B[クラス集約]\n  B --> C[評価]",
        "classDiagram\n  KGSnapshot o-- KGObject : hasObject\n  KGObject <|-- KGNode",
        "graph TD; G1[G1: 目標] --> Q1[Q1: 問い]; Q1 --> M1[M1: 指標]",
        "flowchart LR\n  A -->|uses| B\n  X -- depends --> Y",
        "flowchart TD\n  B <|-- A",
        "sequenceDiagram\n  Alice->>John: Hello",  # unsupported type -> no edges
        "this is not mermaid at all !!!",
    ]
    for s in samples:
        print("=" * 60)
        print(s.splitlines()[0])
        for e in parse_mermaid_edges(s):
            print(f"  {e['src_label']!r} --[{e['edge_label']}]--> {e['dest_label']!r}")
