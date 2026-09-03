"""Parse Mermaid diagrams into format-agnostic structural edge records.

Reads classDiagram, requirementDiagram, architecture-beta, and flowchart --
which the image prompt emits only for architecture, never for a process. Other
types yield no edges: a sequence diagram's arrows are messages in time, not
relationships. Nesting is an edge too, since an architecture diagram often
carries no arrows at all, so a subgraph member or a `service ... in group`
becomes an edge labelled "contains". A line matching no known pattern is
skipped rather than raising.
"""

import re
from typing import Any, Dict, List

# Edge label used for a nesting relationship (subgraph member, `in` clause).
CONTAINS = "contains"

# Diagram-type header -> internal kind. Order matters: the longer, more
# specific header must be tried before a prefix of it.
_DIAGRAM_HEADERS = (
    ("architecture-beta", "architecture"),
    ("requirementdiagram", "requirement"),
    ("classdiagram", "class"),
    ("flowchart", "flowchart"),
    ("graph", "flowchart"),
)

# Non-edge lines to skip outright. \b keeps "end" from swallowing "endpoint".
_DIRECTIVE_RE = re.compile(
    r"^\s*(?:%%|(?:flowchart|graph|classDiagram|sequenceDiagram|stateDiagram(?:-v2)?|"
    r"erDiagram|requirementDiagram|architecture-beta|gantt|pie|journey|"
    r"subgraph|end|classDef|class|style|click|direction)\b)",
    re.IGNORECASE,
)

# Node id characters. CJK included: a diagram transcribed from a CJK document
# often uses the label itself as the id.
_ID_BODY = r"A-Za-z0-9_\-./぀-ヿ㐀-䶿一-鿿＀-￯"

# Node shapes: ID[label], ID(label), ID((label)), ID{label}.
# Capture the node id and its display label.
_NODE_DECL_RE = re.compile(rf"([{_ID_BODY}]+)\s*(?:\(\(([^)]*)\)\)|\[([^\]]*)\]|\(([^)]*)\)|\{{([^}}]*)\}})")

# Same, minus {...}: a classDiagram's braces hold a class body, not a label.
_NODE_DECL_NO_BRACE_RE = re.compile(rf"([{_ID_BODY}]+)\s*(?:\(\(([^)]*)\)\)|\[([^\]]*)\]|\(([^)]*)\))")

_NODE_TOKEN = rf"[{_ID_BODY}]+(?:\(\([^)]*\)\)|\[[^\]]*\]|\([^)]*\)|\{{[^}}]*\}})?"

# Flowchart edge operators (longest first so e.g. '<-->' wins over '-->').
_FLOW_OPS = ["<-->", "-.->", "-.-", "==>", "===", "-->", "---", "--o", "--x"]

# classDiagram relation operators.
_CLASS_OPS = ["<|--", "--|>", "o--", "--o", "*--", "--*", "<|..", "..|>", "-->", "<--", "..>", "<..", "--", ".."]

# Operators whose arrowhead points left, so src/dest must be swapped.
_REVERSED_OPS = {"<--", "<|--", "<|..", "<..", "..<"}

# architecture-beta declarations: group/service/junction, optional icon,
# optional [title], optional `in <parent>`.
_ARCH_DECL_RE = re.compile(
    rf"^(group|service|junction)\s+([{_ID_BODY}]+)\s*"
    r"(?:\(([^)]*)\))?\s*"
    r"(?:\[([^\]]*)\])?\s*"
    rf"(?:in\s+([{_ID_BODY}]+))?\s*$",
    re.IGNORECASE,
)

# architecture-beta edge: `db:R -- L:server`, `server{group}:B --> T:subnet{group}`.
_ARCH_EDGE_RE = re.compile(
    rf"^([{_ID_BODY}]+)(?:\{{group\}})?\s*:\s*([TBLR])\s*"
    r"(<?-+>?)\s*"
    rf"([TBLR])\s*:\s*([{_ID_BODY}]+)(?:\{{group\}})?\s*$",
    re.IGNORECASE,
)

_REQ_TYPES = (
    "requirement|functionalRequirement|interfaceRequirement|"
    "performanceRequirement|physicalRequirement|designConstraint|element"
)
_REQ_BLOCK_RE = re.compile(rf"^(?:{_REQ_TYPES})\s+(\S+)\s*\{{", re.IGNORECASE)
_REQ_TEXT_RE = re.compile(r"^text\s*:\s*(.+)$", re.IGNORECASE)

_REQ_REL_TYPES = "contains|copies|derives|satisfies|verifies|refines|traces"
# `{src} - <type> -> {dst}`
_REQ_REL_FWD_RE = re.compile(rf"^(\S+)\s*-\s*({_REQ_REL_TYPES})\s*->\s*(\S+)\s*$", re.IGNORECASE)
# `{dst} <- <type> - {src}`
_REQ_REL_REV_RE = re.compile(rf"^(\S+)\s*<-\s*({_REQ_REL_TYPES})\s*-\s*(\S+)\s*$", re.IGNORECASE)


def _clean_label(text: str) -> str:
    """Strip wrapping quotes/backticks and line-break markup, collapse whitespace.

    A node label writes its line break as the two characters backslash-n, which
    would otherwise survive into the extracted label.
    """
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = text.replace("\\n", " ")
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


def _build_label_map(text: str, allow_brace: bool = True) -> Dict[str, str]:
    """Map node id -> display label from all shape declarations in the text.

    Set allow_brace False for a classDiagram, whose braces hold a class body
    rather than a node label.
    """
    label_map: Dict[str, str] = {}
    pattern = _NODE_DECL_RE if allow_brace else _NODE_DECL_NO_BRACE_RE
    for m in pattern.finditer(text):
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


def _detect_diagram_type(mermaid_text: str) -> str | None:
    """Read the declared diagram type from the first meaningful line.

    None for a type this module does not read, so no grammar is guessed at.
    """
    for line in mermaid_text.splitlines():
        stmt = line.strip()
        if not stmt or stmt.startswith("%%") or stmt.startswith("```"):
            continue
        head = stmt.lower()
        for prefix, kind in _DIAGRAM_HEADERS:
            if head.startswith(prefix):
                return kind
        return None
    return None


def is_structural_diagram(mermaid_text: str) -> bool:
    """Whether this diagram states structure.

    Shared by both paths out of an image: the text going to class extraction
    and the edges going to property extraction.
    """
    return bool(mermaid_text) and _detect_diagram_type(mermaid_text) is not None


def _make_edge(left: str, right: str, edge_label: str, op: str, label_map: Dict[str, str]) -> Dict[str, str]:
    """Build an edge record, normalizing direction so arrowhead -> dest."""
    src_token, dest_token = (right, left) if op in _REVERSED_OPS else (left, right)
    return {
        "src_label": _node_label(src_token, label_map),
        "dest_label": _node_label(dest_token, label_map),
        "edge_label": _clean_label(edge_label),
    }


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


def _subgraph_label(rest: str) -> str:
    """Resolve the display label of a `subgraph` header's remainder.

    Handles `subgraph name`, `subgraph id [title]` and `subgraph id["title"]`,
    preferring the title the diagram displays over the bare id.
    """
    for pattern in (r"^(\S+)\s*\[([^\]]*)\]\s*$", r"^(\S+)\s*\(([^)]*)\)\s*$"):
        m = re.match(pattern, rest)
        if m:
            return _clean_label(m.group(2)) or _clean_label(m.group(1))
    return _clean_label(rest)


def _parse_flowchart(text: str, label_map: Dict[str, str]) -> List[Dict[str, str]]:
    """Read arrows and subgraph nesting from an architecture-style flowchart."""
    edges: List[Dict[str, str]] = []
    stack: List[str] = []

    for stmt in _split_statements(text):
        low = stmt.lower()

        if low.startswith("subgraph"):
            stack.append(_subgraph_label(stmt[len("subgraph") :].strip()))
            continue
        if low == "end":
            if stack:
                stack.pop()
            continue
        if _DIRECTIVE_RE.match(stmt):
            continue

        edge = _match_edge(stmt, _FLOW_OPS, label_map)
        if edge is not None:
            edges.append(edge)
            continue

        # Not an arrow: inside a subgraph this line declares a member.
        if stack and stack[-1]:
            member = _node_label(stmt, label_map) if _NODE_DECL_RE.match(stmt) else _clean_label(stmt)
            if member:
                edges.append({"src_label": stack[-1], "dest_label": member, "edge_label": CONTAINS})

    return edges


def _parse_class(text: str, label_map: Dict[str, str]) -> List[Dict[str, str]]:
    """Read relations from a classDiagram using its own operator set."""
    edges: List[Dict[str, str]] = []
    for stmt in _split_statements(text):
        if _DIRECTIVE_RE.match(stmt):
            continue
        edge = _match_edge(stmt, _CLASS_OPS, label_map)
        if edge is not None:
            edges.append(edge)
    return edges


def _parse_architecture(text: str) -> List[Dict[str, str]]:
    """Read `in` nesting and side-anchored arrows from an architecture-beta diagram."""
    labels: Dict[str, str] = {}
    parents: List[tuple] = []
    edges: List[Dict[str, str]] = []

    for stmt in _split_statements(text):
        decl = _ARCH_DECL_RE.match(stmt)
        if not decl:
            continue
        node_id, title, parent = decl.group(2), decl.group(4), decl.group(5)
        labels[node_id] = _clean_label(title) or node_id
        if parent:
            parents.append((parent, node_id))

    for parent_id, child_id in parents:
        edges.append(
            {
                "src_label": labels.get(parent_id, parent_id),
                "dest_label": labels.get(child_id, child_id),
                "edge_label": CONTAINS,
            }
        )

    for stmt in _split_statements(text):
        m = _ARCH_EDGE_RE.match(stmt)
        if not m:
            continue
        src_id, op, dest_id = m.group(1), m.group(3), m.group(5)
        # A left-only arrowhead means the edge is written back-to-front.
        if op.startswith("<") and not op.endswith(">"):
            src_id, dest_id = dest_id, src_id
        edges.append(
            {
                "src_label": labels.get(src_id, src_id),
                "dest_label": labels.get(dest_id, dest_id),
                "edge_label": "",
            }
        )

    return edges


def _parse_requirement(text: str) -> List[Dict[str, str]]:
    """Read requirement/element relations, resolving each name to its `text:`.

    Relations name a block, but the statement label matching needs is in `text:`.
    """
    labels: Dict[str, str] = {}
    current: str | None = None

    for line in text.splitlines():
        stmt = line.strip()
        if not stmt:
            continue
        block = _REQ_BLOCK_RE.match(stmt)
        if block:
            current = block.group(1)
            labels.setdefault(current, current)
            continue
        if current is None:
            continue
        if stmt.startswith("}"):
            current = None
            continue
        field = _REQ_TEXT_RE.match(stmt)
        if field:
            label = _clean_label(field.group(1))
            if label:
                labels[current] = label

    edges: List[Dict[str, str]] = []
    for line in text.splitlines():
        stmt = line.strip()
        fwd = _REQ_REL_FWD_RE.match(stmt)
        rev = _REQ_REL_REV_RE.match(stmt)
        if fwd:
            src, rel, dest = fwd.group(1), fwd.group(2), fwd.group(3)
        elif rev:
            dest, rel, src = rev.group(1), rev.group(2), rev.group(3)
        else:
            continue
        edges.append(
            {
                "src_label": labels.get(src, src),
                "dest_label": labels.get(dest, dest),
                "edge_label": rel.lower(),
            }
        )
    return edges


def parse_mermaid_edges(mermaid_text: str) -> List[Dict[str, str]]:
    """Parse a structural Mermaid diagram into edge records.

    Args:
        mermaid_text: Raw Mermaid diagram string.

    Returns:
        List of {"src_label", "dest_label", "edge_label"} records (deduplicated).
        Returns [] for empty input or for a diagram type whose edges do not
        express structure.
    """
    if not mermaid_text or not mermaid_text.strip():
        return []

    kind = _detect_diagram_type(mermaid_text)
    if kind is None:
        return []

    if kind == "architecture":
        raw = _parse_architecture(mermaid_text)
    elif kind == "requirement":
        raw = _parse_requirement(mermaid_text)
    else:
        is_class = kind == "class"
        label_map = _build_label_map(mermaid_text, allow_brace=not is_class)
        raw = _parse_class(mermaid_text, label_map) if is_class else _parse_flowchart(mermaid_text, label_map)

    # A recursive association is real; a component containing itself is not.
    allow_self = kind == "class"

    edges: List[Dict[str, str]] = []
    seen = set()
    for edge in raw:
        src, dest = edge["src_label"], edge["dest_label"]
        if not src or not dest:
            continue
        if not allow_self and src.strip().lower() == dest.strip().lower():
            continue
        key = (src, dest, edge["edge_label"])
        if key in seen:
            continue
        seen.add(key)
        edges.append(edge)

    return edges


def build_extracted_structure(image_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build format-agnostic structure records from structural image chunks.

    Args:
        image_chunks: Image chunks with an "analysis" dict that may carry
            "has_structural_content" and "structural_description" (Mermaid).

    Returns:
        List of {"src_label", "dest_label", "edge_label", "file_id"} records.
    """
    structure: List[Dict[str, str]] = []
    for chunk in image_chunks:
        analysis = chunk.get("analysis", {}) or {}
        if not analysis.get("has_structural_content"):
            continue
        mermaid = analysis.get("structural_description")
        if not mermaid:
            continue
        file_id = chunk.get("file_id", "")
        for edge in parse_mermaid_edges(mermaid):
            structure.append({**edge, "file_id": file_id})
    return structure


if __name__ == "__main__":
    samples = [
        # Architecture flowchart stating its structure purely by nesting.
        "flowchart TD\nsubgraph Handset\n  Screen\n  Auth\n  ContactLog\nend\n"
        "subgraph NotificationServer\n  CaseLinkage\n  Statistics\nend",
        # Bidirectional edge and a label carrying a literal \n. Node ids are
        # non-ASCII on purpose: they cover the CJK ranges in _ID_BODY, which a
        # diagram transcribed from a Japanese document relies on.
        'flowchart TD\nsubgraph 端末A\nA1["接触記録データ"]\nA2["端末間通信 \\n TEK 01(A) ... TEK 14(A)"]\nend\n'
        "端末A <-->|Bluetooth| 端末B",
        "flowchart LR\n  A -->|uses| B\n  X -- depends --> Y",
        # architecture-beta: `in` nesting plus a side-anchored arrow.
        "architecture-beta\n  group api(cloud)[Public API]\n  service db(database)[My Database] in api\n"
        "  service server(server)[Server] in api\n  db:R -- L:server",
        # requirementDiagram: relations resolve to each block's text:.
        "requirementDiagram\nfunctionalRequirement req1 {\n  id: 1\n  text: detect and record contact\n}\n"
        "element app {\n  type: simulation\n}\napp - satisfies -> req1",
        # A class may relate to itself; the self-edge survives.
        "classDiagram\n  KGSnapshot o-- KGObject : hasObject\n  KGObject <|-- KGNode\n  KGNode --> KGNode : parent",
        # A component relating to itself does not; this yields nothing.
        "flowchart TD\n  HealthCenter --> HealthCenter",
        "sequenceDiagram\n  Alice->>John: Hello",  # time, not structure -> no edges
        "this is not mermaid at all !!!",
    ]
    for s in samples:
        print("=" * 68)
        print(s.splitlines()[0])
        for e in parse_mermaid_edges(s):
            print(f"  {e['src_label']!r} --[{e['edge_label']}]--> {e['dest_label']!r}")
