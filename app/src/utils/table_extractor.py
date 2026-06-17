"""Generic table parsing library: normalize tables from various source formats.

Each parser converts a source-specific table representation into a uniform,
source-independent core:

    {
        "row_headers": List[str],   # verbatim text of the leftmost (header) column
        "col_headers": List[str],   # verbatim text of the top (header) row
        "cells": List[List[str]],   # row-major body cells; "" for empty cells
        "caption": Optional[str],   # native caption element only (no heuristics)
    }

Source-dependent context (file_id, section_path, surrounding text window) is NOT
handled here -- callers add that when assembling `structured_blocks`.

Only ``extract_tables_from_xhtml`` is implemented today (used for Confluence
storage format). ``extract_tables_from_markdown`` can be added later (e.g. for
GitHub) with the same return type so downstream processing stays shared.
"""

from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple


def _attr_int(attrs: Any, name: str, default: int = 1) -> int:
    """Read an integer attribute (e.g. colspan/rowspan); fall back to default."""
    for k, v in attrs:
        if k.lower() == name:
            try:
                n = int(v)
                return n if n >= 1 else default
            except TypeError, ValueError:
                return default
    return default


class _TableHTMLParser(HTMLParser):
    """Collect <table> grids from (X)HTML, tolerant of non-table markup.

    Confluence storage format mixes standard table tags with macro tags
    (``<ac:*>`` / ``<ri:*>``); those are simply ignored while their text content
    is still captured. ``colspan``/``rowspan`` are captured per cell so that
    merged cells (including mid-row/mid-column merges) can be expanded into a
    correctly aligned grid downstream. Nested tables are not specially handled.

    Each row is collected as a list of ``(text, colspan, rowspan)`` tuples.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[Dict[str, Any]] = []
        self._in_table = False
        self._in_cell = False
        self._rows: List[List[Tuple[str, int, int]]] = []
        self._current_row: Optional[List[Tuple[str, int, int]]] = None
        self._cell_buffer: List[str] = []
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._caption: Optional[str] = None
        self._in_caption = False
        self._caption_buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._rows = []
            self._caption = None
        elif not self._in_table:
            return
        elif tag == "caption":
            self._in_caption = True
            self._caption_buffer = []
        elif tag == "tr":
            self._current_row = []
        elif tag in ("th", "td"):
            self._in_cell = True
            self._cell_buffer = []
            self._cell_colspan = _attr_int(attrs, "colspan")
            self._cell_rowspan = _attr_int(attrs, "rowspan")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self._in_table:
            return
        if tag == "caption":
            self._in_caption = False
            text = " ".join("".join(self._caption_buffer).split()).strip()
            self._caption = text or None
        elif tag in ("th", "td") and self._in_cell:
            text = " ".join("".join(self._cell_buffer).split()).strip()
            if self._current_row is not None:
                self._current_row.append((text, self._cell_colspan, self._cell_rowspan))
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            self._rows.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            self.tables.append({"rows": self._rows, "caption": self._caption})
            self._in_table = False
            self._rows = []

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption_buffer.append(data)
        elif self._in_cell:
            self._cell_buffer.append(data)


def _build_aligned_grid(rows_with_spans: List[List[Tuple[str, int, int]]]) -> List[List[str]]:
    """Expand colspan/rowspan into a rectangular, column-aligned grid.

    Each merged cell's text is repeated across every grid position it covers, so
    a cell read at ``grid[r][c]`` always lines up with the correct row/column even
    when merges occur mid-row or mid-column. ``rowspan`` reaching past the last
    parsed row is clamped.
    """
    grid: Dict[Tuple[int, int], str] = {}
    occupied = set()  # (r, c) positions reserved by a rowspan from an earlier row
    n_rows = len(rows_with_spans)
    max_col = 0

    for r, row in enumerate(rows_with_spans):
        c = 0
        for text, cspan, rspan in row:
            # Skip columns already taken by a rowspan started in an earlier row.
            while (r, c) in occupied:
                c += 1
            for dc in range(cspan):
                for dr in range(rspan):
                    rr, cc = r + dr, c + dc
                    if rr >= n_rows:
                        continue  # rowspan beyond table bounds
                    grid[(rr, cc)] = text
                    if dr > 0:
                        occupied.add((rr, cc))
            c += cspan
        max_col = max(max_col, c)

    return [[grid.get((r, c), "") for c in range(max_col)] for r in range(n_rows)]


def _normalize_table(
    rows_with_spans: List[List[Tuple[str, int, int]]], caption: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Turn a raw (possibly merged-cell) grid into the source-independent table core.

    Interpretation: the first row holds column headers and the first column holds
    row headers; the remaining cells are the body. The top-left corner cell is
    excluded from both header lists. Tables without at least a 2x2 shape (one
    header row + one header column + one body cell) are skipped.
    """
    # Expand merges first, then drop fully empty rows.
    grid = [r for r in _build_aligned_grid(rows_with_spans) if any(c.strip() for c in r)]
    if len(grid) < 2:
        return None

    width = max(len(r) for r in grid)
    if width < 2:
        return None

    col_headers = grid[0][1:width]  # skip top-left corner
    row_headers = [grid[i][0] for i in range(1, len(grid))]
    cells = [grid[i][1:width] for i in range(1, len(grid))]

    return {
        "row_headers": row_headers,
        "col_headers": col_headers,
        "cells": cells,
        "caption": caption,
    }


def extract_tables_from_xhtml(xhtml: str) -> List[Dict[str, Any]]:
    """Parse all tables in an (X)HTML string into normalized table cores.

    Args:
        xhtml: (X)HTML string (e.g. Confluence storage format body).

    Returns:
        List of normalized table cores (see module docstring). Degenerate tables
        (smaller than 2x2) are omitted. Returns [] for empty/invalid input.
    """
    if not xhtml or not xhtml.strip():
        return []

    parser = _TableHTMLParser()
    try:
        parser.feed(xhtml)
        parser.close()
    except Exception:
        # HTMLParser is lenient, but guard against pathological input.
        return []

    result: List[Dict[str, Any]] = []
    for raw in parser.tables:
        normalized = _normalize_table(raw["rows"], raw["caption"])
        if normalized:
            result.append(normalized)
    return result


def extract_tables_from_pdf(page) -> List[Dict[str, Any]]:
    """Parse all tables in a PDF page into normalized table cores.

    Uses pdfplumber's table extraction to detect tables in the PDF page,
    then normalizes them to the source-independent format. Unlike XHTML tables,
    PDF tables have no semantic structure (no colspan/rowspan, no caption elements),
    so we make standard assumptions: first row = column headers, first column = row headers.

    Args:
        page: pdfplumber Page object

    Returns:
        List of normalized table cores (see module docstring). Degenerate tables
        (smaller than 2x2) are omitted. Returns [] if no tables found or on error.

    Examples:
        >>> import pdfplumber
        >>> with pdfplumber.open("sample.pdf") as pdf:
        ...     tables = extract_tables_from_pdf(pdf.pages[0])
        ...     if tables:
        ...         print(f"Found {len(tables)} tables")
    """
    if page is None:
        return []

    try:
        # Extract tables using pdfplumber (returns list of list of lists)
        raw_tables = page.extract_tables()
        if not raw_tables:
            return []
    except Exception:
        # Guard against pathological PDFs or pdfplumber errors
        return []

    result: List[Dict[str, Any]] = []
    for raw_table in raw_tables:
        try:
            # Normalize None to empty string and strip whitespace
            normalized_grid = []
            for row in raw_table:
                normalized_row = []
                for cell in row:
                    # pdfplumber returns None for empty cells
                    cell_text = "" if cell is None else str(cell).strip()
                    normalized_row.append(cell_text)
                normalized_grid.append(normalized_row)

            # Check minimum 2x2 shape (at least 1 header row + 1 header col + 1 body cell)
            if len(normalized_grid) < 2:
                continue
            if any(len(row) < 2 for row in normalized_grid):
                continue

            # Extract headers and body
            # First row (skip [0][0] corner) = column headers
            col_headers = normalized_grid[0][1:]

            # First column (from row 1 onwards) = row headers
            row_headers = [row[0] for row in normalized_grid[1:]]

            # Remaining cells = body (from [1:][1:])
            cells = [row[1:] for row in normalized_grid[1:]]

            # PDFs don't have semantic caption elements
            caption = None

            # Build normalized table core
            table_core = {
                "row_headers": row_headers,
                "col_headers": col_headers,
                "cells": cells,
                "caption": caption,
            }

            result.append(table_core)

        except Exception:
            # Skip malformed tables, continue with others
            continue

    return result


if __name__ == "__main__":
    import json

    sample = """
    <table>
      <caption>要件・機能対応表</caption>
      <tbody>
        <tr><th></th><th>ログイン機能</th><th>SSO連携機能</th><th>監査ログ機能</th></tr>
        <tr><th>ユーザー認証要件</th><td>○</td><td>○</td><td></td></tr>
        <tr><th>アクセス制御要件</th><td></td><td>○</td><td>○</td></tr>
        <tr><th>データ暗号化要件</th><td></td><td></td><td>○</td></tr>
      </tbody>
    </table>
    """
    tables = extract_tables_from_xhtml(sample)
    print(json.dumps(tables, ensure_ascii=False, indent=2))
    assert tables, "expected one table"
    t = tables[0]
    assert t["caption"] == "要件・機能対応表"
    assert t["col_headers"] == ["ログイン機能", "SSO連携機能", "監査ログ機能"]
    assert t["row_headers"] == ["ユーザー認証要件", "アクセス制御要件", "データ暗号化要件"]
    assert t["cells"][0] == ["○", "○", ""]

    # Merged cells: rowspan on the row header (col 0) + colspan on a data cell.
    # Row 1's "○" spans cols 1-2 (colspan=2); the row header "認証要件" spans rows 1-2 (rowspan=2).
    merged = """
    <table>
      <tbody>
        <tr><th></th><th>ログイン機能</th><th>SSO連携機能</th><th>監査ログ機能</th></tr>
        <tr><th rowspan="2">認証要件</th><td colspan="2">○</td><td></td></tr>
        <tr><td></td><td></td><td>○</td></tr>
      </tbody>
    </table>
    """
    mt = extract_tables_from_xhtml(merged)[0]
    print("\n[merged] " + json.dumps(mt, ensure_ascii=False))
    # rowspan repeats the header down both rows; colspan repeats "○" across cols 1-2.
    assert mt["row_headers"] == ["認証要件", "認証要件"], mt["row_headers"]
    assert mt["cells"][0] == ["○", "○", ""], mt["cells"][0]
    assert mt["cells"][1] == ["", "", "○"], mt["cells"][1]
    print("\nOK")
