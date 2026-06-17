"""Extract text content from Confluence spaces and pages."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.extract.confluence.confluence_url_parser import parse_confluence_url
from src.extract.confluence.storage_format_converter import StorageFormatConverter
from src.utils.confluence_client import ConfluenceClient
from src.utils.table_extractor import extract_tables_from_xhtml

# Page-level context window length (chars) attached to structured table blocks.
CONTEXT_WINDOW_CHARS = 200


def extract_text_from_confluence(confluence_url: str, since_watermark: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract text content from Confluence space or page.

    Follows PDF document pattern: splits each page's Markdown by line.

    Args:
        confluence_url: Confluence space or page URL/identifier
        since_watermark: Optional ISO8601 timestamp for incremental extraction

    Returns:
        List of sentences (line-level, from all pages combined):
        [
            {
                "text": str,  # One line from Markdown
                "metadata": {
                    "page_id": str,
                    "page_title": str,
                    "space_key": str,
                    "parent_id": Optional[str]
                }
            },
            ...
        ]
        First sentence has "_confluence_raw_data" attached for snapshot tracking.

    Examples:
        >>> sentences = extract_text_from_confluence("https://company.atlassian.net/wiki/spaces/DOCS")
        >>> len(sentences) > 0
        True
    """
    print(f"[ConfluenceExtractor] Extracting text from Confluence URL: {confluence_url}")

    # Parse URL
    parsed_url = parse_confluence_url(confluence_url)
    domain = parsed_url["domain"]
    space_key = parsed_url["space_key"]
    page_id = parsed_url.get("page_id")

    print(f"[ConfluenceExtractor] Domain: {domain}, Space: {space_key}, Page: {page_id or 'all'}")

    # Initialize Confluence client
    client = ConfluenceClient(instance_domain=domain)
    converter = StorageFormatConverter()

    # Fetch all pages (or only updated pages in incremental mode)
    all_pages = _fetch_all_pages(client, space_key, page_id, since_watermark)

    if since_watermark:
        print(
            f"[ConfluenceExtractor] Fetched {len(all_pages)} updated pages from space {space_key} "
            f"(since {since_watermark})"
        )
    else:
        print(f"[ConfluenceExtractor] Fetched {len(all_pages)} pages from space {space_key}")

    # Transform pages to sentences (line-level, following PDF pattern)
    all_sentences = []
    for page in all_pages:
        try:
            sentences = _page_to_sentences(client, converter, page, domain, space_key)
            all_sentences.extend(sentences)
        except Exception as e:
            page_id_or_title = page.get("id", page.get("title", "unknown"))
            print(f"[ConfluenceExtractor] Error processing page {page_id_or_title}: {e}")
            # Continue processing other pages
            continue

    # Build structured table blocks per page, preserving table structure before
    # the line-level flattening above. Consumed downstream for class candidates
    # (table headers) and deterministic property edges (filled cells).
    all_blocks = []
    for page in all_pages:
        try:
            all_blocks.extend(_page_to_structured_blocks(converter, page, domain, space_key))
        except Exception as e:
            page_id_or_title = page.get("id", page.get("title", "unknown"))
            print(f"[ConfluenceExtractor] Error extracting tables from page {page_id_or_title}: {e}")
            continue

    # Attach raw data + structured blocks to first sentence for the node to pick up
    if all_sentences:
        all_sentences[0]["_confluence_raw_data"] = _build_raw_data(domain, space_key, all_pages)
        all_sentences[0]["_confluence_structured_blocks"] = all_blocks

    print(
        f"[ConfluenceExtractor] Extracted {len(all_sentences)} sentences and "
        f"{len(all_blocks)} table blocks from {len(all_pages)} pages"
    )

    return all_sentences


def _fetch_all_pages(
    client: ConfluenceClient,
    space_key: str,
    page_id: Optional[str],
    since_watermark: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch all pages from Confluence space or page with pagination.

    Args:
        client: ConfluenceClient instance
        space_key: Space key (e.g., "DOCS")
        page_id: Optional page ID for page-level extraction
        since_watermark: Optional ISO8601 timestamp to fetch only updated pages

    Returns:
        List of all page objects (with full hierarchy)
    """
    all_pages = []

    if page_id:
        # Page-level extraction: fetch page + all descendants
        print(f"[ConfluenceExtractor] Fetching page {page_id} and descendants...")

        # Fetch the root page
        expand = ["body.storage", "version", "metadata.labels", "space", "ancestors"]
        root_page = client.get_page(page_id, expand=expand)
        all_pages.append(root_page)

        # Fetch all descendants recursively
        descendants = client.get_page_children_recursive(page_id)
        # Fetch full details for each descendant
        for child in descendants:
            child_id = child["id"]
            child_page = client.get_page(child_id, expand=expand)
            all_pages.append(child_page)

    else:
        # Space-level extraction: fetch all pages in space
        print(f"[ConfluenceExtractor] Fetching all pages in space {space_key}...")

        # Build CQL query
        if since_watermark:
            # Incremental mode: fetch only updated pages
            # Convert watermark to CQL-compatible format
            try:
                watermark_dt = datetime.fromisoformat(since_watermark)
                cql_timestamp = watermark_dt.strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                print(f"[ConfluenceExtractor] Error parsing watermark: {e}, using as-is")
                cql_timestamp = since_watermark

            cql = (
                f"space = {space_key} AND type = page AND lastModified >= '{cql_timestamp}' ORDER BY lastModified DESC"
            )
        else:
            cql = f"space = {space_key} AND type = page ORDER BY lastModified DESC"

        # Paginate through search results
        start = 0
        limit = 50

        while True:
            response = client.search_pages(cql, start=start, limit=limit)
            pages = response.get("results", [])

            # Fetch full page details for each result
            for page_summary in pages:
                page_id = page_summary["id"]
                expand = ["body.storage", "version", "metadata.labels", "space", "ancestors"]
                full_page = client.get_page(page_id, expand=expand)
                all_pages.append(full_page)

            fetched = len(pages)

            print(f"[ConfluenceExtractor] Fetched {start + fetched} pages...")

            # Check if we've fetched all pages
            if fetched < limit:
                break

            start += limit

    return all_pages


def _page_to_sentences(
    client: ConfluenceClient,
    converter: StorageFormatConverter,
    page: Dict[str, Any],
    domain: str,
    space_key: str,
) -> List[Dict[str, Any]]:
    """Convert a Confluence page to list of sentences (line-level).

    Follows PDF document pattern: split Markdown by line.

    Args:
        client: ConfluenceClient instance (for comments)
        converter: StorageFormatConverter instance
        page: Confluence page object
        domain: Confluence instance domain
        space_key: Space key

    Returns:
        List of sentence dicts (one per line)
    """
    page_id = page["id"]
    page_title = page.get("title", "")

    # Extract parent_id from ancestors
    ancestors = page.get("ancestors", [])
    parent_id = ancestors[-1]["id"] if ancestors else None

    # Extract body content
    body = page.get("body", {}).get("storage", {})
    body_storage = body.get("value", "") if body else ""

    # Convert storage format to Markdown
    body_markdown = converter.convert(body_storage) if body_storage else ""

    # Extract comments
    try:
        comments = client.get_page_comments(page_id)
        comments_md = _extract_comments(converter, comments)
    except Exception as e:
        print(f"[ConfluenceExtractor] Error fetching comments for page {page_id}: {e}")
        comments_md = []

    # Format page as Markdown
    page_markdown = _format_page_markdown(page_title, body_markdown, comments_md)

    # Split Markdown by line (following PDF document pattern)
    sentences = []
    for line in page_markdown.split("\n"):
        # Include empty lines to preserve structure
        sentences.append(
            {
                "text": line,
                "metadata": {
                    "page_id": page_id,
                    "page_title": page_title,
                    "space_key": space_key,
                    "parent_id": parent_id,
                },
            }
        )

    return sentences


def _page_to_structured_blocks(
    converter: StorageFormatConverter,
    page: Dict[str, Any],
    domain: str,
    space_key: str,
) -> List[Dict[str, Any]]:
    """Extract structured table blocks from a Confluence page.

    Parses the page's storage-format (XHTML) body into normalized table cores and
    wraps each with source-dependent context using the shared structured block builder.
    Context is page-level for now: section_path = page title, context_window = a bounded
    prefix of the page text. (Per-table heading/window positioning is a future refinement.)

    Returns:
        List of structured blocks (see KnowledgeExtractionState.structured_blocks).
    """
    from src.utils.structured_block_builder import build_table_blocks

    body = page.get("body", {}).get("storage", {})
    body_storage = body.get("value", "") if body else ""
    if not body_storage:
        return []

    tables = extract_tables_from_xhtml(body_storage)
    if not tables:
        return []

    page_id = page["id"]
    page_title = page.get("title", "")

    # Build metadata
    file_id = f"confluence:{domain}:{space_key}:{page_id}"
    scope_ref = page_id
    section_path = page_title

    # Extract context: page-level context window (bounded prefix of converted page text)
    body_markdown = converter.convert(body_storage)
    context_window = " ".join(body_markdown.split())[:CONTEXT_WINDOW_CHARS]

    # Use shared utility to build structured blocks
    return build_table_blocks(tables, file_id, scope_ref, section_path, context_window)


def _extract_comments(converter: StorageFormatConverter, comments: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract and convert comments to Markdown.

    Args:
        converter: StorageFormatConverter instance
        comments: List of comment objects from Confluence API

    Returns:
        List of comment dicts sorted newest first:
        [
            {
                "author": str,     # Display name or account ID
                "created": str,    # ISO timestamp
                "body": str        # Markdown content
            },
            ...
        ]
    """
    processed_comments = []

    for comment in comments:
        # Extract author
        # Confluence comment structure: version.by.displayName or version.by.email
        version = comment.get("version", {})
        by = version.get("by", {})
        author = by.get("displayName") or by.get("email", "Unknown User")

        # Extract created timestamp
        created = version.get("when", "")

        # Convert comment body (storage format) to Markdown
        body = comment.get("body", {}).get("storage", {})
        body_storage = body.get("value", "") if body else ""
        body_md = converter.convert(body_storage) if body_storage else ""

        processed_comments.append({"author": author, "created": created, "body": body_md})

    # Sort comments newest first (reverse chronological)
    processed_comments.sort(key=lambda c: c["created"], reverse=True)

    return processed_comments


def _format_page_markdown(title: str, body: str, comments: List[Dict[str, str]]) -> str:
    """Format page as Markdown text.

    Args:
        title: Page title
        body: Body content in Markdown
        comments: List of comment dicts (newest first)

    Returns:
        Formatted Markdown string
    """
    lines = []

    # Title
    lines.append("# Title")
    lines.append(title)
    lines.append("")

    # Content
    lines.append("## Content")
    if body:
        lines.append(body)
    else:
        lines.append("(No content)")
    lines.append("")

    # Comments
    if comments:
        lines.append("## Comments")
        lines.append("")
        for comment in comments:
            author = comment["author"]
            created = comment["created"]
            # Format timestamp (extract date portion)
            created_date = created.split("T")[0] if "T" in created else created
            lines.append(f"### {author} ({created_date})")
            lines.append(comment["body"])
            lines.append("")

    return "\n".join(lines)


def _build_raw_data(domain: str, space_key: str, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build raw data structure for snapshot tracking.

    Args:
        domain: Confluence instance domain
        space_key: Space key
        pages: List of all pages

    Returns:
        Raw data dict with page_map and watermark for change detection
    """
    page_map = {}
    updated_timestamps = []

    for page in pages:
        page_id = page["id"]
        title = page.get("title", "")
        version_info = page.get("version", {})
        version_number = version_info.get("number", 1)

        # Extract updated timestamp from version.when
        updated = version_info.get("when", "")
        if updated:
            updated_timestamps.append(updated)

        # Compute content hash for change detection
        # Use title, body, version, and comment count
        body = page.get("body", {}).get("storage", {})
        body_value = body.get("value", "") if body else ""

        content_for_hash = {
            "title": title,
            "body": body_value,
            "updated": updated,
            "version": version_number,
        }
        content_json = json.dumps(content_for_hash, sort_keys=True)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()[:16]

        page_map[page_id] = {"updated": updated, "version": version_number, "sha": content_hash}

    # Watermark: the latest "lastModified" timestamp among fetched pages
    # Sort ISO8601 strings lexicographically (works for same offset)
    watermark = max(updated_timestamps) if updated_timestamps else None

    return {
        "domain": domain,
        "space_key": space_key,
        "page_map": page_map,
        "total_pages": len(pages),
        "watermark": watermark,
    }
