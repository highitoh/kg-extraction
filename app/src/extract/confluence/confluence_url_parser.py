"""Parse Confluence URLs and generate stable identifiers."""

from typing import Dict


def parse_confluence_url(url: str) -> Dict[str, str]:
    """Parse Confluence URL to extract components.

    Supports multiple URL formats:
    - Full URL (space): https://instance.atlassian.net/wiki/spaces/SPACEKEY
    - Full URL (page): https://instance.atlassian.net/wiki/spaces/SPACEKEY/pages/123456/Page+Title
    - Shorthand (space): confluence:instance.atlassian.net:SPACEKEY
    - Shorthand (page): confluence:instance.atlassian.net:SPACEKEY:123456

    Args:
        url: Confluence URL or identifier

    Returns:
        Dict with keys: domain, space_key, page_id (optional), original_url

    Raises:
        ValueError: If URL format is invalid

    Examples:
        >>> parse_confluence_url("https://company.atlassian.net/wiki/spaces/DOCS")
        {'domain': 'company.atlassian.net', 'space_key': 'DOCS', 'page_id': None, 'original_url': '...'}

        >>> parse_confluence_url("confluence:company.atlassian.net:DOCS:123456")
        {'domain': 'company.atlassian.net', 'space_key': 'DOCS', 'page_id': '123456', 'original_url': '...'}
    """
    original_url = url

    # Handle shorthand notation: confluence:domain:SPACE[:PAGE_ID]
    if url.startswith("confluence:"):
        parts = url.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid Confluence shorthand format: {url}. "
                "Expected format: confluence:domain:SPACE_KEY or confluence:domain:SPACE_KEY:PAGE_ID"
            )

        domain = parts[1]
        space_key = parts[2]
        page_id = parts[3] if len(parts) > 3 else None

        return {
            "domain": domain,
            "space_key": space_key,
            "page_id": page_id,
            "original_url": original_url,
        }

    # Handle full URL format
    if ".atlassian.net/wiki/" not in url:
        raise ValueError(
            f"Invalid Confluence URL: {url}. "
            "Must be an Atlassian Confluence URL (*.atlassian.net/wiki/) "
            "or use shorthand notation (confluence:domain:SPACE)"
        )

    # Extract domain
    if "https://" in url:
        domain_start = url.index("https://") + len("https://")
    elif "http://" in url:
        domain_start = url.index("http://") + len("http://")
    else:
        raise ValueError(f"Invalid Confluence URL (missing protocol): {url}")

    domain_end = url.index("/wiki/", domain_start)
    domain = url[domain_start:domain_end]

    # Extract space_key and page_id
    # URL format: https://domain/wiki/spaces/SPACEKEY[/pages/PAGEID/Page+Title]
    if "/spaces/" not in url:
        raise ValueError(f"Invalid Confluence URL (missing /spaces/): {url}")

    spaces_index = url.index("/spaces/") + len("/spaces/")
    # Find the next slash after space key (or end of string)
    next_slash = url.find("/", spaces_index)

    if next_slash == -1:
        # Space-level URL (no page)
        space_key = url[spaces_index:]
        page_id = None
    else:
        space_key = url[spaces_index:next_slash]

        # Check if this is a page URL
        if "/pages/" in url[next_slash:]:
            pages_index = url.index("/pages/", next_slash) + len("/pages/")
            # Page ID is the next segment after /pages/
            page_id_end = url.find("/", pages_index)
            if page_id_end == -1:
                page_id = url[pages_index:]
            else:
                page_id = url[pages_index:page_id_end]
        else:
            page_id = None

    return {
        "domain": domain,
        "space_key": space_key,
        "page_id": page_id,
        "original_url": original_url,
    }


def get_space_identifier(domain: str, space_key: str) -> str:
    """Generate stable identifier for Confluence space.

    Args:
        domain: Confluence instance domain (e.g., "company.atlassian.net")
        space_key: Space key (e.g., "DOCS")

    Returns:
        Stable identifier in format: confluence:domain:SPACE_KEY

    Examples:
        >>> get_space_identifier("company.atlassian.net", "DOCS")
        'confluence:company.atlassian.net:DOCS'
    """
    return f"confluence:{domain}:{space_key}"


def get_page_identifier(domain: str, space_key: str, page_id: str) -> str:
    """Generate stable identifier for Confluence page.

    Args:
        domain: Confluence instance domain (e.g., "company.atlassian.net")
        space_key: Space key (e.g., "DOCS")
        page_id: Page ID (e.g., "123456")

    Returns:
        Stable identifier in format: confluence:domain:SPACE_KEY:PAGE_ID

    Examples:
        >>> get_page_identifier("company.atlassian.net", "DOCS", "123456")
        'confluence:company.atlassian.net:DOCS:123456'
    """
    return f"confluence:{domain}:{space_key}:{page_id}"
