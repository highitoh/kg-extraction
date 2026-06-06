"""Parse Jira URLs and generate identifiers."""

from typing import Dict
from urllib.parse import urlparse


def parse_jira_url(url: str) -> Dict[str, str]:
    """Parse Jira URL and extract components.

    Supported URL formats:
        - https://instance.atlassian.net/browse/PROJ-123
        - https://instance.atlassian.net/browse/PROJ
        - jira:instance.atlassian.net:PROJ

    Args:
        url: Jira URL or identifier

    Returns:
        Parsed URL components:
        {
            "domain": str,           # e.g., "company.atlassian.net"
            "project_key": str,      # e.g., "PROJ"
            "issue_key": str,        # e.g., "PROJ-123" (optional, may be None)
            "original_url": str      # Original input URL
        }

    Raises:
        ValueError: If URL format is invalid

    Examples:
        >>> parse_jira_url("https://company.atlassian.net/browse/PROJ-123")
        {'domain': 'company.atlassian.net', 'project_key': 'PROJ', 'issue_key': 'PROJ-123', ...}

        >>> parse_jira_url("jira:company.atlassian.net:PROJ")
        {'domain': 'company.atlassian.net', 'project_key': 'PROJ', 'issue_key': None, ...}
    """
    original_url = url

    # Handle jira: scheme
    if url.startswith("jira:"):
        parts = url.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid jira: URL format: {url}. Expected format: jira:domain:PROJECT_KEY or jira:domain:ISSUE_KEY"
            )
        domain = parts[1]
        identifier = parts[2]  # Could be project key or issue key

        # Check if it's an issue key (contains hyphen)
        if "-" in identifier:
            issue_key = identifier
            project_key = identifier.split("-")[0]
        else:
            issue_key = None
            project_key = identifier

        return {
            "domain": domain,
            "project_key": project_key,
            "issue_key": issue_key,
            "original_url": original_url,
        }

    # Handle HTTPS URLs
    parsed = urlparse(url)

    if not parsed.netloc or ".atlassian.net" not in parsed.netloc:
        raise ValueError(
            f"Invalid Jira URL: {url}. Expected format: https://{{instance}}.atlassian.net/browse/{{ISSUE-KEY}}"
        )

    domain = parsed.netloc

    # Extract issue key from path
    # Expected: /browse/PROJ-123 or /browse/PROJ
    path_parts = parsed.path.strip("/").split("/")

    if len(path_parts) < 2 or path_parts[0] != "browse":
        raise ValueError(
            f"Invalid Jira URL path: {url}. Expected format: https://{{instance}}.atlassian.net/browse/{{ISSUE-KEY}}"
        )

    identifier = path_parts[1]

    # Extract project key from issue key
    # Issue key format: PROJ-123, MY-PROJECT-456
    # Project key is everything before the first hyphen
    if "-" in identifier:
        issue_key = identifier
        project_key = identifier.split("-")[0]
    else:
        # Just a project key (unusual but possible)
        issue_key = None
        project_key = identifier

    return {
        "domain": domain,
        "project_key": project_key,
        "issue_key": issue_key,
        "original_url": original_url,
    }


def get_project_identifier(domain: str, project_key: str) -> str:
    """Generate stable project identifier.

    Args:
        domain: Jira instance domain (e.g., "company.atlassian.net")
        project_key: Project key (e.g., "PROJ")

    Returns:
        Stable project identifier (e.g., "jira:company.atlassian.net:PROJ")

    Example:
        >>> get_project_identifier("company.atlassian.net", "PROJ")
        'jira:company.atlassian.net:PROJ'
    """
    return f"jira:{domain}:{project_key}"


def get_issue_identifier(domain: str, issue_key: str) -> str:
    """Generate stable issue identifier.

    Args:
        domain: Jira instance domain (e.g., "company.atlassian.net")
        issue_key: Issue key (e.g., "PROJ-123")

    Returns:
        Stable issue identifier (e.g., "jira:company.atlassian.net:PROJ-123")

    Example:
        >>> get_issue_identifier("company.atlassian.net", "PROJ-123")
        'jira:company.atlassian.net:PROJ-123'
    """
    return f"jira:{domain}:{issue_key}"
