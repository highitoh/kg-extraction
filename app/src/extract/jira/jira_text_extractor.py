"""Extract text content from Jira projects."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List

from src.extract.jira.jira_url_parser import get_issue_identifier, parse_jira_url
from src.utils.jira_client import JiraClient


def extract_text_from_jira(jira_url: str) -> List[Dict[str, Any]]:
    """Extract text content from Jira project.

    Main extraction function that orchestrates:
    1. URL parsing
    2. API calls to fetch all issues
    3. ADF to Markdown conversion
    4. Structuring for downstream processing

    Args:
        jira_url: Jira project URL or identifier

    Returns:
        List of text chunks with metadata:
        [
            {
                "text": str,  # Formatted Markdown text
                "metadata": {
                    "issue_key": str,
                    "issue_type": str,
                    "updated": str,
                    "labels": List[str],
                    "parent": Optional[str]
                },
                "file_id": str,  # "jira:domain:PROJ-123"
                "_jira_raw_data": dict  # Attached to first chunk only
            },
            ...
        ]

    Examples:
        >>> chunks = extract_text_from_jira("https://company.atlassian.net/browse/PROJ-123")
        >>> len(chunks) > 0
        True
    """
    print(f"[JiraExtractor] Extracting text from Jira URL: {jira_url}")

    # Parse URL
    parsed_url = parse_jira_url(jira_url)
    domain = parsed_url["domain"]
    project_key = parsed_url["project_key"]

    print(f"[JiraExtractor] Domain: {domain}, Project: {project_key}")

    # Initialize Jira client
    client = JiraClient(instance_domain=domain)

    # Fetch all issues from project
    all_issues = _fetch_all_issues(client, project_key)

    print(f"[JiraExtractor] Fetched {len(all_issues)} issues from project {project_key}")

    # Transform issues to chunks
    chunks = []
    for issue in all_issues:
        try:
            chunk = _issue_to_chunk(client, issue, domain)
            chunks.append(chunk)
        except Exception as e:
            issue_key = issue.get("key", "unknown")
            print(f"[JiraExtractor] Error processing issue {issue_key}: {e}")
            # Continue processing other issues
            continue

    # Attach raw data to first chunk for snapshot tracking
    if chunks:
        chunks[0]["_jira_raw_data"] = _build_raw_data(domain, project_key, all_issues)

    return chunks


def _fetch_all_issues(client: JiraClient, project_key: str) -> List[Dict[str, Any]]:
    """Fetch all issues from a Jira project with pagination.

    Args:
        client: JiraClient instance
        project_key: Project key (e.g., "PROJ")

    Returns:
        List of all issue objects
    """
    jql = f"project = {project_key} ORDER BY updated DESC"
    all_issues = []
    start_at = 0
    max_results = 50

    while True:
        response = client.search_issues(jql, start_at=start_at, max_results=max_results)
        issues = response.get("issues", [])
        all_issues.extend(issues)

        total = response.get("total", 0)
        fetched = len(all_issues)

        print(f"[JiraExtractor] Fetched {fetched}/{total} issues")

        # Check if we've fetched all issues
        if fetched >= total:
            break

        # Move to next page
        start_at += max_results

    return all_issues


def _issue_to_chunk(client: JiraClient, issue: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Convert a Jira issue to a text chunk.

    Args:
        client: JiraClient instance (for ADF conversion)
        issue: Jira issue object
        domain: Jira instance domain

    Returns:
        Text chunk with metadata
    """
    issue_key = issue["key"]
    fields = issue.get("fields", {})

    # Extract basic fields
    title = fields.get("summary", "")
    description_adf = fields.get("description")
    issue_type = fields.get("issuetype", {}).get("name", "")
    updated = fields.get("updated", "")
    labels = fields.get("labels", [])
    parent = fields.get("parent", {}).get("key") if fields.get("parent") else None

    # Convert description ADF to Markdown
    description_md = client.convert_adf_to_markdown(description_adf) if description_adf else ""

    # Extract and convert comments
    comment_field = fields.get("comment", {})
    comments = comment_field.get("comments", [])
    comments_md = _extract_comments(client, comments)

    # Format text as Markdown
    text = _format_issue_markdown(title, description_md, comments_md)

    # Build metadata
    metadata = {
        "issue_key": issue_key,
        "issue_type": issue_type,
        "updated": updated,
        "labels": labels,
        "parent": parent,
    }

    # Generate file_id
    file_id = get_issue_identifier(domain, issue_key)

    return {"text": text, "metadata": metadata, "file_id": file_id}


def _extract_comments(client: JiraClient, comments: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract and convert comments to Markdown.

    Args:
        client: JiraClient instance
        comments: List of comment objects from Jira API

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
        author_obj = comment.get("author", {})
        # Use displayName if available, fallback to accountId
        author = author_obj.get("displayName") or author_obj.get("accountId", "Unknown User")

        # Extract created timestamp
        created = comment.get("created", "")

        # Convert comment body (ADF) to Markdown
        body_adf = comment.get("body")
        body_md = client.convert_adf_to_markdown(body_adf) if body_adf else ""

        processed_comments.append({"author": author, "created": created, "body": body_md})

    # Sort comments newest first (reverse chronological)
    processed_comments.sort(key=lambda c: c["created"], reverse=True)

    return processed_comments


def _format_issue_markdown(title: str, description: str, comments: List[Dict[str, str]]) -> str:
    """Format issue as Markdown text.

    Args:
        title: Issue title
        description: Description in Markdown
        comments: List of comment dicts (newest first)

    Returns:
        Formatted Markdown string
    """
    lines = []

    # Title
    lines.append("# Title")
    lines.append(title)
    lines.append("")

    # Description
    lines.append("## Description")
    if description:
        lines.append(description)
    else:
        lines.append("(No description)")
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


def _build_raw_data(domain: str, project_key: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build raw data structure for snapshot tracking.

    Args:
        domain: Jira instance domain
        project_key: Project key
        issues: List of all issues

    Returns:
        Raw data dict with issue_map for change detection
    """
    issue_map = {}

    for issue in issues:
        issue_key = issue["key"]
        fields = issue.get("fields", {})
        updated = fields.get("updated", "")

        # Compute content hash for change detection
        # Use summary, description, updated timestamp, and comment count
        content_for_hash = {
            "summary": fields.get("summary", ""),
            "description": str(fields.get("description", "")),
            "updated": updated,
            "comment_count": fields.get("comment", {}).get("total", 0),
        }
        content_json = json.dumps(content_for_hash, sort_keys=True)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()[:16]

        issue_map[issue_key] = {"updated": updated, "sha": content_hash}

    return {
        "domain": domain,
        "project_key": project_key,
        "issue_map": issue_map,
        "total_issues": len(issues),
        "fetched_at": datetime.now().isoformat(),
    }
