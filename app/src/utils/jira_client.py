"""Jira Cloud API client for fetching issue data."""

import base64
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()


class JiraClient:
    """Client for interacting with Jira Cloud REST API v3.

    Authentication:
        Uses email + API token from ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN environment variables.
        Falls back to JIRA_EMAIL and JIRA_API_TOKEN for backward compatibility.
        API tokens can be generated at: https://id.atlassian.com/manage-profile/security/api-tokens

    Rate Limiting:
        Implements exponential backoff with jitter for 429 responses.
        Jira Cloud limit: 6,000 requests per hour per instance.

    Error Handling:
        - 401 Unauthorized: Invalid credentials (email/token)
        - 403 Forbidden: No access to project/resource
        - 404 Not Found: Project, issue, or resource not found
        - 429 Rate Limit: Retry with exponential backoff
    """

    def __init__(
        self,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        instance_domain: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """Initialize Jira Cloud API client.

        Args:
            email: Atlassian account email (defaults to ATLASSIAN_EMAIL, falls back to JIRA_EMAIL)
            api_token: Atlassian API token (defaults to ATLASSIAN_API_TOKEN, falls back to JIRA_API_TOKEN)
            instance_domain: Atlassian instance domain (defaults to ATLASSIAN_INSTANCE_DOMAIN,
                            falls back to JIRA_INSTANCE_DOMAIN), e.g., "your-company.atlassian.net"
            max_retries: Maximum number of retries for rate limiting (default: 3)
            timeout: Request timeout in seconds (default: 30.0)

        Note:
            All three credentials (email, api_token, instance_domain) are required.
        """
        self.email = email or os.getenv("ATLASSIAN_EMAIL") or os.getenv("JIRA_EMAIL")
        self.api_token = api_token or os.getenv("ATLASSIAN_API_TOKEN") or os.getenv("JIRA_API_TOKEN")
        self.instance_domain = (
            instance_domain or os.getenv("ATLASSIAN_INSTANCE_DOMAIN") or os.getenv("JIRA_INSTANCE_DOMAIN")
        )

        if not self.email or not self.api_token:
            raise ValueError(
                "Missing Atlassian credentials. Set ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN environment variables "
                "Generate API token at: https://id.atlassian.com/manage-profile/security/api-tokens"
            )

        if not self.instance_domain:
            raise ValueError(
                "Missing Atlassian instance domain. Set ATLASSIAN_INSTANCE_DOMAIN environment variable "
                "(e.g., 'your-company.atlassian.net')"
            )

        self.base_url = f"https://{self.instance_domain}/rest/api/3"
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

        # Set headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "KnowledgeExtractor/1.0",
        }

        # Add Basic Authentication
        # Jira Cloud uses email:api_token encoded in base64
        auth_string = f"{self.email}:{self.api_token}"
        auth_bytes = auth_string.encode("ascii")
        base64_bytes = base64.b64encode(auth_bytes)
        base64_string = base64_bytes.decode("ascii")
        headers["Authorization"] = f"Basic {base64_string}"

        self.session.headers.update(headers)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Any:
        """Make HTTP request to Jira API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/search")
            **kwargs: Additional arguments passed to requests.request()

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: For 4xx/5xx responses (except 429, which is retried)
            ValueError: For authentication/permission errors with clear messages
        """
        # Ensure base_url ends with / and endpoint doesn't start with / for proper joining
        base = self.base_url.rstrip("/") + "/"
        endpoint_clean = endpoint.lstrip("/")
        url = urljoin(base, endpoint_clean)
        kwargs.setdefault("timeout", self.timeout)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                # Handle specific error codes with clear messages
                if response.status_code == 401:
                    raise ValueError(
                        "Invalid Jira credentials (email or API token). "
                        "Generate a new API token at: https://id.atlassian.com/manage-profile/security/api-tokens"
                    )
                elif response.status_code == 403:
                    raise ValueError(
                        "No access to requested Jira resource. "
                        "Check project permissions and ensure your account has access."
                    )
                elif response.status_code == 404:
                    # Extract meaningful info from endpoint for better error message
                    if "/search" in endpoint:
                        raise ValueError("Project not found or no issues match the query. Check project key.")
                    else:
                        raise ValueError(f"Resource not found: {endpoint}")
                elif response.status_code == 429:
                    # Rate limit - retry with exponential backoff
                    if attempt < self.max_retries:
                        retry_after = int(response.headers.get("Retry-After", 2**attempt))
                        jitter = 0.1 * retry_after  # 10% jitter
                        sleep_time = retry_after + jitter
                        print(
                            f"[JiraClient] Rate limited (429). Retrying in {sleep_time:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        response.raise_for_status()

                response.raise_for_status()

                # Return JSON response
                return response.json()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    print(f"[JiraClient] Request timeout. Retrying (attempt {attempt + 1}/{self.max_retries})")
                    continue
                raise
            except requests.exceptions.RequestException as e:
                print(f"[JiraClient] Request failed: {e}")
                raise

        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded")

    def search_issues(
        self,
        jql: str,
        start_at: int = 0,
        max_results: int = 50,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search issues using JQL (Jira Query Language).

        Args:
            jql: JQL query string (e.g., "project = PROJ ORDER BY updated DESC")
            start_at: Pagination offset (0-based)
            max_results: Number of results to return (max 100)
            fields: List of fields to return. Defaults to common fields.

        Returns:
            Search results including:
            - startAt: Pagination offset
            - maxResults: Results per page
            - total: Total number of matching issues
            - issues: List of issue objects

        Example:
            >>> client = JiraClient()
            >>> results = client.search_issues("project = MYPROJ", max_results=50)
            >>> print(f"Total issues: {results['total']}")
            >>> for issue in results['issues']:
            ...     print(issue['key'], issue['fields']['summary'])
        """
        if fields is None:
            # Default fields for knowledge extraction
            fields = [
                "summary",  # Title
                "description",  # Description (ADF format)
                "issuetype",  # Issue type
                "created",  # Created timestamp
                "updated",  # Updated timestamp
                "labels",  # Labels
                "parent",  # Parent issue (Epic or Story)
                "comment",  # Comments
            ]

        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": min(max_results, 100),  # Jira API max is 100
            "fields": ",".join(fields),  # Jira expects comma-separated string, not list
        }

        return self._make_request("GET", "/search/jql", params=params)

    def get_issue(self, issue_key: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch a single issue by key.

        Args:
            issue_key: Issue key (e.g., "PROJ-123")
            fields: List of fields to return

        Returns:
            Issue object with requested fields

        Example:
            >>> client = JiraClient()
            >>> issue = client.get_issue("PROJ-123")
            >>> print(issue['fields']['summary'])
        """
        params = {}
        if fields:
            params["fields"] = ",".join(fields)

        return self._make_request("GET", f"/issue/{issue_key}", params=params)

    def convert_adf_to_markdown(self, adf_content: Dict[str, Any]) -> str:
        """Convert Atlassian Document Format (ADF) to Markdown.

        Uses Jira's official conversion API endpoint.

        Args:
            adf_content: ADF JSON structure

        Returns:
            Markdown string

        Example:
            >>> client = JiraClient()
            >>> adf = {"type": "doc", "content": [...]}
            >>> markdown = client.convert_adf_to_markdown(adf)
        """
        if not adf_content:
            return ""

        try:
            # Use Jira's official ADF to Markdown conversion API
            response = self._make_request(
                "POST",
                "/contentbody/convert/markdown",
                json={"content": adf_content, "representation": "atlas_doc_format"},
            )
            return response.get("value", "")
        except Exception as e:
            # Fallback: extract plain text from ADF if conversion fails
            print(f"[JiraClient] ADF to Markdown conversion failed: {e}. Using plain text fallback.")
            return self._extract_text_from_adf(adf_content)

    def _extract_text_from_adf(self, adf_content: Dict[str, Any]) -> str:
        """Fallback: Extract plain text from ADF JSON structure.

        Args:
            adf_content: ADF JSON structure

        Returns:
            Plain text string
        """
        if not adf_content:
            return ""

        text_parts = []

        def extract_text_recursive(node):
            if isinstance(node, dict):
                # Extract text from "text" type nodes
                if node.get("type") == "text":
                    text_parts.append(node.get("text", ""))
                # Recurse into content
                if "content" in node:
                    for child in node["content"]:
                        extract_text_recursive(child)
            elif isinstance(node, list):
                for item in node:
                    extract_text_recursive(item)

        extract_text_recursive(adf_content)
        return " ".join(text_parts)
