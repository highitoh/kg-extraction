"""Confluence Cloud API client for fetching page data."""

import base64
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()


class ConfluenceClient:
    """Client for interacting with Confluence Cloud REST API v1.

    Authentication:
        Uses email + API token from ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN environment variables.
        Falls back to JIRA_EMAIL and JIRA_API_TOKEN for backward compatibility.
        API tokens can be generated at: https://id.atlassian.com/manage-profile/security/api-tokens

    Rate Limiting:
        Implements exponential backoff with jitter for 429 responses.
        Atlassian Cloud limit: ~200 requests per second with burst.

    Error Handling:
        - 401 Unauthorized: Invalid credentials (email/token)
        - 403 Forbidden: No access to space/page
        - 404 Not Found: Space, page, or resource not found
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
        """Initialize Confluence Cloud API client.

        Args:
            email: Atlassian account email (defaults to ATLASSIAN_EMAIL, falls back to JIRA_EMAIL)
            api_token: Atlassian API token (defaults to ATLASSIAN_API_TOKEN, falls back to JIRA_API_TOKEN)
            instance_domain: Confluence instance domain (defaults to ATLASSIAN_INSTANCE_DOMAIN,
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
                "(or legacy JIRA_EMAIL and JIRA_API_TOKEN for backward compatibility). "
                "Generate API token at: https://id.atlassian.com/manage-profile/security/api-tokens"
            )

        if not self.instance_domain:
            raise ValueError(
                "Missing Atlassian instance domain. Set ATLASSIAN_INSTANCE_DOMAIN environment variable "
                "(or legacy JIRA_INSTANCE_DOMAIN for backward compatibility), e.g., 'your-company.atlassian.net'"
            )

        self.base_url = f"https://{self.instance_domain}/wiki/rest/api"
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
        # Confluence Cloud uses email:api_token encoded in base64 (same as Jira)
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
        """Make HTTP request to Confluence API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/content/search")
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
                        "Invalid Atlassian credentials (email or API token). "
                        "Generate a new API token at: https://id.atlassian.com/manage-profile/security/api-tokens"
                    )
                elif response.status_code == 403:
                    raise ValueError(
                        "No access to requested Confluence resource. "
                        "Check space/page permissions and ensure your account has access."
                    )
                elif response.status_code == 404:
                    # Extract meaningful info from endpoint for better error message
                    if "/content/search" in endpoint:
                        raise ValueError("Space not found or no pages match the query. Check space key.")
                    else:
                        raise ValueError(f"Resource not found: {endpoint}")
                elif response.status_code == 429:
                    # Rate limit - retry with exponential backoff
                    if attempt < self.max_retries:
                        retry_after = int(response.headers.get("Retry-After", 2**attempt))
                        jitter = 0.1 * retry_after  # 10% jitter
                        sleep_time = retry_after + jitter
                        print(
                            f"[ConfluenceClient] Rate limited (429). Retrying in {sleep_time:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        response.raise_for_status()

                # Raise for other HTTP errors
                response.raise_for_status()

                # Return JSON response
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    sleep_time = 2**attempt
                    print(f"[ConfluenceClient] Request failed: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise

        raise RuntimeError(f"Failed to complete request to {url} after {self.max_retries} retries")

    def get_space(self, space_key: str) -> Dict[str, Any]:
        """Fetch space metadata.

        Args:
            space_key: Space key (e.g., "DOCS")

        Returns:
            Space metadata dict

        Raises:
            ValueError: If space not found or no access
        """
        endpoint = f"/space/{space_key}"
        return self._make_request("GET", endpoint)

    def get_page(self, page_id: str, expand: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch page with optional expansions.

        Args:
            page_id: Page ID
            expand: List of fields to expand (e.g., ["body.storage", "version", "metadata.labels"])

        Returns:
            Page object dict

        Raises:
            ValueError: If page not found or no access
        """
        endpoint = f"/content/{page_id}"
        params = {}
        if expand:
            params["expand"] = ",".join(expand)

        return self._make_request("GET", endpoint, params=params)

    def search_pages(self, cql: str, start: int = 0, limit: int = 50) -> Dict[str, Any]:
        """Search pages using CQL (Confluence Query Language).

        Args:
            cql: CQL query string (e.g., "space = DOCS AND type = page")
            start: Starting index for pagination (default: 0)
            limit: Maximum results per page (default: 50, max: 100)

        Returns:
            Search results dict with "results", "start", "limit", "size"

        Examples:
            >>> client.search_pages("space = DOCS AND type = page ORDER BY lastModified DESC")
            >>> client.search_pages("space = DOCS AND lastModified >= '2026-06-08'")
        """
        endpoint = "/content/search"
        params = {"cql": cql, "start": start, "limit": min(limit, 100)}  # API max is 100

        return self._make_request("GET", endpoint, params=params)

    def get_page_children_recursive(self, page_id: str) -> List[Dict[str, Any]]:
        """Recursively fetch all descendant pages (full hierarchy).

        Args:
            page_id: Parent page ID

        Returns:
            List of all descendant pages (flattened)

        Note:
            This performs recursive fetching with no depth limit.
        """
        all_children = []

        # Fetch immediate children
        endpoint = f"/content/{page_id}/child/page"
        params = {"limit": 100}

        start = 0
        while True:
            params["start"] = start
            response = self._make_request("GET", endpoint, params=params)

            children = response.get("results", [])
            all_children.extend(children)

            # Check pagination
            size = len(children)
            if size < params["limit"]:
                break
            start += size

        # Recursively fetch descendants of each child
        for child in all_children.copy():
            child_id = child["id"]
            descendants = self.get_page_children_recursive(child_id)
            all_children.extend(descendants)

        return all_children

    def get_page_comments(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetch inline comments for a page.

        Args:
            page_id: Page ID

        Returns:
            List of comment dicts

        Note:
            Returns page-level comments (not inline/highlight comments).
        """
        endpoint = f"/content/{page_id}/child/comment"
        params = {"limit": 100, "expand": "body.storage,version"}

        all_comments = []
        start = 0

        while True:
            params["start"] = start
            response = self._make_request("GET", endpoint, params=params)

            comments = response.get("results", [])
            all_comments.extend(comments)

            # Check pagination
            size = len(comments)
            if size < params["limit"]:
                break
            start += size

        return all_comments

    def convert_storage_to_markdown(self, storage_content: str) -> str:
        """Convert Confluence storage format (XHTML) to Markdown.

        Args:
            storage_content: Storage format content (XHTML string)

        Returns:
            Markdown text

        Note:
            This is a basic fallback implementation.
            For production, consider using a dedicated XHTML-to-Markdown converter.
        """
        if not storage_content:
            return ""

        # Use fallback plain text extraction
        return self._extract_text_from_storage(storage_content)

    def _extract_text_from_storage(self, storage_content: str) -> str:
        """Extract plain text from Confluence storage format as fallback.

        Args:
            storage_content: Storage format content (XHTML string)

        Returns:
            Plain text extracted from storage format

        Note:
            This is a simple HTML tag stripper. For better formatting,
            consider using BeautifulSoup or markdownify library.
        """
        if not storage_content:
            return ""

        try:
            # Simple HTML tag removal for fallback
            import re

            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", storage_content)
            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception as e:
            print(f"[ConfluenceClient] Error extracting text from storage format: {e}")
            return storage_content  # Return raw content as last resort
