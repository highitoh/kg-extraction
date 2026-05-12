"""GitHub API client for fetching repository data."""

import os
import time
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()


class GitHubClient:
    """Client for interacting with GitHub REST API v3.

    Authentication:
        Uses Personal Access Token from GITHUB_TOKEN environment variable.
        Token is optional for public repositories (60 req/hr limit).
        With token: 5,000 req/hr limit.

    Rate Limiting:
        Implements exponential backoff with jitter for 429 responses.
        Respects X-RateLimit-Remaining and Retry-After headers.

    Error Handling:
        - 401 Unauthorized: Invalid or missing token
        - 403 Forbidden: No access or rate limit exceeded
        - 404 Not Found: Repository, file, or resource not found
        - 429 Rate Limit: Retry with exponential backoff
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """Initialize GitHub API client.

        Args:
            access_token: GitHub personal access token (defaults to GITHUB_TOKEN env var)
            base_url: GitHub API base URL (defaults to GITHUB_API_BASE_URL env var or https://api.github.com)
            max_retries: Maximum number of retries for rate limiting (default: 3)
            timeout: Request timeout in seconds (default: 30.0)

        Note:
            Token is optional. Without token: 60 req/hr. With token: 5,000 req/hr.
        """
        self.access_token = access_token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url or os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

        # Set headers
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "KnowledgeExtractor/1.0"
        }

        # Add authentication if token is provided
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        self.session.headers.update(headers)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Any:
        """Make HTTP request to GitHub API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/repos/{owner}/{repo}")
            **kwargs: Additional arguments passed to requests.request()

        Returns:
            JSON response as dictionary, or raw response for non-JSON content

        Raises:
            requests.HTTPError: For 4xx/5xx responses (except 429, which is retried)
            ValueError: For authentication/permission errors with clear messages
        """
        # Ensure base_url ends with / and endpoint doesn't start with / for proper joining
        base = self.base_url.rstrip('/') + '/'
        endpoint_clean = endpoint.lstrip('/')
        url = urljoin(base, endpoint_clean)
        kwargs.setdefault("timeout", self.timeout)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                # Handle specific error codes with clear messages
                if response.status_code == 401:
                    raise ValueError(
                        "Invalid or missing GITHUB_TOKEN. "
                        "Create a personal access token at https://github.com/settings/tokens"
                    )
                elif response.status_code == 403:
                    # Check if it's rate limit or permission issue
                    remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
                    if remaining == "0":
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        if reset_time:
                            wait_time = reset_time - time.time()
                            raise ValueError(
                                f"Rate limit exceeded. Resets in {wait_time/60:.1f} minutes. "
                                f"Use GITHUB_TOKEN for higher limits (5,000/hr vs 60/hr)."
                            )
                    raise ValueError(
                        f"No access to requested resource. Check repository permissions and token scopes."
                    )
                elif response.status_code == 404:
                    # Extract meaningful info from endpoint for better error message
                    parts = endpoint.split("/")
                    resource = "/".join(parts[:4]) if len(parts) >= 4 else endpoint
                    raise ValueError(f"Resource not found: {resource}")
                elif response.status_code == 429:
                    # Rate limit - retry with exponential backoff
                    if attempt < self.max_retries:
                        retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                        jitter = 0.1 * retry_after  # 10% jitter
                        sleep_time = retry_after + jitter
                        print(f"Rate limited (429). Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(sleep_time)
                        continue
                    else:
                        response.raise_for_status()

                response.raise_for_status()

                # Return JSON if content-type is JSON
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return response.json()
                else:
                    # Return text for raw content
                    return response.text

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    print(f"Request timeout. Retrying (attempt {attempt + 1}/{self.max_retries})")
                    continue
                raise
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                raise

        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded")

    def check_rate_limit(self) -> Dict[str, Any]:
        """Check current rate limit status.

        Returns:
            Rate limit information including:
            - limit: Maximum requests per hour
            - remaining: Remaining requests in current window
            - reset: Unix timestamp when limit resets
            - used: Number of requests used

        Example:
            >>> client = GitHubClient()
            >>> limits = client.check_rate_limit()
            >>> print(f"Remaining: {limits['resources']['core']['remaining']}")
        """
        return self._make_request("GET", "/rate_limit")

    def get_repo(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetch repository metadata.

        Args:
            owner: Repository owner (user or organization)
            repo: Repository name

        Returns:
            Repository metadata including:
            - id: Repository ID
            - name: Repository name
            - full_name: owner/repo
            - description: Repository description
            - default_branch: Default branch name (usually "main" or "master")
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - pushed_at: Last push timestamp
            - size: Repository size in KB
            - stargazers_count: Number of stars
            - language: Primary language

        Example:
            >>> client = GitHubClient()
            >>> repo = client.get_repo("facebook", "react")
            >>> print(repo['description'])
        """
        return self._make_request("GET", f"/repos/{owner}/{repo}")

    def get_readme(self, owner: str, repo: str, ref: str = None) -> Dict[str, Any]:
        """Fetch repository README file.

        Args:
            owner: Repository owner
            repo: Repository name
            ref: Git ref (branch, tag, or commit SHA). Defaults to default branch.

        Returns:
            README file metadata and content:
            - name: File name (e.g., "README.md")
            - path: File path
            - sha: File SHA hash
            - size: File size in bytes
            - content: Base64-encoded content
            - encoding: "base64"

        Example:
            >>> client = GitHubClient()
            >>> readme = client.get_readme("facebook", "react")
            >>> content = base64.b64decode(readme['content']).decode('utf-8')
        """
        endpoint = f"/repos/{owner}/{repo}/readme"
        params = {}
        if ref:
            params["ref"] = ref

        return self._make_request("GET", endpoint, params=params)

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = None) -> str:
        """Fetch raw file content.

        Args:
            owner: Repository owner
            repo: Repository name
            path: File path (e.g., "docs/getting-started.md")
            ref: Git ref (branch, tag, or commit SHA). Defaults to default branch.

        Returns:
            Raw file content as string

        Example:
            >>> client = GitHubClient()
            >>> content = client.get_file_content("facebook", "react", "README.md")
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        # Request raw content
        headers = {"Accept": "application/vnd.github.v3.raw"}
        return self._make_request("GET", endpoint, params=params, headers=headers)

    def get_file_metadata(self, owner: str, repo: str, path: str, ref: str = None) -> Dict[str, Any]:
        """Fetch file metadata without content.

        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Git ref (branch, tag, or commit SHA)

        Returns:
            File metadata including:
            - name: File name
            - path: Full path
            - sha: File SHA hash
            - size: File size in bytes
            - type: "file" or "dir"
            - url: API URL

        Example:
            >>> client = GitHubClient()
            >>> meta = client.get_file_metadata("facebook", "react", "README.md")
            >>> print(meta['sha'])
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        return self._make_request("GET", endpoint, params=params)

    def list_directory(self, owner: str, repo: str, path: str = "", ref: str = None) -> List[Dict[str, Any]]:
        """List contents of a directory.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty string for root)
            ref: Git ref (branch, tag, or commit SHA)

        Returns:
            List of file/directory metadata dicts:
            - name: File/directory name
            - path: Full path
            - type: "file" or "dir"
            - sha: Content SHA
            - size: Size in bytes (0 for directories)
            - url: API URL

        Example:
            >>> client = GitHubClient()
            >>> files = client.list_directory("facebook", "react", "docs")
            >>> md_files = [f for f in files if f['name'].endswith('.md')]
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}" if path else f"/repos/{owner}/{repo}/contents"
        params = {}
        if ref:
            params["ref"] = ref

        result = self._make_request("GET", endpoint, params=params)

        # Ensure result is a list (directory), not a single file
        if isinstance(result, dict):
            raise ValueError(f"Path '{path}' is a file, not a directory")

        return result

    def list_files_recursive(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = None,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Recursively list all files in a directory.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty string for root)
            ref: Git ref (branch, tag, or commit SHA)
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            List of file metadata dicts (directories excluded)

        Example:
            >>> client = GitHubClient()
            >>> all_files = client.list_files_recursive("facebook", "react", "docs")
            >>> md_count = len([f for f in all_files if f['name'].endswith('.md')])
        """
        if max_depth <= 0:
            return []

        all_files = []

        try:
            items = self.list_directory(owner, repo, path, ref)
        except ValueError:
            # Path is a file, not a directory
            return []

        for item in items:
            if item['type'] == 'file':
                all_files.append(item)
            elif item['type'] == 'dir':
                # Recursively list subdirectory
                sub_files = self.list_files_recursive(
                    owner, repo, item['path'], ref, max_depth - 1
                )
                all_files.extend(sub_files)

        return all_files

    def get_commit(self, owner: str, repo: str, ref: str) -> Dict[str, Any]:
        """Fetch commit information.

        Args:
            owner: Repository owner
            repo: Repository name
            ref: Git ref (branch, tag, or commit SHA)

        Returns:
            Commit metadata including:
            - sha: Commit SHA
            - commit: Commit details (message, author, etc.)
            - author: Author information
            - committer: Committer information

        Example:
            >>> client = GitHubClient()
            >>> commit = client.get_commit("facebook", "react", "main")
            >>> print(commit['sha'])
        """
        return self._make_request("GET", f"/repos/{owner}/{repo}/commits/{ref}")

    def compare_commits(self, owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
        """Compare two commits or branches.

        Args:
            owner: Repository owner
            repo: Repository name
            base: Base branch, tag, or commit SHA
            head: Head branch, tag, or commit SHA

        Returns:
            Comparison data including:
            - status: "ahead", "behind", "identical", or "diverged"
            - ahead_by: Number of commits ahead
            - behind_by: Number of commits behind
            - total_commits: Total commits in comparison
            - commits: List of commit objects
            - files: List of changed files with diffs

        Example:
            >>> client = GitHubClient()
            >>> diff = client.compare_commits("facebook", "react", "v18.0.0", "v18.1.0")
            >>> print(f"Changed files: {len(diff['files'])}")
        """
        return self._make_request("GET", f"/repos/{owner}/{repo}/compare/{base}...{head}")

    def get_releases(self, owner: str, repo: str, per_page: int = 10) -> List[Dict[str, Any]]:
        """Fetch repository releases.

        Args:
            owner: Repository owner
            repo: Repository name
            per_page: Number of releases to fetch (max 100)

        Returns:
            List of release objects including:
            - id: Release ID
            - tag_name: Git tag
            - name: Release name
            - body: Release notes (markdown)
            - published_at: Publication timestamp
            - author: Release author

        Example:
            >>> client = GitHubClient()
            >>> releases = client.get_releases("facebook", "react", per_page=5)
            >>> latest = releases[0]
            >>> print(latest['tag_name'])
        """
        params = {"per_page": min(per_page, 100)}
        return self._make_request("GET", f"/repos/{owner}/{repo}/releases", params=params)

    def get_latest_release(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetch the latest release.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Latest release object

        Example:
            >>> client = GitHubClient()
            >>> latest = client.get_latest_release("facebook", "react")
            >>> print(latest['body'])  # Release notes
        """
        return self._make_request("GET", f"/repos/{owner}/{repo}/releases/latest")

    def get_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch repository issues.

        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state ("open", "closed", or "all")
            per_page: Number of issues to fetch (max 100)

        Returns:
            List of issue objects including:
            - number: Issue number
            - title: Issue title
            - body: Issue description (markdown)
            - state: "open" or "closed"
            - user: Issue author
            - labels: List of labels
            - created_at: Creation timestamp

        Example:
            >>> client = GitHubClient()
            >>> issues = client.get_issues("facebook", "react", state="open", per_page=50)
        """
        params = {
            "state": state,
            "per_page": min(per_page, 100)
        }
        return self._make_request("GET", f"/repos/{owner}/{repo}/issues", params=params)

    def get_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch name for a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Default branch name (e.g., "main" or "master")

        Example:
            >>> client = GitHubClient()
            >>> branch = client.get_default_branch("facebook", "react")
            >>> print(branch)  # "main"
        """
        repo_data = self.get_repo(owner, repo)
        return repo_data.get("default_branch", "main")

    def get_commit_at_date(
        self,
        owner: str,
        repo: str,
        date: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find the most recent commit on or before a specific date.

        Args:
            owner: Repository owner
            repo: Repository name
            date: ISO 8601 date string (e.g., "2023-01-15", "2023-01-15T10:30:00Z")
            branch: Branch name to search (defaults to default branch)

        Returns:
            Commit metadata including:
            - sha: Commit SHA
            - commit: Commit details (message, author, date, etc.)
            - author: Author information
            - committer: Committer information

        Raises:
            ValueError: If no commit found before the specified date

        Example:
            >>> client = GitHubClient()
            >>> commit = client.get_commit_at_date("facebook", "react", "2023-01-15")
            >>> print(commit['sha'])
        """
        # Get default branch if not specified
        if not branch:
            branch = self.get_default_branch(owner, repo)

        # Fetch commits until the specified date
        # GitHub API returns commits in reverse chronological order
        params = {
            "sha": branch,
            "until": date,
            "per_page": 1
        }

        commits = self._make_request("GET", f"/repos/{owner}/{repo}/commits", params=params)

        if not commits:
            raise ValueError(
                f"No commits found on branch '{branch}' before date '{date}'. "
                f"Repository may not have existed at that time."
            )

        # Return the most recent commit before the date
        return commits[0]
