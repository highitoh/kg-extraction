"""Analyze and transform GitHub API data into structured extraction items."""

import fnmatch
from typing import Dict, Any, List, Optional


class GitHubAnalyzer:
    """Analyzes GitHub repository data and applies profile-based filtering.

    Transforms raw GitHub API responses into structured items ready for
    text extraction. Applies include/exclude patterns, categorizes items
    by type, and filters based on profile configuration.
    """

    SUPPORTED_TYPES = {"readme", "doc", "issue", "wiki"}

    def __init__(self, profile: Dict[str, Any]):
        """Initialize GitHub Analyzer.

        Args:
            profile: Extraction profile configuration with:
                - include_patterns: List of glob patterns to include
                - exclude_patterns: List of glob patterns to exclude
                - include_wiki: Whether to include wiki pages
                - include_issues: Whether to include issues

        Examples:
            >>> from extract.github.github_profiles import get_profile
            >>> profile = get_profile("documentation")
            >>> analyzer = GitHubAnalyzer(profile)
        """
        self.profile = profile
        self.include_patterns = profile.get("include_patterns", [])
        self.exclude_patterns = profile.get("exclude_patterns", [])
        self.include_wiki = profile.get("include_wiki", False)
        self.include_issues = profile.get("include_issues", False)
        self.max_issues = profile.get("max_issues", 100)

    def analyze(self, raw_repo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze raw repository data and extract relevant items.

        Args:
            raw_repo_data: Dictionary containing:
                - files: List of file objects from GitHub API
                - issues: List of issue objects (optional)
                - releases: List of release objects (optional)
                - wiki_pages: List of wiki page objects (optional)

        Returns:
            List of analyzed item dictionaries

        Examples:
            >>> data = {"files": [...], "issues": [...]}
            >>> items = analyzer.analyze(data)
        """
        analyzed_items = []

        # Analyze files
        files = raw_repo_data.get("files", [])
        for file_data in files:
            item = self._analyze_file(file_data)
            if item:
                analyzed_items.append(item)

        # Analyze issues if enabled
        if self.include_issues:
            issues = raw_repo_data.get("issues", [])
            for issue_data in issues[: self.max_issues]:
                item = self._analyze_issue(issue_data)
                if item:
                    analyzed_items.append(item)

        # Analyze wiki pages if enabled
        if self.include_wiki:
            wiki_pages = raw_repo_data.get("wiki_pages", [])
            for wiki_data in wiki_pages:
                item = self._analyze_wiki(wiki_data)
                if item:
                    analyzed_items.append(item)

        return analyzed_items

    def _analyze_file(self, file_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a file and determine if it should be extracted.

        Args:
            file_data: File object from GitHub API with:
                - name: File name
                - path: Full path
                - sha: File SHA
                - size: File size in bytes
                - type: "file" or "dir"
                - url: API URL
                - content: File content (if fetched)

        Returns:
            Analyzed item dict or None if file should be excluded

        Examples:
            >>> file_data = {"path": "docs/api.md", "sha": "abc123", ...}
            >>> item = analyzer._analyze_file(file_data)
        """
        path = file_data.get("path", "")
        file_type = file_data.get("type", "file")

        # Skip directories
        if file_type != "file":
            return None

        # Apply pattern matching
        if not self._matches_pattern(path):
            return None

        # Determine item type
        item_type = self._categorize_file(path)

        return {
            "id": file_data.get("sha", ""),
            "type": item_type,
            "path": path,
            "title": file_data.get("name", path.split("/")[-1]),
            "content": file_data.get("content", ""),
            "metadata": {
                "source": "github",
                "item_type": item_type,
                "sha": file_data.get("sha", ""),
                "size": file_data.get("size", 0),
                "url": file_data.get("html_url", ""),
                "download_url": file_data.get("download_url", ""),
            },
        }

    def _analyze_issue(self, issue_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze an issue.

        Args:
            issue_data: Issue object from GitHub API

        Returns:
            Analyzed item dict or None if issue should be excluded
        """
        # Skip pull requests (they appear in issues API)
        if "pull_request" in issue_data:
            return None

        issue_number = issue_data.get("number", 0)
        title = issue_data.get("title", "")
        body = issue_data.get("body", "")

        # Combine title and body for content
        content = f"# {title}\n\n{body}" if body else f"# {title}"

        return {
            "id": f"issue-{issue_number}",
            "type": "issue",
            "path": f"issues/{issue_number}",
            "title": title,
            "content": content,
            "metadata": {
                "source": "github",
                "item_type": "issue",
                "issue_number": issue_number,
                "state": issue_data.get("state", ""),
                "author": issue_data.get("user", {}).get("login", ""),
                "created_at": issue_data.get("created_at", ""),
                "updated_at": issue_data.get("updated_at", ""),
                "url": issue_data.get("html_url", ""),
                "labels": [label.get("name") for label in issue_data.get("labels", [])],
            },
        }

    def _analyze_wiki(self, wiki_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a wiki page.

        Args:
            wiki_data: Wiki page data (from cloned wiki repo)

        Returns:
            Analyzed item dict or None if wiki page should be excluded
        """
        path = wiki_data.get("path", "")
        title = wiki_data.get("title", path)
        content = wiki_data.get("content", "")

        return {
            "id": f"wiki-{path}",
            "type": "wiki",
            "path": f"wiki/{path}",
            "title": title,
            "content": content,
            "metadata": {"source": "github", "item_type": "wiki", "url": wiki_data.get("url", "")},
        }

    def _matches_pattern(self, path: str) -> bool:
        """Check if file path matches include/exclude patterns.

        Args:
            path: File path to check

        Returns:
            True if path should be included, False otherwise

        Examples:
            >>> analyzer._matches_pattern("docs/api.md")
            True
            >>> analyzer._matches_pattern("node_modules/package/file.md")
            False
        """
        # Check exclude patterns first
        for exclude_pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, exclude_pattern):
                return False

        # Check include patterns
        if not self.include_patterns:
            return True  # No patterns means include all

        for include_pattern in self.include_patterns:
            if fnmatch.fnmatch(path, include_pattern):
                return True

        return False

    def _categorize_file(self, path: str) -> str:
        """Categorize a file based on its path.

        Args:
            path: File path

        Returns:
            Item type category

        Examples:
            >>> analyzer._categorize_file("README.md")
            'readme'
            >>> analyzer._categorize_file("docs/api.md")
            'doc'
        """
        path_lower = path.lower()
        filename = path.split("/")[-1].lower()

        # README files
        if filename in ["readme.md", "readme.rst", "readme.txt", "readme"]:
            return "readme"

        # Default to doc
        return "doc"

    def to_text_chunks(self, analyzed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert analyzed items to text chunks ready for extraction.

        Args:
            analyzed_items: List of analyzed item dictionaries

        Returns:
            List of text chunks with metadata

        Examples:
            >>> items = analyzer.analyze(raw_data)
            >>> chunks = analyzer.to_text_chunks(items)
        """
        chunks = []

        for item in analyzed_items:
            if not item.get("content"):
                continue

            chunk = {
                "text": item["content"],
                "metadata": {
                    "file_path": item["path"],
                    "item_type": item["type"],
                    "title": item["title"],
                    **item["metadata"],  # Merge in all metadata fields
                },
                "file_id": item["id"],
            }

            chunks.append(chunk)

        return chunks

    def filter_by_file_ids(self, items: List[Dict[str, Any]], file_ids: List[str]) -> List[Dict[str, Any]]:
        """Filter items to only include specific file IDs.

        Used for incremental extraction to process only changed files.

        Args:
            items: List of analyzed items
            file_ids: List of file IDs to include

        Returns:
            Filtered list of items

        Examples:
            >>> filtered = analyzer.filter_by_file_ids(items, ["abc123", "def456"])
        """
        if not file_ids:
            return items

        file_id_set = set(file_ids)
        return [item for item in items if item["id"] in file_id_set]

    def get_file_map(self, analyzed_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create a file map for snapshot tracking.

        Args:
            analyzed_items: List of analyzed items

        Returns:
            Dictionary mapping file_id to file metadata

        Examples:
            >>> file_map = analyzer.get_file_map(items)
            >>> file_map["abc123"]
            {'path': 'README.md', 'sha': 'abc123', ...}
        """
        file_map = {}

        for item in analyzed_items:
            if item["type"] in ["readme", "doc"]:
                file_id = item["id"]
                file_map[file_id] = {
                    "path": item["path"],
                    "sha": item["metadata"].get("sha", ""),
                    "size": item["metadata"].get("size", 0),
                    "type": item["type"],
                }

        return file_map
