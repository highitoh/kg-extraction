"""Parse GitHub URLs and detect extraction strategy."""

import re
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs


def parse_github_url(url: str) -> Dict[str, Optional[str]]:
    """Parse GitHub URL and detect extraction strategy.

    Supported URL formats:
        - https://github.com/owner/repo (auto-detect default branch)
        - https://github.com/owner/repo/tree/ref (root directory, explicit branch)
        - https://github.com/owner/repo/tree/ref/path/to/dir (specific directory)
        - https://github.com/owner/repo/tree/ref/path#profile=profile-name (with profile)

    Args:
        url: GitHub URL

    Returns:
        Parsed URL components:
        {
            "owner": str,
            "repo": str,
            "path": str,                     # Directory path (empty string for root)
            "ref": str,                      # Branch, tag, or commit (or None for auto-detect)
            "profile": str,                  # Detected or overridden profile name
            "url_type": str,                 # "tree" or "root"
            "original_url": str              # Original input URL
        }

    Raises:
        ValueError: If URL format is invalid or cannot be parsed

    Examples:
        >>> parse_github_url("https://github.com/facebook/react")
        {'owner': 'facebook', 'repo': 'react', 'path': '', 'ref': None, 'url_type': 'root', ...}

        >>> parse_github_url("https://github.com/facebook/react/tree/main")
        {'owner': 'facebook', 'repo': 'react', 'path': '', 'ref': 'main', 'url_type': 'tree', ...}

        >>> parse_github_url("https://github.com/facebook/react/tree/main/packages")
        {'owner': 'facebook', 'repo': 'react', 'path': 'packages', 'ref': 'main', 'url_type': 'tree', ...}
    """
    original_url = url

    # Parse GitHub URL
    parsed = urlparse(url)

    if parsed.netloc not in ["github.com", "www.github.com"]:
        raise ValueError(f"Invalid GitHub URL: {url}. Expected github.com domain.")

    # Extract fragment for profile override
    profile_override = _extract_profile_override(url)

    # Split path into components
    # Example: /facebook/react/blob/main/README.md -> ['', 'facebook', 'react', 'blob', 'main', 'README.md']
    path_parts = parsed.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}. Expected format: github.com/owner/repo[/...]")

    owner = path_parts[0]
    repo = path_parts[1]

    # Determine URL type and extract relevant parts
    if len(path_parts) == 2:
        # Root repository URL: https://github.com/owner/repo
        # Will auto-detect default branch during extraction
        return {
            "owner": owner,
            "repo": repo,
            "path": "",  # Root directory
            "ref": None,  # Auto-detect default branch
            "profile": profile_override or "documentation",
            "url_type": "root",
            "original_url": original_url,
        }

    elif len(path_parts) >= 3 and path_parts[2] == "tree":
        # Directory URL: https://github.com/owner/repo/tree/ref[/path/to/dir]
        if len(path_parts) < 4:
            raise ValueError(
                f"Invalid tree URL: {url}\n"
                f"Expected format: https://github.com/{{owner}}/{{repo}}/tree/{{ref}}[/{{path}}]\n"
                f"Examples:\n"
                f"  - https://github.com/facebook/react/tree/main (root directory)\n"
                f"  - https://github.com/facebook/react/tree/main/packages\n"
                f"  - https://github.com/nodejs/node/tree/main/doc/api"
            )

        ref = path_parts[3]
        # Path is optional - if not provided, extract from root directory
        dir_path = "/".join(path_parts[4:]) if len(path_parts) > 4 else ""

        # Determine profile based on directory
        profile = profile_override or _detect_profile_from_path(dir_path) if dir_path else "documentation"

        return {
            "owner": owner,
            "repo": repo,
            "path": dir_path,
            "ref": ref,
            "profile": profile,
            "url_type": "tree",
            "original_url": original_url,
        }

    else:
        # Unsupported URL format
        raise ValueError(
            f"Unsupported GitHub URL format.\n"
            f"Supported formats:\n"
            f"  - https://github.com/{{owner}}/{{repo}} (auto-detect default branch)\n"
            f"  - https://github.com/{{owner}}/{{repo}}/tree/{{ref}}[/{{path}}]\n"
            f"Examples:\n"
            f"  - https://github.com/facebook/react\n"
            f"  - https://github.com/facebook/react/tree/main\n"
            f"  - https://github.com/facebook/react/tree/main/packages\n"
            f"Got: {url}"
        )


def _extract_profile_override(url: str) -> Optional[str]:
    """Extract profile override from URL fragment.

    Args:
        url: URL with optional fragment like #profile=full-knowledge

    Returns:
        Profile name if specified, None otherwise

    Examples:
        >>> _extract_profile_override("https://github.com/owner/repo#profile=full-knowledge")
        'full-knowledge'
        >>> _extract_profile_override("https://github.com/owner/repo")
        None
    """
    # Extract fragment
    if "#profile=" not in url:
        return None

    fragment = url.split("#")[1]
    if fragment.startswith("profile="):
        return fragment.replace("profile=", "")

    return None


def _detect_profile_from_path(path: Optional[str]) -> str:
    """Detect extraction profile from file or directory path.

    Args:
        path: File or directory path

    Returns:
        Profile name (always returns "documentation")

    Examples:
        >>> _detect_profile_from_path("README.md")
        'documentation'
        >>> _detect_profile_from_path("docs/api.md")
        'documentation'
    """
    # Always return documentation profile
    # Profile-specific extraction is no longer needed for knowledge extraction use case
    return "documentation"


def is_github_url(url: str) -> bool:
    """Check if a URL is a GitHub repository URL.

    Args:
        url: URL to check

    Returns:
        True if URL is a GitHub repository URL, False otherwise

    Examples:
        >>> is_github_url("https://github.com/facebook/react")
        True
        >>> is_github_url("github:facebook/react")
        True
        >>> is_github_url("https://gitlab.com/some/repo")
        False
    """
    # Check shorthand formats
    if url.startswith("github:") or url.startswith("gh:"):
        return True

    # Check standard GitHub URLs
    return "github.com/" in url


def get_repo_identifier(owner: str, repo: str) -> str:
    """Generate a stable repository identifier.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Repository identifier in format "github:owner:repo"

    Examples:
        >>> get_repo_identifier("facebook", "react")
        'github:facebook:react'
    """
    return f"github:{owner}:{repo}"


def get_file_identifier(owner: str, repo: str, path: str, sha: str) -> str:
    """Generate a stable file identifier for incremental tracking.

    Args:
        owner: Repository owner
        repo: Repository name
        path: File path
        sha: File content SHA

    Returns:
        File identifier in format "github:owner:repo:path:sha"

    Examples:
        >>> get_file_identifier("facebook", "react", "README.md", "abc123")
        'github:facebook:react:README.md:abc123'
    """
    return f"github:{owner}:{repo}:{path}:{sha}"
