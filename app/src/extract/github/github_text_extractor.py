"""Extract text content from GitHub repositories."""

from typing import Dict, List, Any, Optional

from src.extract.github.github_analyzer import GitHubAnalyzer
from src.extract.github.github_profiles import get_profile
from src.extract.github.github_url_parser import parse_github_url, get_file_identifier
from src.utils.github_client import GitHubClient


def extract_text_from_github(
    github_url: str, profile_override: Optional[str] = None, date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract text content from GitHub repository.

    Main extraction function that orchestrates:
    1. URL parsing
    2. Profile selection
    3. API calls to fetch content
    4. Analysis and filtering
    5. Structuring for downstream processing

    Args:
        github_url: GitHub repository URL or shorthand
        profile_override: Optional profile name to override URL-detected profile
        date: Optional date in YYYYMMDD format to extract from specific point in time
              (e.g., "20230115" for January 15, 2023)

    Returns:
        List of text chunks with metadata:
        [
            {
                "text": str,
                "metadata": {
                    "file_path": str,
                    "item_type": str,
                    "url": str,
                    "sha": str,
                    "owner": str,
                    "repo": str,
                    "ref": str
                },
                "file_id": str,
                "_github_raw_data": dict  # For snapshot storage
            }
        ]

    Examples:
        >>> chunks = extract_text_from_github("https://github.com/facebook/react")
        >>> len(chunks) > 0
        True
        >>> chunks = extract_text_from_github("https://github.com/facebook/react", date="20230115")
    """
    print(f"[GitHubExtractor] Extracting text from GitHub URL: {github_url}")

    # Parse URL
    parsed_url = parse_github_url(github_url)
    owner = parsed_url["owner"]
    repo = parsed_url["repo"]
    ref = parsed_url["ref"]
    url_type = parsed_url["url_type"]
    profile_name = profile_override or parsed_url["profile"]

    # Get profile
    profile = get_profile(profile_name)

    # Initialize GitHub client
    client = GitHubClient()

    # Convert date from YYYYMMDD to ISO 8601 format if provided
    iso_date = None
    if date:
        if len(date) != 8 or not date.isdigit():
            raise ValueError(f"Invalid date format: {date}. Expected YYYYMMDD (e.g., 20230115)")
        iso_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

    # Get repository metadata (always needed)
    try:
        repo_data = client.get_repo(owner, repo)
        print("[GitHubExtractor] " + f"Repository: {repo_data['full_name']} - {repo_data.get('description', '')}")

        # If date is specified, find commit at that date
        if iso_date:
            print(f"[GitHubExtractor] Finding commit at date: {iso_date}")
            commit = client.get_commit_at_date(owner, repo, iso_date, branch=ref)
            ref = commit["sha"]
            commit_date = commit["commit"]["committer"]["date"]
            print(f"[GitHubExtractor] Using commit {ref[:7]} from {commit_date}")
        # Auto-detect default branch if ref is None (root URL)
        elif ref is None:
            ref = repo_data.get("default_branch", "main")
            print("[GitHubExtractor] " + f"Auto-detected default branch: {ref}")
        # Use default branch if ref is "main" but repo uses different default
        elif ref == "main" and repo_data.get("default_branch") != "main":
            ref = repo_data["default_branch"]
            print("[GitHubExtractor] " + f"Using default branch: {ref}")
        else:
            print("[GitHubExtractor] " + f"Using branch: {ref}")

    except ValueError as e:
        print("[GitHubExtractor] Error: " + f"Failed to fetch repository: {e}")
        raise

    print("[GitHubExtractor] " + f"Profile: {profile_name}")

    # Directory extraction
    dir_path = parsed_url["path"] or ""
    print("[GitHubExtractor] " + f"Fetching directory: {dir_path or '(root)'}")

    raw_files = []
    try:
        if profile.get("recursive", True):
            all_files = client.list_files_recursive(owner, repo, dir_path, ref)
        else:
            all_files = client.list_directory(owner, repo, dir_path, ref)
            all_files = [f for f in all_files if f["type"] == "file"]

        print("[GitHubExtractor] " + f"Found {len(all_files)} files in directory")
        print("[GitHubExtractor] " + "Fetching file contents...")

        # Fetch content for each file
        for idx, file_meta in enumerate(all_files, 1):
            file_path = file_meta["path"]
            try:
                print(f"[GitHubExtractor]   [{idx}/{len(all_files)}] {file_path}")
                content = client.get_file_content(owner, repo, file_path, ref)
                file_meta["content"] = content
                file_meta["html_url"] = f"https://github.com/{owner}/{repo}/blob/{ref}/{file_path}"
                raw_files.append(file_meta)
            except Exception as e:
                print("[GitHubExtractor] Warning: " + f"Failed to fetch {file_path}: {e}")
                continue

    except ValueError as e:
        print("[GitHubExtractor] Error: " + f"Failed to list directory: {e}")
        raise

    # Fetch issues if profile includes them
    issues = []
    if profile.get("include_issues", False):
        max_issues = profile.get("max_issues", 100)
        try:
            issues = client.get_issues(owner, repo, state="all", per_page=max_issues)
            print("[GitHubExtractor] " + f"Fetched {len(issues)} issues")
        except Exception as e:
            print("[GitHubExtractor] Warning: " + f"Failed to fetch issues: {e}")

    # Get current commit SHA for snapshot tracking
    try:
        commit = client.get_commit(owner, repo, ref)
        commit_sha = commit["sha"]
    except Exception as e:
        print("[GitHubExtractor] Warning: " + f"Failed to get commit SHA: {e}")
        commit_sha = ref

    # Analyze and filter items
    analyzer = GitHubAnalyzer(profile)

    raw_repo_data = {
        "files": raw_files,
        "issues": issues,
        "wiki_pages": [],  # Wiki support can be added later
    }

    analyzed_items = analyzer.analyze(raw_repo_data)
    print("[GitHubExtractor] " + f"Analyzed {len(analyzed_items)} items after filtering")

    # Convert to text chunks
    chunks = analyzer.to_text_chunks(analyzed_items)

    # Add standardized metadata to all chunks
    for chunk in chunks:
        chunk["metadata"]["owner"] = owner
        chunk["metadata"]["repo"] = repo
        chunk["metadata"]["ref"] = ref
        chunk["metadata"]["commit_sha"] = commit_sha

        # Generate stable file ID
        path = chunk["metadata"]["file_path"]
        sha = chunk["metadata"].get("sha", chunk["file_id"])
        chunk["file_id"] = get_file_identifier(owner, repo, path, sha)

    # Add raw data for snapshot (first chunk only to avoid duplication)
    if chunks:
        chunks[0]["_github_raw_data"] = {
            "owner": owner,
            "repo": repo,
            "ref": ref,
            "commit_sha": commit_sha,
            "profile": profile_name,
            "url": github_url,
            "file_count": len(raw_files),
            "issue_count": len(issues),
            "file_map": analyzer.get_file_map(analyzed_items),
        }

    print("[GitHubExtractor] " + f"Extracted {len(chunks)} text chunks from {owner}/{repo}")

    return chunks
