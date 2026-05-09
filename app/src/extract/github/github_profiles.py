"""Extraction profiles for different GitHub repository types."""

from typing import Dict, Any, List


# Pre-defined extraction profiles
PROFILES: Dict[str, Dict[str, Any]] = {
    "documentation": {
        "description": "Current documentation (default)",
        "include_patterns": ["**/*.md", "**/*.rst"],
        "exclude_patterns": ["node_modules/**", ".github/**", "vendor/**", ".git/**"],
        "include_wiki": False,
        "include_issues": False,
        "recursive": True
    },

    "full-knowledge": {
        "description": "Documentation + current issues (high API usage)",
        "include_patterns": ["**/*.md", "**/*.rst", "**/*.txt"],
        "exclude_patterns": [
            "node_modules/**",
            "vendor/**",
            ".git/**",
            ".github/workflows/**",
            "test/**",
            "tests/**",
            "__pycache__/**"
        ],
        "include_wiki": True,
        "include_issues": True,
        "max_issues": 100,
        "recursive": True
    }
}


def get_profile(profile_name: str) -> Dict[str, Any]:
    """Get extraction profile by name.

    Args:
        profile_name: Name of the profile (e.g., "documentation", "full-knowledge")

    Returns:
        Profile configuration dictionary

    Raises:
        ValueError: If profile name is not found

    Examples:
        >>> profile = get_profile("documentation")
        >>> print(profile["description"])
        'README + /docs folder (default)'
    """
    if profile_name not in PROFILES:
        raise ValueError(
            f"Unknown profile: {profile_name}. "
            f"Available profiles: {', '.join(PROFILES.keys())}"
        )

    return PROFILES[profile_name].copy()


def get_default_profile() -> Dict[str, Any]:
    """Get the default extraction profile.

    Returns:
        Default profile configuration ("documentation")

    Examples:
        >>> profile = get_default_profile()
        >>> print(profile["description"])
        'README + /docs folder (default)'
    """
    return get_profile("documentation")


def list_profiles() -> List[str]:
    """List all available profile names.

    Returns:
        List of profile names

    Examples:
        >>> profiles = list_profiles()
        >>> 'documentation' in profiles
        True
    """
    return list(PROFILES.keys())


def get_profile_description(profile_name: str) -> str:
    """Get description of a profile.

    Args:
        profile_name: Name of the profile

    Returns:
        Profile description string

    Raises:
        ValueError: If profile name is not found

    Examples:
        >>> desc = get_profile_description("api-docs")
        >>> print(desc)
        'API documentation files'
    """
    profile = get_profile(profile_name)
    return profile["description"]


def merge_profiles(base_profile: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge profile with custom overrides.

    Args:
        base_profile: Name of base profile
        overrides: Dictionary of profile settings to override

    Returns:
        Merged profile configuration

    Examples:
        >>> profile = merge_profiles("documentation", {"include_issues": True})
        >>> profile["include_issues"]
        True
    """
    profile = get_profile(base_profile)
    profile.update(overrides)
    return profile


def validate_profile(profile: Dict[str, Any]) -> bool:
    """Validate a profile configuration.

    Args:
        profile: Profile configuration dictionary

    Returns:
        True if profile is valid

    Raises:
        ValueError: If profile is invalid with explanation

    Examples:
        >>> profile = get_profile("documentation")
        >>> validate_profile(profile)
        True
    """
    required_keys = ["description", "include_patterns", "exclude_patterns"]

    for key in required_keys:
        if key not in profile:
            raise ValueError(f"Profile missing required key: {key}")

    if not isinstance(profile["include_patterns"], list):
        raise ValueError("include_patterns must be a list")

    if not isinstance(profile["exclude_patterns"], list):
        raise ValueError("exclude_patterns must be a list")

    return True


