"""Project profile: what a project's own knowledge is about.

A profile is optional and sits at {output_dir}/{project}/project_profile.json.
When a project has one, the relevance screener keeps the content about the
project and drops the content about other organizations or projects.
"""

import json
import os
from typing import Any, Dict, Optional

import jsonschema

PROFILE_FILENAME = "project_profile.json"

# Go up 3 levels from src/utils/ to /workspace/app/
_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SCHEMA_PATH = os.path.join(_APP_DIR, "schemas", "project-profile", "profile.schema.json")


def profile_path(output_dir: str, project: str) -> str:
    """Path of the project's profile file."""
    return os.path.join(output_dir, project, PROFILE_FILENAME)


def load_project_profile(output_dir: str, project: str) -> Optional[Dict[str, Any]]:
    """Load the project's profile.

    Returns:
        The profile, or None when the project has none

    Raises:
        jsonschema.ValidationError: The file does not match profile.schema.json
    """
    path = profile_path(output_dir, project)
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(profile, schema)
    return profile


def render_profile(profile: Dict[str, Any]) -> str:
    """Render a profile as prompt text."""
    lines = [f"Description: {profile['description']}", "Products:"]
    products = profile.get("products") or []
    for product in products:
        aliases = product.get("aliases") or []
        alias_text = f" (also called: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {product['name']}{alias_text}")
    if not products:
        lines.append("- (none named)")
    return "\n".join(lines)
