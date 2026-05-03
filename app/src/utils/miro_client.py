"""Miro API client for fetching board data."""

import os
import time
from typing import Optional
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()


class MiroClient:
    """Client for interacting with Miro REST API v2.

    Authentication:
        Uses OAuth 2.0 access token from MIRO_API_KEY environment variable.
        User must create a Miro app and generate access token via Miro Developer Portal.

    Rate Limiting:
        Implements exponential backoff with jitter for 429 responses.

    Error Handling:
        - 401 Unauthorized: Invalid or missing API key
        - 403 Forbidden: No access to requested board
        - 404 Not Found: Board or item not found
        - 429 Rate Limit: Retry with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """Initialize Miro API client.

        Args:
            api_key: Miro API access token (defaults to MIRO_API_KEY env var)
            base_url: Miro API base URL (defaults to MIRO_BASE_URL env var or https://api.miro.com/v2)
            max_retries: Maximum number of retries for rate limiting (default: 3)
            timeout: Request timeout in seconds (default: 30.0)

        Raises:
            ValueError: If API key is not provided and MIRO_API_KEY is not set
        """
        self.api_key = api_key or os.getenv("MIRO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MIRO_API_KEY must be set in environment or passed as api_key parameter. "
                "Obtain an access token from https://developers.miro.com/"
            )

        self.base_url = base_url or os.getenv("MIRO_BASE_URL", "https://api.miro.com/v2")
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict:
        """Make HTTP request to Miro API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/boards/{board_id}")
            **kwargs: Additional arguments passed to requests.request()

        Returns:
            JSON response as dictionary

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
                        "Invalid or missing MIRO_API_KEY. "
                        "Check your access token at https://developers.miro.com/"
                    )
                elif response.status_code == 403:
                    raise ValueError(
                        f"No access to requested resource. Check board permissions and API token scopes."
                    )
                elif response.status_code == 404:
                    board_id = endpoint.split("/")[2] if "/boards/" in endpoint else "unknown"
                    raise ValueError(f"Board or resource not found: {board_id}")
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
                return response.json()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    print(f"Request timeout. Retrying (attempt {attempt + 1}/{self.max_retries})")
                    continue
                raise
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                raise

        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded")

    def get_board(self, board_id: str) -> dict:
        """Fetch board metadata.

        Args:
            board_id: Miro board ID (e.g., "o9J_abc123")

        Returns:
            Board metadata including:
            - id: Board ID
            - name: Board name
            - description: Board description (optional)
            - createdAt: Creation timestamp
            - modifiedAt: Last modification timestamp

        Example:
            >>> client = MiroClient()
            >>> board = client.get_board("o9J_abc123")
            >>> print(board["name"])
            "My Knowledge Board"
        """
        endpoint = f"/boards/{board_id}"
        return self._make_request("GET", endpoint)

    def get_board_items(self, board_id: str, limit: int = 50) -> list[dict]:
        """Fetch all items from a board.

        Args:
            board_id: Miro board ID
            limit: Items per page (default: 50, max: 50)

        Returns:
            List of board items. Each item includes:
            - id: Item ID
            - type: Item type (sticky_note, card, shape, text, connector, frame, etc.)
            - data: Item-specific data (content, position, style, etc.)
            - position: {x, y} coordinates
            - geometry: {width, height} dimensions

        Example:
            >>> items = client.get_board_items("o9J_abc123")
            >>> sticky_notes = [i for i in items if i["type"] == "sticky_note"]
        """
        endpoint = f"/boards/{board_id}/items"

        all_items = []
        cursor = None

        while True:
            params = {"limit": min(limit, 50)}
            if cursor:
                params["cursor"] = cursor

            response = self._make_request("GET", endpoint, params=params)
            items = response.get("data", [])
            all_items.extend(items)

            cursor = response.get("cursor")
            if not cursor:
                break

        return all_items

    def get_item(self, board_id: str, item_id: str) -> dict:
        """Fetch a specific item from a board.

        Args:
            board_id: Miro board ID
            item_id: Item ID

        Returns:
            Item data (same structure as get_board_items)

        Example:
            >>> item = client.get_item("o9J_abc123", "item_xyz")
            >>> print(item["type"], item["data"]["content"])
            "sticky_note" "Customer needs faster service"
        """
        endpoint = f"/boards/{board_id}/items/{item_id}"
        return self._make_request("GET", endpoint)
