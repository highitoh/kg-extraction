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
        - 429 Rate Limit, 5xx, timeouts, connection errors: Retry with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        max_concurrency: int = 8,
    ):
        """Initialize Miro API client.

        Args:
            api_key: Miro API access token (defaults to MIRO_API_KEY env var)
            base_url: Miro API base URL (defaults to MIRO_BASE_URL env var or https://api.miro.com/v2)
            max_retries: Maximum number of retries for rate limiting (default: 3)
            timeout: Request timeout in seconds (default: 30.0)
            max_concurrency: Maximum concurrent requests when callers fan out (default: 8)

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
        self.max_concurrency = max_concurrency
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
        )

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
                        "Invalid or missing MIRO_API_KEY. Check your access token at https://developers.miro.com/"
                    )
                elif response.status_code == 403:
                    raise ValueError("No access to requested resource. Check board permissions and API token scopes.")
                elif response.status_code == 404:
                    board_id = endpoint.split("/")[2] if "/boards/" in endpoint else "unknown"
                    raise ValueError(f"Board or resource not found: {board_id}")
                elif response.status_code == 429 or response.status_code >= 500:
                    # Rate limit or transient server error - retry with exponential backoff
                    if attempt < self.max_retries:
                        header = response.headers.get("Retry-After", "")
                        retry_after = int(header) if header.isdigit() else 2**attempt
                        jitter = 0.1 * retry_after  # 10% jitter
                        sleep_time = retry_after + jitter
                        print(
                            f"Miro API returned {response.status_code}. Retrying in {sleep_time:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    print(f"{type(e).__name__}. Retrying (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2**attempt)
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

    def get_board_items(
        self, board_id: str, limit: int = 50, parent_item_id: str = None, item_type: str = None
    ) -> list[dict]:
        """Fetch items from a board.

        Args:
            board_id: Miro board ID
            limit: Items per page (default: 50, max: 50)
            parent_item_id: If specified, fetch only direct children of this item
            item_type: If specified, fetch only items of this type (e.g., "frame", "diagram")

        Returns:
            List of board items. Each item includes:
            - id: Item ID
            - type: Item type (sticky_note, card, shape, text, connector, frame, etc.)
            - data: Item-specific data (content, position, style, etc.)
            - position: {x, y} coordinates
            - geometry: {width, height} dimensions

        Example:
            >>> frames = client.get_board_items("o9J_abc123", item_type="frame")
            >>> children = client.get_board_items("o9J_abc123", parent_item_id="frame_id")
        """
        endpoint = f"/boards/{board_id}/items"

        all_items = []
        cursor = None

        while True:
            params = {"limit": min(limit, 50)}
            if cursor:
                params["cursor"] = cursor
            if parent_item_id:
                params["parent_item_id"] = parent_item_id
            if item_type:
                params["type"] = item_type

            response = self._make_request("GET", endpoint, params=params)
            items = response.get("data", [])
            all_items.extend(items)

            cursor = response.get("cursor")
            if not cursor:
                break

        return all_items

    def get_connectors(self, board_id: str, limit: int = 50) -> list[dict]:
        """Fetch connectors (arrows/lines) from a board.

        Connectors are not included in get_board_items() - they live on a
        separate endpoint.

        Args:
            board_id: Miro board ID
            limit: Connectors per page (default: 50, max: 50)

        Returns:
            List of connectors. Each connector includes:
            - id: Connector ID
            - startItem / endItem: {"id": ...} of the connected items
            - captions: List of {"content": ...} labels (optional)
            - shape: Line shape (straight, curved, elbowed)
            - style: Stroke color/width/style and arrowhead caps

        Example:
            >>> connectors = client.get_connectors("o9J_abc123")
            >>> print(connectors[0]["startItem"]["id"], connectors[0]["endItem"]["id"])
        """
        endpoint = f"/boards/{board_id}/connectors"

        all_connectors = []
        cursor = None

        while True:
            params = {"limit": min(limit, 50)}
            if cursor:
                params["cursor"] = cursor

            response = self._make_request("GET", endpoint, params=params)
            all_connectors.extend(response.get("data", []))

            cursor = response.get("cursor")
            if not cursor:
                break

        return all_connectors

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
