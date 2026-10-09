import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class SourceCache:
    """Read-through cache of source reads, one JSON file per key under {output_dir}/{project}/cache/.

    Entries never expire; delete the cache directory to read the source again.
    """

    def __init__(self, project: str, output_dir: str = "/workspace/app/output"):
        self.cache_dir = Path(output_dir) / project / "cache"

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}.json"

    def get_or_fetch(self, key: str, fetch: Callable[[], Any]) -> Any:
        """Return the cached data for key, or call fetch and cache what it returns.

        Data round-trips through JSON, so tuples come back as lists.
        """
        path = self._path(key)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            print(f"[SourceCache] Using {key} fetched at {entry['fetched_at']}")
            return entry["data"]

        data = fetch()
        entry = {"key": key, "fetched_at": datetime.now(timezone.utc).isoformat(), "data": data}

        # Write then rename, so an interrupted write leaves no partial entry
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        print(f"[SourceCache] Cached {key} at {path}")
        return data
