from typing import Any, Dict
import hashlib
import os
import uuid

from langchain_core.runnables import Runnable

from src.utils.logger import Logger


class TextExtractor(Runnable):
    """Base class for text extractors with shared output helpers."""

    def __init__(self, log_dir: str):
        self.logger = Logger(log_dir)

    def compute_file_id(self, file_path: str) -> str:
        """Return a deterministic file ID based on the file contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_output(self, file_path: str, sentences: list[Dict[str, str]]) -> Dict[str, Any]:
        """Build the shared extractor output shape."""
        return {
            "id": str(uuid.uuid4()),
            "file_id": self.compute_file_id(file_path),
            "file_name": os.path.basename(file_path),
            "sentences": sentences,
        }
