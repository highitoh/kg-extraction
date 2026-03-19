from typing import Any, Dict
import os

from langchain_core.runnables import Runnable

from src.utils.id_generator import generate_file_id
from src.utils.logger import Logger


class TextExtractor(Runnable):
    """Base class for text extractors with shared output helpers."""

    def __init__(self, log_dir: str):
        self.logger = Logger(log_dir)

    def build_output(self, file_path: str, sentences: list[Dict[str, str]]) -> Dict[str, Any]:
        """Build the shared extractor output shape."""
        return {
            "file_id": generate_file_id(file_path),
            "file_name": os.path.basename(file_path),
            "sentences": sentences,
        }
