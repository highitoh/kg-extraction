from typing import Dict, Any

from langchain.schema.runnable import Runnable

from logger import Logger

class ClassFilter(Runnable):
    """
    Filter and process extracted classes
    """

    def __init__(self, progress: bool = True, log_dir: str = "log/class_filter"):
        self.progress = progress
        self.logger = Logger(log_dir)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        output = input.copy()

        if self.progress:
            classes_count = len(output.get("classes", []))
            print(f"ClassFilter: {classes_count} classes (no filtering applied)")

        self.logger.save_log(output)

        return output
