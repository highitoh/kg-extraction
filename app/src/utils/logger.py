import csv
import os
import json
from typing import Any, Dict
from datetime import datetime


class Logger:
    """Class for logging output"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def save_log(self, data: Dict[str, Any], filename_prefix: str = "") -> None:
        """Save data to log as JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{filename_prefix}{timestamp}.json" if filename_prefix else f"{timestamp}.json"
        log_file = os.path.join(self.log_dir, filename)

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error occurred while saving log file: {e}")

    def error_log(self, label: str, attempt: int, error_type: str, error_message: str) -> None:
        """Append an error entry to a daily CSV log file"""
        date_stamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"errors_{date_stamp}.log")
        timestamp = datetime.now().isoformat(timespec="seconds")
        try:
            file_exists = os.path.exists(log_file)
            with open(log_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "label", "attempt", "error_type", "error_message"])
                writer.writerow([timestamp, label, attempt, error_type, error_message])
        except Exception as e:
            print(f"Error occurred while saving error log: {e}")