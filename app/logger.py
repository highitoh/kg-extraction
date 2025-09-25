import os
import json
from typing import Any, Dict
from datetime import datetime


class Logger:
    """ログ出力を行うクラス"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def save_log(self, data: Dict[str, Any], filename_prefix: str = "") -> None:
        """データをJSONファイルとしてログに保存"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{filename_prefix}{timestamp}.json" if filename_prefix else f"{timestamp}.json"
        log_file = os.path.join(self.log_dir, filename)

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"ログファイルの保存中にエラーが発生しました: {e}")