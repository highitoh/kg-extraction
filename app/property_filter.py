from typing import Dict, Any

from langchain.schema.runnable import Runnable

from logger import Logger

class PropertyFilter(Runnable):
    """
    プロパティフィルタリング（空実装）
    入力: PropertyChainOutput
    出力: PropertyChainOutput（フィルタリング済み）

    現在は空実装
    """

    def __init__(self, progress: bool = True, log_dir: str = "log/property_filter"):
        self.progress = progress
        # Logger設定
        self.logger = Logger(log_dir)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """PropertyChainOutputをそのまま返す（空実装）"""
        output = input.copy()

        if self.progress:
            properties_count = len(output.get("properties", []))
            print(f"PropertyFilter: {properties_count} properties (no filtering applied)")

        # Loggerでpropertyチェーンの出力を保存
        self.logger.save_log(output)

        return output
