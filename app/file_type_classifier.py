from typing import Any, Dict

from langchain.schema.runnable import Runnable


class FileTypeClassifier(Runnable):
    """ファイル種別を判定してinputに追加するRunnable"""

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力データにfile_typeフィールドを追加

        Args:
            input: 入力データ
            config: 実行設定

        Returns:
            file_typeフィールドが追加された入力データ
        """
        # 現時点では常に "document" を返す（判定ロジックは未実装）
        file_type = "document"
        return {**input, "file_type": file_type}
