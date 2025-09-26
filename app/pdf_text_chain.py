from typing import Any, Dict
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig

from text_extractor import TextExtractor
from text_filter import TextFilter
from logger import Logger

class PDFTextChainPostProcess(Runnable):
    """PDFテキスト抽出チェインの後処理"""

    def __init__(self):
        self.logger = Logger("./app/log/pdf_text_chain")

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: TextExtractorOutput または TextFilterOutput
        # output: PDFTextChainOutput

        # 入力の形式を判定し、適切にデータを抽出
        if "items" in input:
            # TextExtractorOutput または TextFilterOutput
            sentences = [
                {"line": item["line"], "text": item["text"]}
                for item in input["items"]
            ]
        else:
            # 予期しない形式の場合は空配列
            sentences = []

        output = {
            "id": input.get("id", str(uuid.uuid4())),
            "file_name": input.get("file_name", "unknown"),
            "sentences": sentences
        }

        # ログ出力
        self.logger.save_log(output)

        return output

class PDFTextChain(Runnable):
    """PDFテキスト抽出チェイン"""

    def __init__(self):
        self.extractor = TextExtractor()
        self.filter = TextFilter()
        self.post_process = PDFTextChainPostProcess()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: テキスト抽出
        extracted = self.extractor.invoke(input, config)
        # Step 2: フィルタリング
        filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        # Step 3: 後処理
        output = self.post_process.invoke(filtered, config)
        return output


if __name__ == "__main__":
    chain = PDFTextChain()
    result = chain.invoke({
        "files": [{"path": "../doc/sample.pdf"}]
    })
    print(result)
