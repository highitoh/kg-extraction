from typing import Any, Dict
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
import pdfplumber
from logger import Logger

class PresentationTextExtractor(Runnable):
    """プレゼンテーションファイルからスライドごとにテキストを抽出するタスク

    各スライドのテキストの先頭に「スライド{番号}:」を付与して出力します。
    既存のtext-extractorと同じスキーマ構造を維持します。
    """

    def __init__(self):
        self.logger = Logger("./log/presentation_text_extractor")

    def _extract_text_by_slide(self, pdf_path: str) -> list[tuple[int, str]]:
        """
        pdfplumberを使用してPDFからスライドごとにテキストを抽出

        Returns:
            list[tuple[int, str]]: (スライド番号, テキスト)のリスト
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDFファイルではありません")

        slides = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        slides.append((page_num, page_text.strip()))
        except Exception as e:
            print(f"pdfplumberでのテキスト抽出中にエラーが発生しました: {e}")

        return slides

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: PresentationTextExtractorInput (text-extractorと同じ)
        # output: PresentationTextExtractorOutput (text-extractorと同じ構造)

        file_path = input["files"][0]["path"]

        # PDFからスライドごとにテキストを抽出
        slides = self._extract_text_by_slide(file_path)

        # スライドごとの文章を「スライド{番号}: 」プレフィックス付きで生成
        sentences = []
        for slide_num, slide_text in slides:
            for line_text in slide_text.split('\n'):
                if line_text.strip():  # 空行は除外
                    sentences.append({
                        "text": f"[[スライド{slide_num}]] {line_text.strip()}"
                    })

        output = {
            "id": str(uuid.uuid4()),
            "file_name": os.path.basename(file_path),
            "sentences": sentences
        }

        # 出力JSONをログに保存
        self.logger.save_log(output, "presentation_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = PresentationTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/presentation_sample.pdf"}]})
    print(f"抽出されたスライド数: {len([s for s in result['sentences'] if 'スライド1:' in s['text']])}")
    print(f"総文章数: {len(result['sentences'])}")
    print("\n最初の5文章:")
    for i, sentence in enumerate(result['sentences'][:5], 1):
        print(f"{i}. {sentence['text']}")
