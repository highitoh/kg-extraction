from typing import Any, Dict
import os
import json
import tempfile
from pypdf import PdfReader, PdfWriter

from langchain.schema.runnable import Runnable
from openai import OpenAI
from pydantic import BaseModel

from logger import Logger

class FileTypeClassifierLLMSchema(BaseModel):
    is_presentation: bool
    justification: str

class FileTypeClassifier(Runnable):
    """ファイル種別を判定してinputに追加するRunnable"""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        temperature: float = 0.0,
        log_dir: str = "./log/file_type_classifier",
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

        # プロンプトとスキーマを読み込み
        self.prompt = self._load_prompt()

        # Loggerを初期化
        self.logger = Logger(log_dir)

    def _load_prompt(self) -> str:
        """プロンプトファイルを読み込む"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "file_type_classifier.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_first_two_pages(self, input_pdf, output_pdf):
        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        # 最初の2ページを追加（ページ数が2未満なら存在する分だけ）
        for i in range(min(2, len(reader.pages))):
            writer.add_page(reader.pages[i])

        with open(output_pdf, "wb") as f:
            writer.write(f)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力データにfile_typeフィールドを追加

        Args:
            input: 入力データ
            config: 実行設定

        Returns:
            file_typeフィールドが追加された入力データ
        """
        # 入力ファイルパスを取得
        file_path = input["files"][0]["path"]

        # 一時ファイルで最初の2ページを抽出
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            temp_pdf_path = tmp_file.name

        try:
            # 最初の2ページを抽出
            self._extract_first_two_pages(file_path, temp_pdf_path)

            # Responses APIでプロンプト実行
            from openai import OpenAI
            client = OpenAI()

            # PDFをアップロード
            pdf = self.client.files.create(
                file=open(temp_pdf_path, "rb"),
                purpose="user_data",
            )

            # PDFとプロンプトを入力
            response = self.client.responses.parse(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": pdf.id},
                        {"type": "input_text", "text": self.prompt}
                    ]
                }],
                text_format=FileTypeClassifierLLMSchema
            )

            # レスポンスをパース
            result_text = response.output_text
            result_json = json.loads(result_text)

            is_presentation = result_json["is_presentation"]
            justification = result_json["justification"]

            # file_typeを決定
            file_type = "presentation" if is_presentation else "document"

            # 判定結果を表示
            print(f"File: {file_path}, Type: {file_type}")

            # ログを保存
            log_data = {
                "file_path": file_path,
                "file_type": file_type,
                "is_presentation": is_presentation,
                "justification": justification
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")

        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

        return {**input, "file_type": file_type}


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_input = {
        "files": [{"path": "../doc/sample.pdf"}]
    }

    print("FileTypeClassifier テスト開始...")

    # FileTypeClassifierのインスタンス作成
    classifier = FileTypeClassifier(
        model="gpt-5-nano",  # コスト削減のため
        temperature=0.0
    )

    try:
        # テスト実行
        result = classifier.invoke(sample_input)

        print("\n=== テスト結果 ===")
        print(f"ファイルパス: {result['files'][0]['path']}")
        print(f"ファイルタイプ: {result['file_type']}")
        print("\n✅ FileTypeClassifier テスト完了")

    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
