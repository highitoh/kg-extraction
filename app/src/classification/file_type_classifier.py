from typing import Any, Dict
import os
import json
import tempfile
from pypdf import PdfReader, PdfWriter

from langchain_core.runnables import Runnable
from openai import OpenAI
from pydantic import BaseModel

from src.utils.logger import Logger

class FileTypeClassifierLLMSchema(BaseModel):
    is_presentation: bool
    justification: str

class FileTypeClassifier(Runnable):
    """Runnable that determines file type and adds it to input"""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        temperature: float = 0.0,
        log_dir: str = "/workspace/app/log/file_type_classifier",
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

        # Load prompt and schema
        self.prompt = self._load_prompt()

        # Initialize Logger
        self.logger = Logger(log_dir)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        # Go up 2 levels from src/classification/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts", "file_type_classifier.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_first_two_pages(self, input_pdf, output_pdf):
        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        # Add first 2 pages (or as many as exist if less than 2)
        for i in range(min(2, len(reader.pages))):
            writer.add_page(reader.pages[i])

        with open(output_pdf, "wb") as f:
            writer.write(f)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Add file_type field to input data

        Args:
            input: Input data
            config: Execution configuration

        Returns:
            Input data with file_type field added
        """
        # Get input file path
        file_path = input["files"][0]["path"]

        # Extract first 2 pages to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            temp_pdf_path = tmp_file.name

        try:
            # Extract first 2 pages
            self._extract_first_two_pages(file_path, temp_pdf_path)

            # Execute prompt with Responses API
            from openai import OpenAI
            client = OpenAI()

            # Upload PDF
            pdf = self.client.files.create(
                file=open(temp_pdf_path, "rb"),
                purpose="user_data",
            )

            # Input PDF and prompt
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

            # Parse response
            result_text = response.output_text
            result_json = json.loads(result_text)

            is_presentation = result_json["is_presentation"]
            justification = result_json["justification"]

            # Determine file_type
            file_type = "presentation" if is_presentation else "document"

            # Display classification result
            print(f"File: {file_path}, Type: {file_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "file_type": file_type,
                "is_presentation": is_presentation,
                "justification": justification
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")

        finally:
            # Delete temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

        return {**input, "file_type": file_type}


if __name__ == "__main__":
    # Sample data for testing
    sample_input = {
        "files": [{"path": "../doc/sample.pdf"}]
    }

    print("FileTypeClassifier テスト開始...")

    # Create FileTypeClassifier instance
    classifier = FileTypeClassifier(
        model="gpt-5-nano",  # For cost reduction
        temperature=0.0
    )

    try:
        # Execute test
        result = classifier.invoke(sample_input)

        print("\n=== Test Results ===")
        print(f"File path: {result['files'][0]['path']}")
        print(f"File type: {result['file_type']}")
        print("\nFileTypeClassifier test completed successfully")

    except Exception as e:
        print(f"\nTest execution error: {e}")
        import traceback
        traceback.print_exc()
