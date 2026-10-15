import base64
import csv
import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader, PdfWriter

from src.utils.llm_factory import LLMFactory
from src.utils.logger import Logger


class FileTypeClassifierLLMSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_presentation: bool
    justification: str


class FileTypeClassifier(Runnable):
    """Runnable that determines file type and adds it to input"""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model: str | None = None,
        temperature: float = 1.0,
        log_dir: str = "/workspace/app/log/file_type_classifier",
    ):
        self.model = model
        self.temperature = temperature
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )
        self.llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "file_type_classifier",
                    "schema": FileTypeClassifierLLMSchema.model_json_schema(),
                    "strict": True,
                },
            }
        )

        # Load prompt and schema
        self.prompt = self._load_prompt()

        # Initialize Logger
        self.logger = Logger(log_dir)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        # Go up 3 levels from src/extract/classification/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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

    @staticmethod
    def _extract_usage(response) -> Optional[dict]:
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            return usage

        meta = getattr(response, "response_metadata", None)
        if isinstance(meta, dict):
            usage = meta.get("usage") or meta.get("token_usage")
            if isinstance(usage, dict):
                return usage
            if any(
                k in meta
                for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
            ):
                return meta

        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            return usage
        if usage is not None and hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            if isinstance(dumped, dict):
                return dumped
        return None

    @staticmethod
    def _to_text(content: Union[str, List[Any]]) -> str:
        if isinstance(content, str):
            return content

        for part in content:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")

        return ""

    @staticmethod
    def _read_file_base64(file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    @staticmethod
    def _log_usage(usage: dict) -> None:
        log_dir = os.path.join("log", "tokens")
        os.makedirs(log_dir, exist_ok=True)
        date_stamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"tokens_{date_stamp}.log")
        timestamp = datetime.now().isoformat(timespec="seconds")
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", ""))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", ""))
        total_tokens = usage.get("total_tokens", "")
        file_exists = os.path.exists(log_file)
        with open(log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "label", "input_tokens", "output_tokens", "total_tokens"])
            writer.writerow([timestamp, "FileTypeClassifier", input_tokens, output_tokens, total_tokens])

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

        # Check if this is a GitHub repository URL
        if "github.com/" in file_path:
            source_type = "github_repo"
            content_type = None
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "GitHub repository URL detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Check if this is a Jira URL
        if ".atlassian.net/browse/" in file_path or file_path.startswith("jira:"):
            source_type = "jira_project"
            content_type = None
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "Jira Cloud URL detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Check if this is a Confluence URL
        if ".atlassian.net/wiki/" in file_path or file_path.startswith("confluence:"):
            source_type = "confluence_space"
            content_type = None
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "Confluence Cloud URL detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Check if miro_board_url is provided
        miro_board_url = input.get("miro_board_url")
        is_board_url = "miro.com/app/board/" in file_path or file_path.startswith("miro:")
        if is_board_url or (miro_board_url and file_path.endswith(".pdf")):
            source_type = "miro_board"
            content_type = None
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "miro_board_url": miro_board_url,
                "justification": "Miro board URL detected" if is_board_url else "PDF with Miro board URL detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Check if this is a DOCX file
        if file_path.lower().endswith(".docx"):
            source_type = "docx_file"
            content_type = "document"
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "DOCX file extension detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Check if this is a PPTX file
        if file_path.lower().endswith(".pptx"):
            source_type = "pptx_file"
            content_type = "presentation"
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "PPTX file extension detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # TXT file detection
        if file_path.lower().endswith(".txt"):
            source_type = "txt_file"
            content_type = "document"
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "TXT file extension detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Markdown file detection
        if file_path.lower().endswith((".md", ".markdown")):
            source_type = "md_file"
            content_type = "document"
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "justification": "Markdown file extension detected",
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")
            return {**input, "source_type": source_type, "content_type": content_type}

        # Extract first 2 pages to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            temp_pdf_path = tmp_file.name

        try:
            # Extract first 2 pages
            self._extract_first_two_pages(file_path, temp_pdf_path)

            msg = HumanMessage(
                content=[
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "file",
                        "base64": self._read_file_base64(temp_pdf_path),
                        "mime_type": "application/pdf",
                        "filename": os.path.basename(file_path),
                    },
                ]
            )

            response = self.llm_json.invoke([msg])
            usage = self._extract_usage(response)
            if usage:
                self._log_usage(usage)

            # Parse response
            result_text = self._to_text(response.content)
            result_json = json.loads(result_text)

            is_presentation = result_json["is_presentation"]
            justification = result_json["justification"]

            # Set classification
            source_type = "pdf_file"
            content_type = "presentation" if is_presentation else "document"

            # Display classification result
            print(f"File: {file_path}, Source: {source_type}, Content: {content_type}")

            # Save log
            log_data = {
                "file_path": file_path,
                "source_type": source_type,
                "content_type": content_type,
                "is_presentation": is_presentation,
                "justification": justification,
            }
            self.logger.save_log(log_data, "file_type_classifier_output_")

        finally:
            # Delete temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

        return {**input, "source_type": source_type, "content_type": content_type}


if __name__ == "__main__":
    # Sample data for testing
    sample_input = {"files": [{"path": "../doc/sample.pdf"}]}

    print("FileTypeClassifier test starting...")

    # Create FileTypeClassifier instance
    classifier = FileTypeClassifier(temperature=1.0)

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
