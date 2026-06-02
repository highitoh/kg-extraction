from typing import Any, Dict
import os

from langchain_core.runnables.config import RunnableConfig
import pdfplumber
from .text_extractor import TextExtractor


class PresentationTextExtractor(TextExtractor):
    """Task to extract text from presentation files slide by slide

    Adds "Slide {number}:" prefix to the beginning of each slide's text.
    Maintains the same schema structure as the existing text-extractor.
    """

    def __init__(self):
        super().__init__("/workspace/app/log/presentation_text_extractor")

    def _extract_text_by_slide(self, pdf_path: str) -> list[tuple[int, str]]:
        """
        Extract text from PDF slide by slide using pdfplumber

        Returns:
            list[tuple[int, str]]: List of (slide number, text) tuples
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Not a PDF file")

        slides = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        slides.append((page_num, page_text.strip()))
        except Exception as e:
            print(f"Error occurred during text extraction with pdfplumber: {e}")

        return slides

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: PresentationTextExtractorInput (same as text-extractor)
        # output: PresentationTextExtractorOutput (same structure as text-extractor)

        file_path = input["files"][0]["path"]

        # Extract text from PDF slide by slide
        slides = self._extract_text_by_slide(file_path)

        # Generate sentences with "Slide {number}:" prefix for each slide
        sentences = []
        for slide_num, slide_text in slides:
            for line_text in slide_text.split("\n"):
                if line_text.strip():  # Exclude empty lines
                    sentences.append({"text": f"[[Slide{slide_num}]] {line_text.strip()}"})

        output = self.build_output(file_path, sentences)

        # Save output JSON to log
        self.logger.save_log(output, "presentation_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = PresentationTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/presentation_sample.pdf"}]})
    print(f"Extracted Slides: {len([s for s in result['sentences'] if 'Slide1' in s['text']])}")
    print(f"Total Sentences: {len(result['sentences'])}")
    print("\nFirst 5 Sentences:")
    for i, sentence in enumerate(result["sentences"][:5], 1):
        print(f"{i}. {sentence['text']}")
