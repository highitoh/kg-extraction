"""Convert Confluence storage format (XHTML) to Markdown."""

import re


class StorageFormatConverter:
    """Convert Confluence storage format to Markdown.

    Confluence uses storage format (XHTML-based) for page content.
    This converter handles basic elements and provides fallback to plain text.
    """

    def convert(self, storage_html: str) -> str:
        """Convert Confluence storage format to Markdown.

        Args:
            storage_html: Storage format content (XHTML string)

        Returns:
            Markdown text

        Note:
            This is a basic implementation. For production use, consider:
            - Using markdownify library for better conversion
            - Handling Confluence-specific macros (code blocks, panels, etc.)
            - Preserving table formatting
        """
        if not storage_html:
            return ""

        try:
            # Basic HTML to Markdown conversion
            text = storage_html

            # Convert headings
            text = re.sub(r"<h1>(.*?)</h1>", r"# \1\n\n", text, flags=re.DOTALL)
            text = re.sub(r"<h2>(.*?)</h2>", r"## \1\n\n", text, flags=re.DOTALL)
            text = re.sub(r"<h3>(.*?)</h3>", r"### \1\n\n", text, flags=re.DOTALL)
            text = re.sub(r"<h4>(.*?)</h4>", r"#### \1\n\n", text, flags=re.DOTALL)
            text = re.sub(r"<h5>(.*?)</h5>", r"##### \1\n\n", text, flags=re.DOTALL)
            text = re.sub(r"<h6>(.*?)</h6>", r"###### \1\n\n", text, flags=re.DOTALL)

            # Convert lists (add line breaks around list items)
            text = re.sub(r"<li>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)
            text = re.sub(r"<ul>", r"\n", text)
            text = re.sub(r"</ul>", r"\n", text)
            text = re.sub(r"<ol>", r"\n", text)
            text = re.sub(r"</ol>", r"\n", text)

            # Convert tables (add line breaks)
            text = re.sub(r"<tr>", r"", text)
            text = re.sub(r"</tr>", r"\n", text)
            text = re.sub(r"<td>(.*?)</td>", r"\1 | ", text, flags=re.DOTALL)
            text = re.sub(r"<th>(.*?)</th>", r"**\1** | ", text, flags=re.DOTALL)
            text = re.sub(r"<table[^>]*>", r"\n", text)
            text = re.sub(r"</table>", r"\n", text)

            # Convert bold and italic
            text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text, flags=re.DOTALL)
            text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.DOTALL)
            text = re.sub(r"<em>(.*?)</em>", r"*\1*", text, flags=re.DOTALL)
            text = re.sub(r"<i>(.*?)</i>", r"*\1*", text, flags=re.DOTALL)

            # Convert paragraphs
            text = re.sub(r"<p>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)

            # Convert line breaks
            text = re.sub(r"<br\s*/?>", "\n", text)

            # Convert divs (block elements should have line breaks)
            text = re.sub(r"<div[^>]*>", r"\n", text)
            text = re.sub(r"</div>", r"\n", text)

            # Convert code blocks
            text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)

            # Remove remaining HTML tags
            text = re.sub(r"<[^>]+>", "", text)

            # Post-processing: improve sentence segmentation
            text = re.sub(r"([。！？]|[.!?]\s+(?=[A-Z]))\s*(?=[^\n])", r"\1\n", text)

            # Clean up extra whitespace
            text = re.sub(r"\n\n\n+", "\n\n", text)
            text = re.sub(r"  +", " ", text)  # Multiple spaces to single space
            text = text.strip()

            return text

        except Exception as e:
            print(f"[StorageFormatConverter] Error converting storage format: {e}")
            # Fallback: strip all HTML tags
            return self._fallback_extract_text(storage_html)

    def _fallback_extract_text(self, storage_html: str) -> str:
        """Fallback: extract plain text by stripping HTML tags.

        Args:
            storage_html: Storage format content

        Returns:
            Plain text
        """
        if not storage_html:
            return ""

        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", storage_html)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()
