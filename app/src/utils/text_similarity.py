"""Text similarity utilities using TF-IDF for matching PDF pages to Miro items."""

from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        float: Similarity score in [0.0, 1.0]

    Example:
        >>> score = compute_text_similarity("hello world", "hello there")
        >>> 0.0 <= score <= 1.0
        True
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(similarity)


def match_pages_to_items(
    pdf_page_texts: List[str],
    item_texts: List[Dict[str, Any]],
    threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Match PDF pages to Miro items (Frames/Diagrams) by text similarity.

    Args:
        pdf_page_texts: List of text strings (one per PDF page)
        item_texts: List of dicts with item metadata and text:
            [
                {
                    "item_id": str,
                    "item_type": str,  # "frame" or "diagram"
                    "item_text": str,  # Concatenated text from child items
                    ...
                },
                ...
            ]
        threshold: Minimum similarity score to consider a match (default: 0.9)

    Returns:
        List of mapping dicts:
        [
            {
                "page_number": int,  # 0-indexed
                "item_id": str,
                "item_type": str,
                "similarity_score": float
            },
            ...
        ]

    Example:
        >>> pdf_texts = ["Page 1 content", "Page 2 diagram"]
        >>> items = [
        ...     {"item_id": "frame1", "item_type": "frame", "item_text": "Page 1 content"},
        ...     {"item_id": "diagram1", "item_type": "diagram", "item_text": "Page 2 diagram"}
        ... ]
        >>> mappings = match_pages_to_items(pdf_texts, items, threshold=0.5)
        >>> len(mappings) == 2
        True
    """
    mappings = []

    for page_num, page_text in enumerate(pdf_page_texts):
        if not page_text.strip():
            continue

        best_match = None
        best_score = threshold

        for item in item_texts:
            item_text = item.get("item_text", "")
            if not item_text.strip():
                continue

            score = compute_text_similarity(page_text, item_text)

            if score > best_score:
                best_score = score
                best_match = item

        if best_match:
            mappings.append({
                "page_number": page_num,
                "item_id": best_match["item_id"],
                "item_type": best_match["item_type"],
                "similarity_score": best_score
            })

    return mappings
