from backend.ingestion.paragraph_extractor import ExtractedParagraph
from backend.schemas.chunk import Chunk


def create_chunks(
    paper_id: str,
    paragraphs: list[ExtractedParagraph]
) -> list[Chunk]:
    """
    Converts extracted paragraphs into canonical Pydantic Chunk objects adhering
    to the shared team integration contract.

    Args:
        paper_id: Unique identifier for the parent document (e.g., 'P001').
        paragraphs: List of ExtractedParagraph objects in document order.

    Returns:
        list[Chunk]: Canonical Chunk objects with sequential IDs and preserved metadata.

    Raises:
        ValueError: If paper_id is empty or whitespace-only.
        TypeError: If an item in paragraphs is not an ExtractedParagraph.
        pydantic.ValidationError: If any chunk violates the canonical Chunk schema.
    """
    if not paper_id or not paper_id.strip():
        raise ValueError("paper_id cannot be empty or whitespace only")

    chunks: list[Chunk] = []

    for idx, para in enumerate(paragraphs, start=1):
        if not isinstance(para, ExtractedParagraph):
            raise TypeError(
                f"Expected ExtractedParagraph instance at index {idx - 1}, got {type(para).__name__}"
            )

        # Sequential, deterministic chunk ID within the paper: {paper_id}_C001, {paper_id}_C002, ...
        chunk_id = f"{paper_id}_C{idx:03d}"

        # Construct Chunk model, validated by Pydantic schema
        chunk = Chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            page_number=para.page_number,
            paragraph_number=para.paragraph_number,
            section=para.section,
            text=para.text,
        )
        chunks.append(chunk)

    return chunks


class Chunker:
    """Chunker utility wrapper around create_chunks."""

    @staticmethod
    def create_chunks(paper_id: str, paragraphs: list[ExtractedParagraph]) -> list[Chunk]:
        return create_chunks(paper_id, paragraphs)
