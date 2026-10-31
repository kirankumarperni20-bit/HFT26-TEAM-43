import copy
from abc import ABC, abstractmethod
from typing import Optional

from backend.schemas.document import DocumentMetadata, DocumentStatus
from backend.schemas.chunk import Chunk
from backend.ingestion.pdf_parser import ExtractedPage


class DocumentRepository(ABC):
    """
    Abstract interface for document, page, and chunk persistence.
    Provides a modular integration boundary allowing Member 2 to replace
    the in-memory implementation with PostgreSQL/pgvector without changing
    the Ingestion module's contracts.
    """

    @abstractmethod
    def save_document(self, document: DocumentMetadata) -> None:
        """Saves new document metadata. Raises ValueError if paper_id already exists."""
        pass

    @abstractmethod
    def get_document(self, paper_id: str) -> Optional[DocumentMetadata]:
        """Retrieves document metadata by paper_id, returning None if not found."""
        pass

    @abstractmethod
    def update_document_status(self, paper_id: str, status: DocumentStatus) -> Optional[DocumentMetadata]:
        """Updates the processing status of an existing document while preserving other metadata."""
        pass

    @abstractmethod
    def save_pages(self, paper_id: str, pages: list[ExtractedPage]) -> None:
        """Stores extracted page-level text for a document."""
        pass

    @abstractmethod
    def get_page_text(self, paper_id: str, page_number: int) -> Optional[str]:
        """Retrieves raw extracted text for a specific page of a document, or None if not found."""
        pass

    @abstractmethod
    def save_chunks(self, paper_id: str, chunks: list[Chunk]) -> None:
        """Stores generated chunks for a document, preserving their original order."""
        pass

    @abstractmethod
    def get_chunks(self, paper_id: str) -> list[Chunk]:
        """Retrieves stored chunks for a paper_id in their original order."""
        pass

    @abstractmethod
    def list_documents(self) -> list[DocumentMetadata]:
        """Lists all stored document metadata records."""
        pass


class InMemoryDocumentRepository(DocumentRepository):
    """
    Temporary in-memory implementation of DocumentRepository with duplicate protection,
    strict input validation, and deep-copy data isolation.

    NOTE: All data is stored strictly in memory and will disappear when the application restarts.
    This interface is designed for seamless replacement by Member 2 with PostgreSQL/pgvector.
    """

    def __init__(self) -> None:
        # Isolated per-instance state (no global mutable state)
        self._documents: dict[str, DocumentMetadata] = {}
        self._pages: dict[tuple[str, int], ExtractedPage] = {}
        self._chunks: dict[str, list[Chunk]] = {}

    def save_document(self, document: DocumentMetadata) -> None:
        if not isinstance(document, DocumentMetadata):
            raise TypeError(f"Expected DocumentMetadata, got {type(document).__name__}")
        if document.paper_id in self._documents:
            raise ValueError(f"Document with paper_id '{document.paper_id}' already exists.")
        # Store deep copy to prevent mutation of stored data via the input object
        self._documents[document.paper_id] = copy.deepcopy(document)

    def get_document(self, paper_id: str) -> Optional[DocumentMetadata]:
        doc = self._documents.get(paper_id)
        # Return deep copy so callers cannot mutate repository internal state
        return copy.deepcopy(doc) if doc is not None else None

    def update_document_status(self, paper_id: str, status: DocumentStatus) -> Optional[DocumentMetadata]:
        doc = self._documents.get(paper_id)
        if doc is None:
            return None
        # Preserve existing metadata (filename, paper_id) and only update status
        updated_doc = DocumentMetadata(
            paper_id=doc.paper_id,
            filename=doc.filename,
            status=status,
        )
        self._documents[paper_id] = updated_doc
        return copy.deepcopy(updated_doc)

    def save_pages(self, paper_id: str, pages: list[ExtractedPage]) -> None:
        if not paper_id or not paper_id.strip():
            raise ValueError("paper_id cannot be empty or whitespace only")

        seen_page_numbers: set[int] = set()

        # Batch validation prior to committing any records
        for idx, page in enumerate(pages):
            if not isinstance(page, ExtractedPage):
                raise TypeError(f"Expected ExtractedPage at index {idx}, got {type(page).__name__}")
            if page.page_number <= 0:
                raise ValueError(f"Page number must be greater than zero, got {page.page_number}")
            if page.page_number in seen_page_numbers:
                raise ValueError(
                    f"Duplicate page number {page.page_number} detected in input batch for paper '{paper_id}'"
                )
            seen_page_numbers.add(page.page_number)
            if (paper_id, page.page_number) in self._pages:
                raise ValueError(f"Page {page.page_number} for paper '{paper_id}' already exists")

        # Store deep copies to isolate stored text from external changes
        for page in pages:
            self._pages[(paper_id, page.page_number)] = copy.deepcopy(page)

    def get_page_text(self, paper_id: str, page_number: int) -> Optional[str]:
        page = self._pages.get((paper_id, page_number))
        return page.text if page is not None else None

    def save_chunks(self, paper_id: str, chunks: list[Chunk]) -> None:
        if not paper_id or not paper_id.strip():
            raise ValueError("paper_id cannot be empty or whitespace only")

        seen_chunk_ids: set[str] = set()

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"Expected Chunk at index {idx}, got {type(chunk).__name__}")
            if chunk.paper_id != paper_id:
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' has paper_id '{chunk.paper_id}', which does not match target paper_id '{paper_id}'"
                )
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID '{chunk.chunk_id}' detected in input batch for paper '{paper_id}'"
                )
            seen_chunk_ids.add(chunk.chunk_id)

        # Preserve original order and store deep copies
        self._chunks[paper_id] = [copy.deepcopy(c) for c in chunks]

    def get_chunks(self, paper_id: str) -> list[Chunk]:
        stored = self._chunks.get(paper_id, [])
        # Return deep copies so callers cannot mutate stored Chunk instances
        return [copy.deepcopy(c) for c in stored]

    def list_documents(self) -> list[DocumentMetadata]:
        return [copy.deepcopy(doc) for doc in self._documents.values()]
