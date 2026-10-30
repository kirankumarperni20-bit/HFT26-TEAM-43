import hashlib
import logging
import os
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.schemas.document import (
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
    DocumentStatusResponse,
)
from backend.schemas.chunk import Chunk
from backend.ingestion.pdf_parser import (
    extract_pdf_pages,
    InvalidPDFError,
    PasswordProtectedPDFError,
    EmptyPDFError,
    NoExtractableTextError,
)
from backend.ingestion.paragraph_extractor import extract_paragraphs
from backend.ingestion.chunker import create_chunks
from backend.ingestion.repository import DocumentRepository, InMemoryDocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Single shared repository instance across the application lifecycle
_shared_repository: DocumentRepository = InMemoryDocumentRepository()

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB default


def get_repository() -> DocumentRepository:
    """
    Dependency provider for DocumentRepository.
    Enables dependency injection and clean swappability for Member 2's PostgreSQL implementation.
    """
    return _shared_repository


def get_max_upload_bytes() -> int:
    """Returns the maximum allowed upload size in bytes, read from environment or default."""
    try:
        return int(os.getenv("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES


def _generate_paper_id(file_bytes: bytes, repo: DocumentRepository) -> str:
    """
    Generates a deterministic, collision-resistant paper_id from content hash,
    checking the repository to ensure uniqueness.
    """
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:8].upper()
    base_id = f"P{content_hash}"
    paper_id = base_id

    counter = 1
    # Check repository to prevent ID collision
    while repo.get_document(paper_id) is not None:
        paper_id = f"{base_id}_{counter}"
        counter += 1

    return paper_id


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    repo: DocumentRepository = Depends(get_repository),
):
    """
    Accepts a PDF upload, validates file type by content signature, extracts pages,
    reconstructs paragraphs, generates canonical chunks, and tracks lifecycle status.
    Enforces upload size limit in chunks and closes the uploaded file handle.
    """
    filename = file.filename or "unknown.pdf"

    # Validate filename extension as a preliminary check
    if not filename.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .pdf extension.",
        )

    max_bytes = get_max_upload_bytes()
    chunk_size = 64 * 1024  # 64 KiB read buffer
    content_accumulator = bytearray()

    # Read uploaded file in chunks while enforcing maximum size limit
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content_accumulator.extend(chunk)
            if len(content_accumulator) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {max_bytes} bytes.",
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        ) from None
    finally:
        # Guarantee that file handle is closed
        await file.close()

    content = bytes(content_accumulator)

    # Validate file is non-empty
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided PDF file is empty (0 bytes).",
        )

    # Validate magic header signature: PDF files must start with %PDF
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid PDF (invalid magic file signature).",
        )

    # Generate deterministic, collision-resistant paper ID
    paper_id = _generate_paper_id(content, repo)

    # Register initial document metadata with status=uploaded
    initial_doc = DocumentMetadata(
        paper_id=paper_id,
        filename=filename,
        status=DocumentStatus.UPLOADED,
    )
    repo.save_document(initial_doc)

    # Transition status to processing
    repo.update_document_status(paper_id, DocumentStatus.PROCESSING)

    try:
        # Step 1: Extract pages using PyMuPDF
        pages = extract_pdf_pages(content)
        repo.save_pages(paper_id, pages)

        # Step 2: Reconstruct paragraphs
        paragraphs = extract_paragraphs(pages)

        # Step 3: Generate canonical chunks
        chunks = create_chunks(paper_id, paragraphs)
        repo.save_chunks(paper_id, chunks)

        # Step 4: Mark processing completed
        completed_doc = repo.update_document_status(paper_id, DocumentStatus.COMPLETED)
        if completed_doc is None:
            raise RuntimeError("Failed to finalize document status.")

        return DocumentUploadResponse(documents=[completed_doc])

    except (InvalidPDFError, PasswordProtectedPDFError, EmptyPDFError, NoExtractableTextError) as exc:
        repo.update_document_status(paper_id, DocumentStatus.FAILED)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except Exception as exc:
        # Log exception and traceback securely on server; return generic message to client
        logger.exception("Unexpected error during document ingestion for paper_id '%s'", paper_id)
        repo.update_document_status(paper_id, DocumentStatus.FAILED)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from None


@router.get("", response_model=list[DocumentMetadata])
@router.get("/", response_model=list[DocumentMetadata], include_in_schema=False)
async def list_documents(repo: DocumentRepository = Depends(get_repository)):
    """Retrieves metadata for all stored documents."""
    return repo.list_documents()


@router.get("/{paper_id}", response_model=DocumentMetadata)
async def get_document(paper_id: str, repo: DocumentRepository = Depends(get_repository)):
    """Retrieves metadata for a specific document by paper_id."""
    doc = repo.get_document(paper_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with paper_id '{paper_id}' not found.",
        )
    return doc


@router.get("/{paper_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(paper_id: str, repo: DocumentRepository = Depends(get_repository)):
    """Retrieves processing status for a specific document."""
    doc = repo.get_document(paper_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with paper_id '{paper_id}' not found.",
        )
    return DocumentStatusResponse(paper_id=doc.paper_id, status=doc.status)


@router.get("/{paper_id}/chunks", response_model=list[Chunk])
async def get_document_chunks(paper_id: str, repo: DocumentRepository = Depends(get_repository)):
    """Retrieves all stored chunks for a specific document in original order."""
    doc = repo.get_document(paper_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with paper_id '{paper_id}' not found.",
        )
    return repo.get_chunks(paper_id)


@router.get("/{paper_id}/pages/{page_number}")
async def get_page_text(
    paper_id: str,
    page_number: int,
    repo: DocumentRepository = Depends(get_repository),
):
    """Retrieves the exact, verbatim extracted text for a specific page."""
    doc = repo.get_document(paper_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with paper_id '{paper_id}' not found.",
        )

    text = repo.get_page_text(paper_id, page_number)
    if text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page_number} for paper_id '{paper_id}' not found.",
        )

    return {
        "paper_id": paper_id,
        "page_number": page_number,
        "text": text,
    }
