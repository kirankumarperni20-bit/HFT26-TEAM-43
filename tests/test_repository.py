import pytest

from backend.ingestion.repository import InMemoryDocumentRepository
from backend.schemas.document import DocumentMetadata, DocumentStatus
from backend.schemas.chunk import Chunk
from backend.ingestion.pdf_parser import ExtractedPage


def test_saving_document_metadata():
    """Saving document metadata."""
    repo = InMemoryDocumentRepository()
    doc = DocumentMetadata(paper_id="P001", filename="paper.pdf", status=DocumentStatus.UPLOADED)
    repo.save_document(doc)

    retrieved = repo.get_document("P001")
    assert retrieved is not None
    assert retrieved.paper_id == "P001"
    assert retrieved.filename == "paper.pdf"
    assert retrieved.status == DocumentStatus.UPLOADED


def test_duplicate_paper_id_rejection():
    """1. Duplicate paper ID rejection in save_document()."""
    repo = InMemoryDocumentRepository()
    doc1 = DocumentMetadata(paper_id="P001", filename="paper1.pdf", status=DocumentStatus.UPLOADED)
    doc2 = DocumentMetadata(paper_id="P001", filename="paper2.pdf", status=DocumentStatus.PROCESSING)

    repo.save_document(doc1)
    with pytest.raises(ValueError) as exc_info:
        repo.save_document(doc2)

    assert "already exists" in str(exc_info.value)
    # Ensure original document is preserved
    original = repo.get_document("P001")
    assert original.filename == "paper1.pdf"
    assert original.status == DocumentStatus.UPLOADED


def test_retrieving_document_by_paper_id():
    """Retrieving document by paper_id."""
    repo = InMemoryDocumentRepository()
    doc = DocumentMetadata(paper_id="P002", filename="test.pdf", status=DocumentStatus.PROCESSING)
    repo.save_document(doc)

    retrieved = repo.get_document("P002")
    assert retrieved == doc


def test_unknown_paper_handling():
    """Unknown paper handling returns None or empty results."""
    repo = InMemoryDocumentRepository()
    assert repo.get_document("NONEXISTENT") is None
    assert repo.get_chunks("NONEXISTENT") == []
    assert repo.get_page_text("NONEXISTENT", 1) is None
    assert repo.update_document_status("NONEXISTENT", DocumentStatus.COMPLETED) is None


def test_updating_processing_status():
    """Updating processing status preserves other metadata."""
    repo = InMemoryDocumentRepository()
    doc = DocumentMetadata(paper_id="P001", filename="original.pdf", status=DocumentStatus.UPLOADED)
    repo.save_document(doc)

    updated = repo.update_document_status("P001", DocumentStatus.COMPLETED)
    assert updated is not None
    assert updated.status == DocumentStatus.COMPLETED
    assert updated.filename == "original.pdf"
    assert updated.paper_id == "P001"

    retrieved = repo.get_document("P001")
    assert retrieved.status == DocumentStatus.COMPLETED


def test_external_mutation_of_retrieved_document():
    """2. External mutation of a retrieved document does not mutate stored data."""
    repo = InMemoryDocumentRepository()
    doc = DocumentMetadata(paper_id="P001", filename="original.pdf", status=DocumentStatus.UPLOADED)
    repo.save_document(doc)

    retrieved = repo.get_document("P001")
    # Mutate retrieved object attributes directly if possible or reassign
    # Note: Pydantic models can have attributes modified
    try:
        retrieved.filename = "mutated.pdf"
    except Exception:
        pass

    # Verify repository copy remains untouched
    fresh = repo.get_document("P001")
    assert fresh.filename == "original.pdf"


def test_external_mutation_of_stored_input_objects():
    """External mutation of input objects after save does not affect stored data."""
    repo = InMemoryDocumentRepository()
    doc = DocumentMetadata(paper_id="P001", filename="original.pdf", status=DocumentStatus.UPLOADED)
    repo.save_document(doc)

    # Mutate the input object passed to save_document
    doc.filename = "modified_externally.pdf"

    stored = repo.get_document("P001")
    assert stored.filename == "original.pdf"


def test_external_mutation_of_retrieved_chunks():
    """External mutation of retrieved chunks does not affect stored data."""
    repo = InMemoryDocumentRepository()
    chunk = Chunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Original text",
    )
    repo.save_chunks("P001", [chunk])

    retrieved_chunks = repo.get_chunks("P001")
    retrieved_chunks[0].text = "Mutated text"
    retrieved_chunks.append(
        Chunk(
            chunk_id="P001_C999",
            paper_id="P001",
            page_number=1,
            paragraph_number=99,
            text="Injected chunk",
        )
    )

    fresh_chunks = repo.get_chunks("P001")
    assert len(fresh_chunks) == 1
    assert fresh_chunks[0].text == "Original text"
    assert fresh_chunks[0].chunk_id == "P001_C001"


def test_saving_extracted_pages():
    """Saving extracted pages."""
    repo = InMemoryDocumentRepository()
    pages = [
        ExtractedPage(page_number=1, text="Page 1 text"),
        ExtractedPage(page_number=2, text="Page 2 text"),
    ]
    repo.save_pages("P001", pages)

    assert repo.get_page_text("P001", 1) == "Page 1 text"
    assert repo.get_page_text("P001", 2) == "Page 2 text"


def test_invalid_page_numbers():
    """3. Invalid page numbers (zero or negative) are rejected."""
    repo = InMemoryDocumentRepository()

    with pytest.raises(ValueError) as exc1:
        repo.save_pages("P001", [ExtractedPage(page_number=0, text="Zero page")])
    assert "greater than zero" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        repo.save_pages("P001", [ExtractedPage(page_number=-5, text="Negative page")])
    assert "greater than zero" in str(exc2.value)


def test_duplicate_page_number_rejection():
    """Duplicate page numbers within batch and existing pages are rejected."""
    repo = InMemoryDocumentRepository()

    # Duplicate in same input batch
    batch_with_dups = [
        ExtractedPage(page_number=1, text="Page 1 first copy"),
        ExtractedPage(page_number=1, text="Page 1 second copy"),
    ]
    with pytest.raises(ValueError) as exc1:
        repo.save_pages("P001", batch_with_dups)
    assert "Duplicate page number" in str(exc1.value)

    # Duplicate across subsequent save_pages calls
    repo.save_pages("P001", [ExtractedPage(page_number=1, text="First save")])
    with pytest.raises(ValueError) as exc2:
        repo.save_pages("P001", [ExtractedPage(page_number=1, text="Second save")])
    assert "already exists" in str(exc2.value)


def test_retrieving_page_text():
    """Retrieving page text for existing and missing pages."""
    repo = InMemoryDocumentRepository()
    repo.save_pages("P001", [ExtractedPage(page_number=1, text="Specific text")])

    assert repo.get_page_text("P001", 1) == "Specific text"
    assert repo.get_page_text("P001", 999) is None
    assert repo.get_page_text("OTHER_PAPER", 1) is None


def test_saving_chunks_and_duplicate_chunk_rejection():
    """4. Duplicate chunk ID rejection within batch."""
    repo = InMemoryDocumentRepository()
    duplicate_chunks = [
        Chunk(chunk_id="P001_C001", paper_id="P001", page_number=1, paragraph_number=1, text="Chunk 1"),
        Chunk(chunk_id="P001_C001", paper_id="P001", page_number=1, paragraph_number=2, text="Duplicate Chunk"),
    ]

    with pytest.raises(ValueError) as exc_info:
        repo.save_chunks("P001", duplicate_chunks)

    assert "Duplicate chunk ID" in str(exc_info.value)


def test_retrieving_chunks_in_original_order():
    """Retrieving chunks preserves original order."""
    repo = InMemoryDocumentRepository()
    chunks = [
        Chunk(chunk_id=f"P001_C{i:03d}", paper_id="P001", page_number=1, paragraph_number=i, text=f"Text {i}")
        for i in range(1, 10)
    ]
    repo.save_chunks("P001", chunks)

    retrieved = repo.get_chunks("P001")
    assert [c.chunk_id for c in retrieved] == [f"P001_C{i:03d}" for i in range(1, 10)]


def test_saving_chunks_for_different_papers_independently():
    """Saving chunks for different papers independently."""
    repo = InMemoryDocumentRepository()
    chunks_p1 = [
        Chunk(chunk_id="P001_C001", paper_id="P001", page_number=1, paragraph_number=1, text="P1 Text")
    ]
    chunks_p2 = [
        Chunk(chunk_id="P002_C001", paper_id="P002", page_number=1, paragraph_number=1, text="P2 Text")
    ]

    repo.save_chunks("P001", chunks_p1)
    repo.save_chunks("P002", chunks_p2)

    assert repo.get_chunks("P001")[0].text == "P1 Text"
    assert repo.get_chunks("P002")[0].text == "P2 Text"


def test_rejecting_chunks_belonging_to_different_paper_id():
    """Rejecting chunks belonging to a different paper ID."""
    repo = InMemoryDocumentRepository()
    mismatched_chunks = [
        Chunk(chunk_id="P002_C001", paper_id="P002", page_number=1, paragraph_number=1, text="Wrong paper text")
    ]

    with pytest.raises(ValueError) as exc_info:
        repo.save_chunks("P001", mismatched_chunks)

    assert "does not match target paper_id" in str(exc_info.value)


def test_listing_documents():
    """Listing documents returns deep copies."""
    repo = InMemoryDocumentRepository()
    assert repo.list_documents() == []

    doc1 = DocumentMetadata(paper_id="P001", filename="doc1.pdf", status=DocumentStatus.COMPLETED)
    doc2 = DocumentMetadata(paper_id="P002", filename="doc2.pdf", status=DocumentStatus.PROCESSING)

    repo.save_document(doc1)
    repo.save_document(doc2)

    docs = repo.list_documents()
    assert len(docs) == 2
    # Mutate listed list and objects
    docs[0].filename = "mutated.pdf"

    fresh = repo.get_document(docs[0].paper_id)
    assert fresh.filename != "mutated.pdf"


def test_separate_repository_instances_do_not_share_state():
    """Separate repository instances do not share in-memory state unintentionally."""
    repo1 = InMemoryDocumentRepository()
    repo2 = InMemoryDocumentRepository()

    doc = DocumentMetadata(paper_id="P001", filename="doc.pdf", status=DocumentStatus.UPLOADED)
    repo1.save_document(doc)

    assert repo1.get_document("P001") is not None
    assert repo2.get_document("P001") is None
    assert len(repo2.list_documents()) == 0
