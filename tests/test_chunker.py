import pytest
from backend.ingestion.paragraph_extractor import ExtractedParagraph, extract_paragraphs
from backend.ingestion.pdf_parser import ExtractedPage
from backend.ingestion.chunker import create_chunks
from backend.schemas.chunk import Chunk


def test_one_paragraph_produces_one_chunk():
    """1. One paragraph produces one chunk."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Introduction", text="Single paragraph text.")
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].chunk_id == "P001_C001"
    assert chunks[0].text == "Single paragraph text."


def test_multiple_paragraphs_produce_multiple_chunks():
    """2. Multiple paragraphs produce multiple chunks."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Intro", text="Para 1"),
        ExtractedParagraph(page_number=1, paragraph_number=2, section="Intro", text="Para 2"),
        ExtractedParagraph(page_number=1, paragraph_number=3, section="Intro", text="Para 3"),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert len(chunks) == 3


def test_correct_paper_id_assignment():
    """3. Correct paper_id assignment."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section=None, text="Sample body paragraph.")
    ]
    chunks = create_chunks(paper_id="P999", paragraphs=paragraphs)

    assert chunks[0].paper_id == "P999"


def test_correct_sequential_chunk_ids():
    """4. Correct sequential chunk IDs: {paper_id}_C001, {paper_id}_C002, ..."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section=None, text="Para 1"),
        ExtractedParagraph(page_number=1, paragraph_number=2, section=None, text="Para 2"),
        ExtractedParagraph(page_number=1, paragraph_number=3, section=None, text="Para 3"),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert [c.chunk_id for c in chunks] == ["P001_C001", "P001_C002", "P001_C003"]


def test_chunk_numbering_continues_across_pages():
    """5. Chunk numbering continues across pages without resetting."""
    paragraphs = [
        # Page 1
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Intro", text="P1 Para 1"),
        ExtractedParagraph(page_number=1, paragraph_number=2, section="Intro", text="P1 Para 2"),
        # Page 2
        ExtractedParagraph(page_number=2, paragraph_number=1, section="Intro", text="P2 Para 1"),
        ExtractedParagraph(page_number=2, paragraph_number=2, section="Methods", text="P2 Para 2"),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert len(chunks) == 4
    assert chunks[0].chunk_id == "P001_C001" and chunks[0].page_number == 1
    assert chunks[1].chunk_id == "P001_C002" and chunks[1].page_number == 1
    # Page 2 chunks must continue numbering: C003, C004
    assert chunks[2].chunk_id == "P001_C003" and chunks[2].page_number == 2
    assert chunks[3].chunk_id == "P001_C004" and chunks[3].page_number == 2


def test_paragraph_numbering_remains_unchanged():
    """6. Paragraph numbering remains unchanged (retained from input ExtractedParagraph)."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=5, section="Results", text="Para 5 on page 1"),
        ExtractedParagraph(page_number=2, paragraph_number=1, section="Results", text="Para 1 on page 2"),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].paragraph_number == 5
    assert chunks[1].paragraph_number == 1


def test_page_numbers_remain_unchanged():
    """7. Page numbers remain unchanged."""
    paragraphs = [
        ExtractedParagraph(page_number=3, paragraph_number=1, section="Methods", text="Content on page 3"),
        ExtractedParagraph(page_number=8, paragraph_number=2, section="Discussion", text="Content on page 8"),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].page_number == 3
    assert chunks[1].page_number == 8


def test_section_information_is_preserved():
    """8. Section information is preserved."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Methodology", text="Model specs.")
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].section == "Methodology"


def test_null_section_is_accepted():
    """9. Null section is accepted."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section=None, text="Unclassified intro.")
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].section is None


def test_original_paragraph_text_is_preserved_exactly():
    """10. Original paragraph text is preserved exactly."""
    exact_text = "The model achieves 94.85% accuracy on CIFAR-10; Loss: L = -sum(y * log(p))."
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Results", text=exact_text)
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].text == exact_text


def test_internal_newlines_and_whitespace_are_preserved():
    """11. Internal newlines and whitespace are preserved."""
    multiline_text = "Line one of wrapped paragraph.\nLine two of wrapped paragraph.\nLine three."
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Intro", text=multiline_text)
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].text == multiline_text


def test_input_order_is_preserved():
    """12. Input order is preserved."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section=None, text="First"),
        ExtractedParagraph(page_number=1, paragraph_number=2, section=None, text="Second"),
        ExtractedParagraph(page_number=1, paragraph_number=3, section=None, text="Third"),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert [c.text for c in chunks] == ["First", "Second", "Third"]


def test_identical_paragraph_text_does_not_produce_duplicate_ids():
    """13. Identical paragraph text does not produce duplicate IDs."""
    same_text = "Repeatable passage."
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section=None, text=same_text),
        ExtractedParagraph(page_number=1, paragraph_number=2, section=None, text=same_text),
    ]
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert chunks[0].chunk_id == "P001_C001"
    assert chunks[1].chunk_id == "P001_C002"


def test_repeated_calls_generate_same_ids():
    """14. Repeated calls with the same input generate the same IDs (deterministic)."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section="Intro", text="Para A"),
        ExtractedParagraph(page_number=1, paragraph_number=2, section="Intro", text="Para B"),
    ]
    run_1 = create_chunks(paper_id="P001", paragraphs=paragraphs)
    run_2 = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert [c.chunk_id for c in run_1] == [c.chunk_id for c in run_2]
    assert [c.model_dump() for c in run_1] == [c.model_dump() for c in run_2]


def test_empty_paragraph_list_returns_empty_list():
    """15. Empty paragraph list returns an empty list."""
    chunks = create_chunks(paper_id="P001", paragraphs=[])
    assert chunks == []


def test_empty_paper_id_is_rejected():
    """16. Empty paper ID is rejected."""
    paragraphs = [
        ExtractedParagraph(page_number=1, paragraph_number=1, section=None, text="Valid text")
    ]
    with pytest.raises(ValueError):
        create_chunks(paper_id="", paragraphs=paragraphs)

    with pytest.raises(ValueError):
        create_chunks(paper_id="   ", paragraphs=paragraphs)


def test_end_to_end_unit_integration():
    """
    End-to-end unit integration test:
    ExtractedPage -> extract_paragraphs() -> create_chunks()
    """
    # 2 pages with multiple paragraphs and section headings
    pages = [
        ExtractedPage(
            page_number=1,
            text=(
                "1. Introduction\n\n"
                "This research proposes a new approach to verifiable summarization.\n"
                "It employs atomic claim decomposition.\n\n"
                "The system verifies each claim against original evidence passages."
            ),
        ),
        ExtractedPage(
            page_number=2,
            text=(
                "Continuing discussion from page one.\n\n"
                "2. Methodology\n\n"
                "The architecture integrates PyMuPDF and Pydantic schemas."
            ),
        ),
    ]

    # Pipeline execution: Page -> Paragraphs -> Chunks
    paragraphs = extract_paragraphs(pages)
    chunks = create_chunks(paper_id="P001", paragraphs=paragraphs)

    assert len(chunks) == 4

    # Chunk 1 (Page 1, Paragraph 1, Section: Introduction)
    assert chunks[0].chunk_id == "P001_C001"
    assert chunks[0].paper_id == "P001"
    assert chunks[0].page_number == 1
    assert chunks[0].paragraph_number == 1
    assert chunks[0].section == "Introduction"
    assert "This research proposes a new approach" in chunks[0].text

    # Chunk 2 (Page 1, Paragraph 2, Section: Introduction)
    assert chunks[1].chunk_id == "P001_C002"
    assert chunks[1].paper_id == "P001"
    assert chunks[1].page_number == 1
    assert chunks[1].paragraph_number == 2
    assert chunks[1].section == "Introduction"
    assert chunks[1].text == "The system verifies each claim against original evidence passages."

    # Chunk 3 (Page 2, Paragraph 1, Section: Introduction - continued across page boundary)
    assert chunks[2].chunk_id == "P001_C003"
    assert chunks[2].paper_id == "P001"
    assert chunks[2].page_number == 2
    assert chunks[2].paragraph_number == 1
    assert chunks[2].section == "Introduction"
    assert chunks[2].text == "Continuing discussion from page one."

    # Chunk 4 (Page 2, Paragraph 2, Section: Methodology - updated heading on page 2)
    assert chunks[3].chunk_id == "P001_C004"
    assert chunks[3].paper_id == "P001"
    assert chunks[3].page_number == 2
    assert chunks[3].paragraph_number == 2
    assert chunks[3].section == "Methodology"
    assert chunks[3].text == "The architecture integrates PyMuPDF and Pydantic schemas."
