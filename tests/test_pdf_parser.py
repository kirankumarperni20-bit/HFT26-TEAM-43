import pytest
import pymupdf

from backend.ingestion.pdf_parser import (
    extract_pdf_pages,
    ExtractedPage,
    InvalidPDFError,
    PasswordProtectedPDFError,
    EmptyPDFError,
    NoExtractableTextError,
)


def _create_single_page_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_multi_page_pdf(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_password_protected_pdf(text: str, password: str = "secret") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="admin",
        user_pw=password,
    )
    doc.close()
    return pdf_bytes


def test_single_page_pdf_extraction():
    """1. Successful extraction from a one-page PDF."""
    sample_text = "This is a single page research paper summary."
    pdf_bytes = _create_single_page_pdf(sample_text)

    pages = extract_pdf_pages(pdf_bytes)

    assert len(pages) == 1
    assert isinstance(pages[0], ExtractedPage)
    assert sample_text in pages[0].text
    assert pages[0].page_number == 1


def test_multi_page_pdf_extraction():
    """2. Successful extraction from a multi-page PDF."""
    texts = [
        "Page 1: Abstract and Introduction.",
        "Page 2: Methodology and Experimental Setup.",
        "Page 3: Results and Conclusion.",
    ]
    pdf_bytes = _create_multi_page_pdf(texts)

    pages = extract_pdf_pages(pdf_bytes)

    assert len(pages) == 3
    for i, expected_text in enumerate(texts):
        assert expected_text in pages[i].text


def test_one_based_page_numbering():
    """3. Correct 1-based page numbering."""
    pdf_bytes = _create_multi_page_pdf(["Section A", "Section B", "Section C"])

    pages = extract_pdf_pages(pdf_bytes)

    page_numbers = [p.page_number for p in pages]
    assert page_numbers == [1, 2, 3]


def test_preservation_of_page_order():
    """4. Preservation of page order."""
    ordered_snippets = ["First Page Content", "Second Page Content", "Third Page Content"]
    pdf_bytes = _create_multi_page_pdf(ordered_snippets)

    pages = extract_pdf_pages(pdf_bytes)

    assert len(pages) == 3
    assert "First Page Content" in pages[0].text
    assert "Second Page Content" in pages[1].text
    assert "Third Page Content" in pages[2].text


def test_preservation_of_exact_extracted_text():
    """5. Preservation of exact extracted text (no lowercase, no summary, no formula modification)."""
    exact_text = "Formula: E = mc^2; Accuracy = 94.85%; Greek: alpha, beta."
    pdf_bytes = _create_single_page_pdf(exact_text)

    pages = extract_pdf_pages(pdf_bytes)

    assert len(pages) == 1
    # Check that casing, mathematical symbols, and numeric values are retained
    assert exact_text in pages[0].text
    assert "E = mc^2" in pages[0].text
    assert "94.85%" in pages[0].text


def test_empty_page_handling():
    """6. Empty page handling (retains empty page without skipping it)."""
    # Page 1 has text, Page 2 is blank, Page 3 has text
    pdf_bytes = _create_multi_page_pdf(["Page 1 Content", "", "Page 3 Content"])

    pages = extract_pdf_pages(pdf_bytes)

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert "Page 1 Content" in pages[0].text

    assert pages[1].page_number == 2
    assert pages[1].text.strip() == ""

    assert pages[2].page_number == 3
    assert "Page 3 Content" in pages[2].text


def test_invalid_pdf_handling():
    """7. Invalid PDF handling (corrupted bytes or non-PDF data)."""
    with pytest.raises(InvalidPDFError):
        extract_pdf_pages(b"This is just plain text, not a valid PDF file.")

    with pytest.raises(InvalidPDFError):
        extract_pdf_pages(b"%PDF-1.4\ncorrupted incomplete binary data...")


def test_empty_pdf_handling():
    """8. Empty PDF handling (0 bytes)."""
    with pytest.raises(EmptyPDFError):
        extract_pdf_pages(b"")


def test_no_extractable_text_handling():
    """9. No-extractable-text handling (blank or scanned image-only PDF)."""
    # Create PDF with 2 completely blank pages
    pdf_bytes = _create_multi_page_pdf(["", ""])

    with pytest.raises(NoExtractableTextError) as exc_info:
        extract_pdf_pages(pdf_bytes)

    assert "OCR may be required" in str(exc_info.value)


def test_password_protected_pdf_handling():
    """10. Password-protected PDF handling."""
    pdf_bytes = _create_password_protected_pdf("Confidential Research", password="secure_pass")

    with pytest.raises(PasswordProtectedPDFError) as exc_info:
        extract_pdf_pages(pdf_bytes)

    assert "password-protected" in str(exc_info.value).lower()
