from backend.ingestion.pdf_parser import ExtractedPage
from backend.ingestion.paragraph_extractor import extract_paragraphs, ExtractedParagraph


def test_one_paragraph_on_one_page():
    """1. One paragraph on one page."""
    pages = [
        ExtractedPage(page_number=1, text="This is a standalone single paragraph on the page.")
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 1
    assert isinstance(paragraphs[0], ExtractedParagraph)
    assert paragraphs[0].page_number == 1
    assert paragraphs[0].paragraph_number == 1
    assert paragraphs[0].text == "This is a standalone single paragraph on the page."


def test_multiple_paragraphs_on_one_page():
    """2. Multiple paragraphs on one page."""
    text = (
        "First paragraph on page one discussing the motivation.\n\n"
        "Second paragraph on page one introducing the core hypothesis.\n\n"
        "Third paragraph on page one summarizing the contributions."
    )
    pages = [ExtractedPage(page_number=1, text=text)]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 3
    assert [p.paragraph_number for p in paragraphs] == [1, 2, 3]
    assert paragraphs[0].text.startswith("First paragraph")
    assert paragraphs[1].text.startswith("Second paragraph")
    assert paragraphs[2].text.startswith("Third paragraph")


def test_multiple_pages():
    """3. Multiple pages."""
    pages = [
        ExtractedPage(page_number=1, text="Content belonging to page one."),
        ExtractedPage(page_number=2, text="Content belonging to page two."),
        ExtractedPage(page_number=3, text="Content belonging to page three."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 3
    assert paragraphs[0].page_number == 1
    assert paragraphs[1].page_number == 2
    assert paragraphs[2].page_number == 3


def test_paragraph_numbers_reset_on_new_page():
    """4. Paragraph numbers resetting on a new page."""
    pages = [
        ExtractedPage(page_number=1, text="Page 1 Para 1.\n\nPage 1 Para 2."),
        ExtractedPage(page_number=2, text="Page 2 Para 1.\n\nPage 2 Para 2.\n\nPage 2 Para 3."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 5
    # Page 1 paragraph numbers
    assert paragraphs[0].paragraph_number == 1 and paragraphs[0].page_number == 1
    assert paragraphs[1].paragraph_number == 2 and paragraphs[1].page_number == 1
    # Page 2 paragraph numbers must reset to 1
    assert paragraphs[2].paragraph_number == 1 and paragraphs[2].page_number == 2
    assert paragraphs[3].paragraph_number == 2 and paragraphs[3].page_number == 2
    assert paragraphs[4].paragraph_number == 3 and paragraphs[4].page_number == 2


def test_page_number_preservation():
    """5. Correct page-number preservation."""
    pages = [
        ExtractedPage(page_number=7, text="Paragraph located on page seven."),
        ExtractedPage(page_number=15, text="Paragraph located on page fifteen."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert paragraphs[0].page_number == 7
    assert paragraphs[1].page_number == 15


def test_section_heading_detection():
    """6. Section heading detection for common and numbered formats."""
    pages = [
        ExtractedPage(page_number=1, text="Introduction\n\nBody text for introduction."),
        ExtractedPage(page_number=2, text="3.1 Dataset\n\nDataset description and collection methods."),
        ExtractedPage(page_number=3, text="Section 4: Results\n\nExperimental evaluation findings."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 3
    assert paragraphs[0].section == "Introduction"
    assert paragraphs[1].section == "Dataset"
    assert paragraphs[2].section == "Results"


def test_section_assignment_to_subsequent_paragraphs():
    """7. Section assignment to subsequent paragraphs."""
    pages = [
        ExtractedPage(
            page_number=1,
            text=(
                "2. Methodology\n\n"
                "First paragraph describing model architecture.\n\n"
                "Second paragraph detailing the loss function.\n\n"
                "Third paragraph listing training hyperparameters."
            ),
        )
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 3
    for p in paragraphs:
        assert p.section == "Methodology"


def test_section_continuity_across_pages():
    """8. Section continuity across pages until a new heading is detected."""
    pages = [
        ExtractedPage(page_number=1, text="Introduction\n\nIntroduction start on page 1."),
        ExtractedPage(page_number=2, text="Continuing introduction on page 2."),
        ExtractedPage(page_number=3, text="Methods\n\nMethods paragraph on page 3."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 3
    assert paragraphs[0].section == "Introduction"
    assert paragraphs[1].section == "Introduction"  # Carried over to page 2
    assert paragraphs[2].section == "Methods"       # Updated by new heading on page 3


def test_null_section_when_no_heading_identifiable():
    """9. Null section when no heading is identifiable."""
    pages = [
        ExtractedPage(
            page_number=1,
            text="This is an opening paragraph with no title or recognized heading preceding it."
        )
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 1
    assert paragraphs[0].section is None


def test_empty_page_handling():
    """10. Empty page handling (blank, whitespace-only, or no text)."""
    pages = [
        ExtractedPage(page_number=1, text=""),
        ExtractedPage(page_number=2, text="   \n\n   \t  \n"),
        ExtractedPage(page_number=3, text="Valid content on page three."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 1
    assert paragraphs[0].page_number == 3
    assert paragraphs[0].paragraph_number == 1
    assert paragraphs[0].text == "Valid content on page three."


def test_preservation_of_original_paragraph_text():
    """11. Preservation of original paragraph text (formulas, punctuation, casing, numbers)."""
    original = "Model achieved 94.85% accuracy on CIFAR-10; Loss: L = -sum(y * log(p))."
    pages = [ExtractedPage(page_number=1, text=original)]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 1
    assert paragraphs[0].text == original


def test_preservation_of_paragraph_order():
    """12. Preservation of paragraph order."""
    text = "Paragraph Alpha.\n\nParagraph Beta.\n\nParagraph Gamma."
    pages = [ExtractedPage(page_number=1, text=text)]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 3
    assert paragraphs[0].text == "Paragraph Alpha."
    assert paragraphs[1].text == "Paragraph Beta."
    assert paragraphs[2].text == "Paragraph Gamma."


def test_single_newlines_not_treated_as_paragraph_boundaries():
    """13. Single newlines not automatically becoming paragraph boundaries."""
    text_with_line_wrapping = (
        "This is line 1 of a wrapped paragraph in a multi-column PDF.\n"
        "This is line 2 of the exact same paragraph continuing the thought.\n"
        "This is line 3 concluding the paragraph."
    )
    pages = [ExtractedPage(page_number=1, text=text_with_line_wrapping)]
    paragraphs = extract_paragraphs(pages)

    # Must remain exactly 1 paragraph
    assert len(paragraphs) == 1
    assert paragraphs[0].text == text_with_line_wrapping


def test_heading_without_body_text_updates_subsequent_pages():
    """Handling pages containing headings without body text."""
    pages = [
        ExtractedPage(page_number=1, text="Introduction\n\nIntro content on page 1."),
        ExtractedPage(page_number=2, text="Results"),  # Heading only on page 2
        ExtractedPage(page_number=3, text="Results discussion and metrics on page 3."),
    ]
    paragraphs = extract_paragraphs(pages)

    assert len(paragraphs) == 2
    assert paragraphs[0].page_number == 1
    assert paragraphs[0].section == "Introduction"
    assert paragraphs[1].page_number == 3
    assert paragraphs[1].section == "Results"  # Successfully inherited from page 2 heading
