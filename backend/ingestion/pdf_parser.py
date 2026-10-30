from dataclasses import dataclass
import pymupdf


class InvalidPDFError(Exception):
    """Raised when the PDF file is corrupted, invalid, or cannot be opened."""
    pass


class PasswordProtectedPDFError(Exception):
    """Raised when the PDF file is encrypted or password-protected."""
    pass


class EmptyPDFError(Exception):
    """Raised when the PDF file contains no bytes or zero pages."""
    pass


class NoExtractableTextError(Exception):
    """Raised when the PDF contains pages, but no extractable text is found (e.g., scanned PDF)."""
    pass


@dataclass
class ExtractedPage:
    """
    Internal structure representing text extracted from a single PDF page.
    Note: This is an internal representation and does not replace the shared Chunk schema.
    """
    page_number: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes) -> list[ExtractedPage]:
    """
    Extracts text separately from every page of a PDF using PyMuPDF.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        list[ExtractedPage]: List of pages with 1-based page numbers and original text.

    Raises:
        EmptyPDFError: If the input bytes are empty or the PDF has zero pages.
        InvalidPDFError: If the input is not a valid or readable PDF document.
        PasswordProtectedPDFError: If the PDF requires a password to open.
        NoExtractableTextError: If no extractable text was found on any page (OCR may be required).
    """
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise InvalidPDFError("Expected PDF content as bytes or bytearray.")

    if len(pdf_bytes) == 0:
        raise EmptyPDFError("The provided PDF file is empty (0 bytes).")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise InvalidPDFError("Failed to parse PDF document: file is invalid or corrupted.") from None

    try:
        if doc.needs_pass or (doc.is_encrypted and not doc.authenticate("")):
            raise PasswordProtectedPDFError(
                "The PDF document is password-protected and cannot be opened without credentials."
            )

        if len(doc) == 0:
            raise EmptyPDFError("The PDF document contains zero pages.")

        extracted_pages: list[ExtractedPage] = []
        has_any_text = False

        for idx in range(len(doc)):
            page = doc[idx]
            page_number = idx + 1
            # Retain original text extracted by PyMuPDF without alterations
            page_text = page.get_text("text")

            if page_text.strip():
                has_any_text = True

            # Preserve exact text returned by PyMuPDF; include empty pages without skipping
            extracted_pages.append(ExtractedPage(page_number=page_number, text=page_text))

        if not has_any_text:
            raise NoExtractableTextError(
                "No extractable text found in the PDF. The document may be scanned or image-based; OCR may be required."
            )

        return extracted_pages
    finally:
        doc.close()


class PDFParser:
    """PDF parser utility wrapper around extract_pdf_pages."""

    @staticmethod
    def extract_pages(pdf_bytes: bytes) -> list[ExtractedPage]:
        return extract_pdf_pages(pdf_bytes)
