import re
from dataclasses import dataclass
from typing import Optional

from backend.ingestion.pdf_parser import ExtractedPage


COMMON_HEADINGS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "methodology",
    "methods",
    "materials and methods",
    "experimental setup",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
]


@dataclass
class ExtractedParagraph:
    """
    Intermediate structured representation of an extracted paragraph.
    Note: Chunk and paper IDs are assigned in the subsequent chunking phase.
    """
    page_number: int
    paragraph_number: int
    section: Optional[str]
    text: str


def detect_section_heading(text: str) -> Optional[str]:
    """
    Conservatively detects if a string represents a section heading.

    Returns the identified normalized heading title, or None if unidentified.
    """
    line = text.strip()
    if not line or len(line) > 80:
        return None

    # Check unnumbered common heading (e.g., 'Introduction', 'Related Work:', 'Abstract')
    clean_line = re.sub(r"[:\-–—\s\.]+$", "", line).strip()
    if clean_line.lower() in COMMON_HEADINGS:
        return clean_line.title()

    # Exclude lines ending with typical sentence punctuation
    if re.search(r"[a-zA-Z]{2,}\.\s*$", line):
        return None
    if line.endswith(("?", "!", ";")):
        return None

    # Check numbered heading (e.g., '1 Introduction', '2. Methodology', '3.1 Dataset', 'Section 4: Results')
    prefix_match = re.match(
        r"^(?:(?:section|sec\.?)\s+)?((?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.:\s\-–—]+)\s*(.+)$",
        line,
        re.IGNORECASE,
    )
    if prefix_match:
        title_part = prefix_match.group(2).strip()
        title_part_clean = re.sub(r"[:\-–—\s\.]+$", "", title_part).strip()
        if title_part_clean and len(title_part_clean) <= 60:
            if not re.search(r"[a-zA-Z]{2,}\.\s*$", title_part_clean):
                if title_part_clean.lower() in COMMON_HEADINGS:
                    return title_part_clean.title()
                if title_part_clean[0].isupper():
                    return title_part_clean.title()

    return None


def extract_paragraphs(pages: list[ExtractedPage]) -> list[ExtractedParagraph]:
    """
    Reconstructs structured paragraphs from page-level text.

    Args:
        pages: List of ExtractedPage objects in original document order.

    Returns:
        list[ExtractedParagraph]: Structured paragraphs with 1-based page numbers,
        per-page reset paragraph numbers, identified section context, and verbatim text.
    """
    paragraphs: list[ExtractedParagraph] = []
    current_section: Optional[str] = None

    for page in pages:
        page_number = page.page_number
        raw_text = page.text

        if not raw_text or not raw_text.strip():
            # Skip empty or whitespace-only pages without inventing false paragraphs
            continue

        # Split text into blocks separated by blank lines (2 or more newlines)
        raw_blocks = re.split(r"\n\s*\n+", raw_text)
        paragraph_number = 1

        for raw_block in raw_blocks:
            cleaned_block = raw_block.strip()
            if not cleaned_block:
                continue

            # Check if this entire block is a standalone section heading
            standalone_heading = detect_section_heading(cleaned_block)
            if standalone_heading:
                current_section = standalone_heading
                continue

            # Check if the block begins with a heading line followed by paragraph body
            lines = cleaned_block.split("\n")
            if len(lines) >= 2:
                first_line = lines[0].strip()
                heading_on_first = detect_section_heading(first_line)
                if heading_on_first:
                    current_section = heading_on_first
                    body_text = "\n".join(lines[1:]).strip()
                    if body_text:
                        paragraphs.append(
                            ExtractedParagraph(
                                page_number=page_number,
                                paragraph_number=paragraph_number,
                                section=current_section,
                                text=body_text,
                            )
                        )
                        paragraph_number += 1
                    continue

                # Check two-line heading: "1" followed by "Introduction"
                if re.match(r"^\d+(\.\d+)*$", first_line) and len(lines) >= 2:
                    second_line = lines[1].strip()
                    cand_heading = detect_section_heading(f"{first_line} {second_line}")
                    if cand_heading:
                        current_section = cand_heading
                        body_text = "\n".join(lines[2:]).strip()
                        if body_text:
                            paragraphs.append(
                                ExtractedParagraph(
                                    page_number=page_number,
                                    paragraph_number=paragraph_number,
                                    section=current_section,
                                    text=body_text,
                                )
                            )
                            paragraph_number += 1
                        continue

            # Check for inline headings like "Abstract—..." or "Abstract: ..."
            inline_match = re.match(
                r"^(abstract|introduction|background|methodology|methods|experiments|results|discussion|conclusion|conclusions|limitations|future work)[\s—–\-:]+(.+)$",
                cleaned_block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if inline_match:
                current_section = inline_match.group(1).title()
                body_text = inline_match.group(2).strip()
                paragraphs.append(
                    ExtractedParagraph(
                        page_number=page_number,
                        paragraph_number=paragraph_number,
                        section=current_section,
                        text=body_text,
                    )
                )
                paragraph_number += 1
                continue

            # Standard body paragraph: retain internal line breaks and original characters
            paragraphs.append(
                ExtractedParagraph(
                    page_number=page_number,
                    paragraph_number=paragraph_number,
                    section=current_section,
                    text=cleaned_block,
                )
            )
            paragraph_number += 1

    return paragraphs


class ParagraphExtractor:
    """Paragraph extractor utility wrapper around extract_paragraphs."""

    @staticmethod
    def extract_paragraphs(pages: list[ExtractedPage]) -> list[ExtractedParagraph]:
        return extract_paragraphs(pages)
