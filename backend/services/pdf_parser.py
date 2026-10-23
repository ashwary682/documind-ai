"""PDF text extraction.

This module reads a PDF file and returns the text of each page together
with its page number. The page number is kept because it is needed later
to show citations (for example: "Answer found on page 4").
"""

from pathlib import Path

# PyMuPDF is installed as "pymupdf", but it is imported as "fitz".
import fitz


def extract_pages(pdf_path):
    """Read a PDF and return a list of pages that contain text.

    Each item in the returned list is a dictionary:
        {"page_number": 1, "text": "text of the page..."}

    Pages with no text (for example, pages that are only images) are skipped.
    """
    path = Path(pdf_path)

    # Stop early with a clear message if the file does not exist.
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    pages = []

    # "with" closes the PDF automatically when we are done with it.
    with fitz.open(str(path)) as doc:
        # enumerate(..., start=1) makes page numbers start at 1, like a real book.
        for page_number, page in enumerate(doc, start=1):
            # get_text() returns the plain text of the page.
            # strip() removes extra spaces and blank lines at the start and end.
            text = page.get_text().strip()

            # Skip pages that have no text.
            if not text:
                continue

            pages.append({"page_number": page_number, "text": text})

    return pages