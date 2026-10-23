"""Temporary script to test the PDF parser by hand."""

from backend.services.pdf_parser import extract_pages

# Change this path if your PDF has a different name.
PDF_PATH = "CD_unit1.pdf"

pages = extract_pages(PDF_PATH)

print(f"Pages with text: {len(pages)}")

# Show the first 200 characters of the first page to check the text quality.
if pages:
    print(f"\nFirst page number: {pages[0]['page_number']}")
    print("First 200 characters:")
    print(pages[0]["text"][:200])
else:
    print("No text found. The PDF may be scanned or image-only.")