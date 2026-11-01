"""Text chunking.

This module splits the text of each PDF page into smaller overlapping chunks.
Every chunk keeps metadata (chunk id, document id, page number) so that
answers can later be traced back to the exact page they came from.

Unit of measurement: CHARACTERS (not tokens or words).
"""

DEFAULT_CHUNK_SIZE = 700      # maximum number of characters in one chunk
DEFAULT_CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks


def _split_text(text, chunk_size, chunk_overlap):
    """Split one string into overlapping pieces.

    The function tries to end each piece at a space or a new line so that
    words are not cut in half. If no space is found, it cuts at chunk_size.
    """
    pieces = []
    length = len(text)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)

        # If this is not the last piece, try to move "end" back to the
        # nearest space or new line. We only look after (start + overlap)
        # so the piece is always longer than the overlap, which guarantees
        # that the loop moves forward.
        if end < length:
            search_from = start + chunk_overlap + 1
            last_space = max(
                text.rfind(" ", search_from, end),
                text.rfind("\n", search_from, end),
            )
            if last_space != -1:
                end = last_space

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        # Stop after the final piece.
        if end >= length:
            break

        # The next piece starts a little before the end of this one (overlap).
        start = end - chunk_overlap

    return pieces


def chunk_pages(
    pages,
    document_id,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """Turn a list of pages into a list of chunks with metadata.

    Input:  pages = [{"page_number": 1, "text": "..."}, ...]
    Output: [{"chunk_id": "chunk-001", "document_id": "doc-123",
              "page_number": 1, "text": "..."}, ...]
    """
    # Validate the settings before doing any work.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be 0 or more and less than chunk_size")

    chunks = []
    counter = 1  # keeps chunk ids unique across the whole document

    for page in pages:
        pieces = _split_text(page["text"], chunk_size, chunk_overlap)

        for piece in pieces:
            chunks.append(
                {
                    "chunk_id": f"chunk-{counter:03d}",
                    "document_id": document_id,
                    "page_number": page["page_number"],
                    "text": piece,
                }
            )
            counter += 1

    return chunks