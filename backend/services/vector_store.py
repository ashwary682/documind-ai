"""FAISS vector store.

Each document gets its own folder inside the storage directory:
    storage/<document_id>/index.faiss   the vectors
    storage/<document_id>/chunks.json   the chunk text and metadata
    storage/<document_id>/meta.json     embedding model, vector size, version

The vectors are normalized, so the inner product between two vectors equals
their cosine similarity. A HIGHER score means a MORE similar chunk. The score
is not a probability.
"""

import json
import re
import shutil
from pathlib import Path

import faiss
import numpy as np

DEFAULT_STORAGE_DIR = "storage"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"
INDEX_VERSION = 1

# Only these characters are allowed in a document id. This stops a value such
# as "../secret" from pointing outside the storage folder.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _document_dir(document_id, storage_dir):
    """Return the folder path of a document after checking its id."""
    if not _SAFE_ID.match(document_id):
        raise ValueError(
            "document_id may only contain letters, digits, '-' and '_'"
        )
    return Path(storage_dir) / document_id


def save_index(
    document_id, chunks, vectors, model_name, storage_dir=DEFAULT_STORAGE_DIR
):
    """Build a FAISS index from chunk vectors and save it with its metadata.

    chunks:  list of chunk dictionaries (from chunker.py)
    vectors: array of shape (number_of_chunks, vector_size)
    The i-th vector must belong to the i-th chunk.
    """
    vectors = np.ascontiguousarray(vectors, dtype="float32")

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if len(chunks) != vectors.shape[0]:
        raise ValueError("chunks and vectors must have the same length")

    dimension = vectors.shape[1]

    # IndexFlatIP compares the query with every stored vector (exact search).
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    folder = _document_dir(document_id, storage_dir)
    folder.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(folder / INDEX_FILE))

    # encoding="utf-8" avoids problems with special characters on Windows.
    (folder / CHUNKS_FILE).write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "document_id": document_id,
        "embedding_model": model_name,
        "dimension": dimension,
        "num_chunks": len(chunks),
        "index_type": "IndexFlatIP",
        "metric": "inner_product (cosine for normalized vectors)",
        "index_version": INDEX_VERSION,
    }
    (folder / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def load_index(document_id, storage_dir=DEFAULT_STORAGE_DIR):
    """Load the index, the chunks and the metadata of one document."""
    folder = _document_dir(document_id, storage_dir)

    index_path = folder / INDEX_FILE
    chunks_path = folder / CHUNKS_FILE
    meta_path = folder / META_FILE

    for path in (index_path, chunks_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file for '{document_id}': {path}")

    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # The three files must describe the same data. If they do not, the index
    # is damaged and should be rebuilt instead of returning wrong citations.
    if not (index.ntotal == len(chunks) == meta["num_chunks"]):
        raise ValueError(
            f"Index and metadata do not match for '{document_id}'. Rebuild it."
        )

    return index, chunks, meta


def search(
    document_id,
    query_vector,
    top_k=5,
    model_name=None,
    storage_dir=DEFAULT_STORAGE_DIR,
):
    """Return the top_k most similar chunks for a query vector.

    Each result is a copy of the chunk dictionary with two extra keys:
        "score": similarity score (higher is more similar)
        "rank":  1 for the best match, 2 for the next one, and so on
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    index, chunks, meta = load_index(document_id, storage_dir)

    # Vectors from different embedding models cannot be compared.
    if model_name is not None and model_name != meta["embedding_model"]:
        raise ValueError(
            f"Index was built with '{meta['embedding_model']}', "
            f"but the query uses '{model_name}'."
        )

    query_vector = np.ascontiguousarray(query_vector, dtype="float32")
    if query_vector.ndim != 2 or query_vector.shape[1] != index.d:
        raise ValueError(f"query_vector must have shape (1, {index.d})")

    # Never ask for more results than the number of stored vectors.
    k = min(top_k, index.ntotal)
    scores, positions = index.search(query_vector, k)

    results = []
    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        # FAISS uses -1 when it has no result for a slot.
        if position == -1:
            continue

        result = dict(chunks[position])
        result["score"] = float(score)
        result["rank"] = rank
        results.append(result)

    return results


def delete_index(document_id, storage_dir=DEFAULT_STORAGE_DIR):
    """Delete all stored files of a document. Return True if anything was deleted."""
    folder = _document_dir(document_id, storage_dir)

    if not folder.exists():
        return False

    shutil.rmtree(folder)
    return True