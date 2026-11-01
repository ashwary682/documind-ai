"""Retriever.

This module connects the embedding model and the FAISS index. Given a
question, it returns the most relevant chunks of one document.

An empty result list means "nothing relevant was found". The caller should
then answer that the information is not in the document instead of asking
the language model to guess.
"""

from backend.services.embeddings import DEFAULT_MODEL_NAME, embed_query
from backend.services.vector_store import DEFAULT_STORAGE_DIR, search

DEFAULT_TOP_K = 5


def retrieve(
    question,
    document_id,
    top_k=DEFAULT_TOP_K,
    min_score=None,
    model_name=DEFAULT_MODEL_NAME,
    storage_dir=DEFAULT_STORAGE_DIR,
):
    """Return the chunks that best match the question.

    question:  the user's question as plain text
    top_k:     maximum number of chunks to return
    min_score: if set, chunks with a lower score are removed. Leave it as
               None until a good value has been measured on test questions.

    Each result is a chunk dictionary with these extra keys:
        "score":     similarity score (higher is more similar)
        "rank":      1 for the best match
        "source_id": "S1", "S2", ... (a short id the language model can cite)
    """
    # The question must use the same model that built the index.
    query_vector = embed_query(question, model_name=model_name)

    results = search(
        document_id,
        query_vector,
        top_k=top_k,
        model_name=model_name,
        storage_dir=storage_dir,
    )

    # Optional score filter.
    if min_score is not None:
        results = [result for result in results if result["score"] >= min_score]

    # Source ids are given after filtering, so they always start at S1.
    for position, result in enumerate(results, start=1):
        result["source_id"] = f"S{position}"

    return results