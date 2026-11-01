"""Embedding generation.

This module turns text into numeric vectors using a Sentence Transformers model.
The same model must be used for document chunks and for user questions,
otherwise their vectors cannot be compared.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The model is large, so we load it only once and reuse it.
_model = None
_model_name = None


def get_model(model_name=DEFAULT_MODEL_NAME):
    """Load the embedding model the first time, then return the cached copy."""
    global _model, _model_name

    if _model is None or _model_name != model_name:
        _model = SentenceTransformer(model_name)
        _model_name = model_name

    return _model


def embed_texts(texts, model_name=DEFAULT_MODEL_NAME, batch_size=32):
    """Convert a list of texts into a 2D array of normalized vectors.

    The result has shape (number_of_texts, vector_size) and type float32.
    float32 is the type that FAISS expects.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    model = get_model(model_name)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        # Normalizing makes every vector length 1, so the dot product of two
        # vectors equals their cosine similarity.
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype="float32")


def embed_query(question, model_name=DEFAULT_MODEL_NAME):
    """Convert one question into a vector of shape (1, vector_size)."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    return embed_texts([question], model_name=model_name)