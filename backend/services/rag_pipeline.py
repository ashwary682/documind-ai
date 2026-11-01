"""RAG pipeline.

This module connects retrieval and answer generation:
    question -> retrieve chunks -> (abstain if nothing relevant) -> LLM -> verify citations

Page numbers in the result always come from the stored chunk metadata. They
are never taken from the text written by the language model.
"""

import os
import re

from dotenv import load_dotenv

from backend.services.llm_service import generate_answer
from backend.services.retriever import DEFAULT_TOP_K, retrieve

load_dotenv()

NOT_FOUND_MESSAGE = "I could not find this information in the document."

# Matches source ids such as S1, S2, S10 inside the answer text.
_SOURCE_PATTERN = re.compile(r"\bS(\d+)\b")


def _min_score_from_env():
    """Read MIN_RETRIEVAL_SCORE from .env. Return None if it is empty."""
    raw_value = os.getenv("MIN_RETRIEVAL_SCORE", "").strip()
    if not raw_value:
        return None

    try:
        return float(raw_value)
    except ValueError:
        raise ValueError("MIN_RETRIEVAL_SCORE in .env must be a number")


def extract_citations(answer, sources):
    """Find the source ids used in the answer and check them against the sources.

    Ids that were not retrieved (for example an invented "S9") are ignored.
    Each id is returned once, in the order it first appears in the answer.
    """
    sources_by_id = {source["source_id"]: source for source in sources}

    citations = []
    seen = set()

    for number in _SOURCE_PATTERN.findall(answer):
        source_id = f"S{number}"

        if source_id in seen or source_id not in sources_by_id:
            continue

        seen.add(source_id)
        source = sources_by_id[source_id]
        citations.append(
            {
                "source_id": source_id,
                "chunk_id": source["chunk_id"],
                "page_number": source["page_number"],
                "score": source["score"],
            }
        )

    return citations


def answer_question(question, document_id, top_k=DEFAULT_TOP_K, min_score=None):
    """Answer a question about one document.

    Returns a dictionary:
        "answer":    the answer text
        "abstained": True if no relevant chunk was found and the LLM was not used
        "citations": verified sources that the answer refers to
        "sources":   every chunk that was sent to the LLM
    """
    # An explicit argument wins over the value in .env.
    if min_score is None:
        min_score = _min_score_from_env()

    sources = retrieve(question, document_id, top_k=top_k, min_score=min_score)

    # Nothing relevant: do not ask the LLM to guess.
    if not sources:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "abstained": True,
            "citations": [],
            "sources": [],
        }

    answer = generate_answer(question, sources)

    return {
        "answer": answer,
        "abstained": False,
        "citations": extract_citations(answer, sources),
        "sources": sources,
    }