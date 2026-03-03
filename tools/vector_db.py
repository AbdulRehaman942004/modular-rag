"""
vector_db.py  –  Vector-DB tool for the Modular-RAG orchestrator.

Assumes the ChromaDB collection is already populated.
To build / refresh the knowledge base, manually run:
    1. python "tools/vector_db components/extraction.py"
    2. python "tools/vector_db components/ingestion.py"
"""

import os
import sys

# Make component files importable regardless of working directory
_COMPONENTS_DIR = os.path.join(os.path.dirname(__file__), "vector_db components")
if _COMPONENTS_DIR not in sys.path:
    sys.path.insert(0, _COMPONENTS_DIR)

from generation import run_rag_query  # noqa: E402

COLLECTION_NAME = "servicenow_knowledge_base"
N_RESULTS = 10  # number of chunks to retrieve per query


def vector_db(query: str) -> str:
    """Retrieve relevant context from ChromaDB and return a grounded answer.

    The ChromaDB collection must already be populated by running
    extraction.py and ingestion.py beforehand.

    Args:
        query: The user's question forwarded by the orchestrator.

    Returns:
        A grounded answer string to be used as context by the orchestrator.
    """
    print(f"[vector_db] Running RAG query: {query!r}")

    try:
        answer, context_chunks = run_rag_query(
            user_query=query,
            collection_name=COLLECTION_NAME,
            n_results=N_RESULTS,
        )
        print(f"[vector_db] Retrieved {len(context_chunks)} context chunk(s).")
        return answer

    except Exception as e:
        return (
            f"[vector_db] Error: Could not query the knowledge base. "
            f"Make sure you have run extraction.py and ingestion.py first.\n"
            f"Details: {e}"
        )
