"""
ingestion.py  –  Step 2 of 2 for building the knowledge base.

Run this script manually after extraction.py:
    python "tools/vector_db components/ingestion.py"

It loads the chunks saved by extraction.py (pdfs/chunks.json) and stores
them in a persistent ChromaDB collection ready for querying.
"""

import os
import json
from typing import List, Tuple
import chromadb


# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PDF_FOLDER   = os.path.join(PROJECT_ROOT, "pdfs")
CHUNKS_FILE  = os.path.join(PDF_FOLDER, "chunks.json")
CHROMADB_PATH = os.path.join(PROJECT_ROOT, "chromadb_data")

DEFAULT_COLLECTION = "servicenow_knowledge_base"


# ── Core functions ────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    os.makedirs(CHROMADB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMADB_PATH)


def ingest_chunks(
    chunks: List[str],
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = True,
) -> Tuple[chromadb.Collection, List[str]]:
    """Store text chunks in a ChromaDB collection.

    Args:
        chunks:          List of text chunks to store.
        collection_name: Name of the ChromaDB collection.
        reset:           If True (default), wipe the collection before ingesting
                         so stale data from old PDFs is removed.

    Returns:
        A tuple of (collection, list_of_chunk_ids).
    """
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(name=collection_name)
            print(f"  [reset] Deleted old collection: '{collection_name}'")
        except Exception:
            pass  # didn't exist yet

    collection = client.get_or_create_collection(name=collection_name)
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=chunk_ids, documents=chunks)
    return collection, chunk_ids


def collection_exists(collection_name: str = DEFAULT_COLLECTION) -> bool:
    """Return True if the named collection already exists and has documents."""
    try:
        col = get_chroma_client().get_collection(name=collection_name)
        return col.count() > 0
    except Exception:
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isfile(CHUNKS_FILE):
        print(f"[ingestion] chunks.json not found at: {CHUNKS_FILE}")
        print("[ingestion] Please run extraction.py first.")
        raise SystemExit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[ingestion] Loaded {len(chunks)} chunks from chunks.json")
    print("[ingestion] Ingesting into ChromaDB (appending to existing data)...")

    collection, ids = ingest_chunks(chunks, reset=False)

    print(f"\n[ingestion] Done! {collection.count()} chunks now in collection '{DEFAULT_COLLECTION}'.")
    print("[ingestion] Your knowledge base is ready. You can now run the main application.")

