"""
extraction.py  –  Step 1 of 2 for building the knowledge base.

Run this script manually whenever you add new PDFs:
    python "tools/vector_db components/extraction.py"

It scans the pdfs/ folder at the project root, extracts text from every PDF,
splits it into fixed-size chunks, and saves them to chunks.json in the same
folder so ingestion.py can load them in Step 2.
"""

import os
import json
from typing import List
import PyPDF2


# ── Paths (relative to project root) ─────────────────────────────────────────
_HERE       = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PDF_FOLDER  = os.path.join(PROJECT_ROOT, "pdfs")
CHUNKS_FILE = os.path.join(PDF_FOLDER, "chunks.json")

CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between consecutive chunks



# ── Core functions ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Return all text extracted from a single PDF file."""
    full_text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def extract_pdfs_to_chunks(
    pdf_paths: List[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Extract and chunk text from multiple PDFs using overlapping windows.

    Args:
        pdf_paths:  Paths to the PDF files to process.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters each chunk shares with the previous one.
                    Prevents context loss at chunk boundaries.

    Returns:
        A flat list of overlapping text chunks.
    """
    combined_text = ""
    for path in pdf_paths:
        if not os.path.isfile(path):
            print(f"  [!] Skipping (not found): {path}")
            continue
        print(f"  → Extracting: {os.path.basename(path)}")
        combined_text += extract_text_from_pdf(path) + "\n"

    if not combined_text.strip():
        return []

    # Overlapping sliding window
    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(combined_text):
        chunks.append(combined_text[i : i + chunk_size])
        i += step

    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(PDF_FOLDER):
        print(f"[extraction] 'pdfs/' folder not found at: {PDF_FOLDER}")
        print("[extraction] Create the folder and add your ServiceNow PDFs, then run this script again.")
        raise SystemExit(1)

    pdf_files = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("[extraction] No PDF files found in 'pdfs/'. Add your PDFs and try again.")
        raise SystemExit(1)

    print(f"[extraction] Found {len(pdf_files)} PDF(s). Extracting...")
    chunks = extract_pdfs_to_chunks(pdf_files)

    if not chunks:
        print("[extraction] No text could be extracted. Check that your PDFs contain readable text.")
        raise SystemExit(1)

    # Save chunks so ingestion.py can load them
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[extraction] Done! {len(chunks)} chunks saved to: {CHUNKS_FILE}")
    print("[extraction] Next step: run ingestion.py to load them into ChromaDB.")

