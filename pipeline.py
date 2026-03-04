"""
pipeline.py  –  Clean, side-effect-free API for the Streamlit frontend.

Call run_query(query) to get a structured result dict. All module-level
side effects (prints, LLM calls at import time) are avoided here.
"""

import os
import sys
import re

# Project root on sys.path
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Component imports — functions only, no module-level execution
from LLM import call_groq, call_groq_stream, call_chatgpt_stream
from query_refinement import confidence_score
from query_decomposer import query_decomposer
from tools.llm_response import llm_response
from tools.vector_db import vector_db
from tools.web_search import web_search

# ── Constants (mirrors router.py) ────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.60

ROUTER_PROMPT = """
System Role
You are the Router for "Now Assist," an AI assistant for the ServiceNow platform.
Your sole responsibility is to analyze the incoming user QUERY and determine the
correct tool(s) required to fulfill the request.

Available Tools

web_search:
Use when: The query requires Company's information, live data, historical data,
current trends, statistics, or up-to-date external information.

vector_db:
Use when: The query asks for theoretical information, definitions, concepts, or
foundational knowledge specific to ServiceNow modules. The database contains all
learning materials for basic and advanced ServiceNow understanding.

llm_response:
Use when: The query asks for specific technical use cases, code debugging, script
generation, logical reasoning, OR when the user is simply greeting the assistant (e.g., "Hello", "Who are you?").

Output Rules
- Respond with ONLY tool names from: web_search, vector_db, llm_response
- Separate multiple tools with commas (no spaces): tool1,tool2
- No explanations, no punctuation other than commas.

QUERY: {query}
RESPONSE:
"""

FINAL_PROMPT = """You are "Now Assist", an expert AI assistant for the ServiceNow platform.
You possess a deep, inherent knowledge of all things ServiceNow.

INFORMATION GATHERED:
{context}

INSTRUCTIONS:
- Keep the token limit of the response under 1024 tokens.
- You must answer the user's latest query factually and confidently, taking into account the conversation history.
- Use the INFORMATION GATHERED above to form your answer, but ACT AS IF you already knew this information. 
- CRITICAL: Output your answer directly in standard Markdown.
- You must leave a blank empty line after every heading before starting a paragraph.
- Do NOT wrap your entire response or headings in quotation marks or bold asterisks (e.g. use `# Heading` NOT `**# Heading**` or `# **Heading**`).
- Synthesize the details into a highly detailed, comprehensive, and professional response.
- Use headings, bullet points, numbered lists, and bold text extensively to organize the information and provide a structured output.
- EXCEPTION TO STRUCTURE: If the user explicitly asks for a specific format or constraint (e.g., "summarize in one paragraph", "make it short", "only give me the code"), you MUST prioritize their request and ignore the requirement to be "highly detailed" or use headings/bullets.
- If the INFORMATION GATHERED is insufficient to answer the query, say so clearly (e.g., "I don't have enough information to answer that.").
- Do NOT hallucinate.

CRITICAL DIRECTIVE: 
You are speaking directly to an end-user who does not know how you work.
You MUST NOT use phrases like "Based on the provided context", "According to the search results", "Source 1 says", or "The information gathered indicates".
NEVER mention the words "context", "search results", "sources", or "provided context". Present everything as your own expert knowledge.
"""

# ── Internal helpers ──────────────────────────────────────────────────────────

def _route(query: str) -> list[str]:
    prompt = ROUTER_PROMPT.format(query=query)
    raw = call_groq(prompt).strip()
    return re.split(r'\s*,\s*', raw)


def _run_tool(tool_name: str, q: str, n_results: int = 10, status_cb=None) -> str:
    """Run a tool with fallback to llm_response on failure."""
    if status_cb:
        status_cb(tool_name)
    try:
        if tool_name == "llm_response":
            result = llm_response(q)
        elif tool_name == "vector_db":
            result = vector_db(q, n_results=n_results)
        elif tool_name == "web_search":
            result = web_search(q)
        else:
            result = ""

        if not result or result.strip().startswith("["):
            raise ValueError("unusable result")
        return result

    except Exception as e:
        print(f"[pipeline] '{tool_name}' failed: {e} — falling back to llm_response")
        try:
            fallback = llm_response(q)
            if fallback:
                return f"[Fallback from {tool_name}]\n{fallback}"
        except Exception:
            pass
        return "Unable to retrieve information. Please try again."


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(query: str, chat_history: list = None, n_results: int = 10, status_cb=None) -> dict:
    """Run the full RAG pipeline and return a structured result.

    Args:
        query:        The user's question.
        chat_history: Optional list of dicts [{"role": "user"/"assistant", "content": "..."}] 
                      representing the conversation history.
        status_cb:    Optional callable(tool_name: str) called before each tool runs.

    Returns:
        {
            "answer":      str (or generator if streaming),
            "tools_used":  list[str],
            "confidence":  float,
            "is_relevant": bool,
            "sub_queries": list[str],
        }
    """
    if chat_history is None:
        chat_history = []

    # Step 1 — confidence scoring
    confidence = confidence_score(query, chat_history)
    if confidence < RELEVANCE_THRESHOLD:
        return {
            "answer": f"⚠️ Your question doesn't appear to be related to ServiceNow (Confidence: {confidence:.0%}). Please ask a ServiceNow-related question.",
            "tools_used": [],
            "confidence": confidence,
            "is_relevant": False,
            "sub_queries": [],
        }

    # Step 2 — routing
    tools = _route(query)

    # Step 3 — decompose (if multi-tool) and run
    if len(tools) == 1:
        sub_queries = [query]
        context = _run_tool(tools[0], query, n_results, status_cb)
    else:
        sub_queries = query_decomposer(query, tools)
        # Guard length mismatch
        pairs = list(zip(tools, sub_queries))
        contexts = [_run_tool(tool, sq, n_results, status_cb) for tool, sq in pairs]
        context = "\n\n".join(contexts)
        sub_queries = [sq for _, sq in pairs]
        tools = [t for t, _ in pairs]

    # Step 4 — final synthesis via messages array
    if status_cb:
        status_cb("synthesizing")
        
    system_prompt = FINAL_PROMPT.format(context=context)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})
    
    answer_generator = call_groq_stream(messages=messages)

    return {
        "answer": answer_generator,
        "tools_used": tools,
        "confidence": confidence,
        "is_relevant": True,
        "sub_queries": sub_queries,
    }


def get_collection_info() -> dict:
    """Return info about the current ChromaDB knowledge base."""
    try:
        import chromadb
        db_path = os.path.join(ROOT, "chromadb_data")
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection(name="servicenow_knowledge_base")
        return {"exists": True, "count": col.count()}
    except Exception:
        return {"exists": False, "count": 0}


def ingest_pdfs(pdf_files) -> dict:
    """Save uploaded Streamlit file objects to pdfs/ and run extraction + ingestion.

    Args:
        pdf_files: list of Streamlit UploadedFile objects.

    Returns:
        {"success": bool, "chunks": int, "message": str}
    """
    import json
    pdf_dir = os.path.join(ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    saved = []
    for f in pdf_files:
        dest = os.path.join(pdf_dir, f.name)
        with open(dest, "wb") as out:
            out.write(f.read())
        saved.append(dest)

    if not saved:
        return {"success": False, "chunks": 0, "message": "No PDFs to ingest."}

    # Run extraction
    components = os.path.join(ROOT, "tools", "vector_db components")
    if components not in sys.path:
        sys.path.insert(0, components)

    from extraction import extract_pdfs_to_chunks
    from ingestion import ingest_chunks

    chunks = extract_pdfs_to_chunks(saved)
    if not chunks:
        return {"success": False, "chunks": 0, "message": "Could not extract text from PDFs."}

    # Save chunks.json
    chunks_file = os.path.join(pdf_dir, "chunks.json")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # Ingest
    ingest_chunks(chunks, reset=False)

    return {
        "success": True,
        "chunks": len(chunks),
        "message": f"✅ Added {len(chunks):,} chunks to the knowledge base from {len(saved)} PDF(s).",
    }
