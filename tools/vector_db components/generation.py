import os
from typing import List, Tuple
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Groq via OpenAI-compatible client
_client_groq = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Default model
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ChromaDB path (relative to project root)
CHROMADB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chromadb_data")

# Default collection name
DEFAULT_COLLECTION = "servicenow_knowledge_base"

# Module-level singleton client — created once, reused on every query
_chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)


def _get_collection(collection_name: str) -> chromadb.Collection:
    """Retrieve an existing ChromaDB collection using the shared client."""
    return _chroma_client.get_collection(name=collection_name)



def retrieve_context(
    user_query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 10,
) -> List[str]:
    """Query ChromaDB and return the top-N relevant document chunks.

    Args:
        user_query:      The user's question.
        collection_name: ChromaDB collection to query.
        n_results:       Number of top chunks to retrieve.

    Returns:
        A list of relevant text chunks.
    """
    collection = _get_collection(collection_name)
    results = collection.query(
        query_texts=[user_query],
        n_results=n_results,
    )
    documents = results.get("documents") or []
    return documents[0] if documents else []


def generate_answer(
    user_query: str,
    context_chunks: List[str],
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Generate a grounded answer from retrieved context chunks.

    Args:
        user_query:     The user's question.
        context_chunks: List of relevant text excerpts from the knowledge base.
        model_name:     Groq model to use for generation.

    Returns:
        A string containing the generated answer.
    """
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    prompt = f"""You are a helpful assistant specialized in ServiceNow. 
Provide a clear, accurate answer to the user's question using the information below.

INFORMATION GATHERED:
{context_text}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Base your answer strictly on the INFORMATION GATHERED provided.
- Be clear, highly detailed, comprehensively structured, and directly address the question.
- Use headings, bullet points, numbered lists, and bold text extensively to organize the information and provide a structured output.
- Do not include information not present in the INFORMATION GATHERED.
- CRITICAL: Do NOT mention "context", "search results", or "the provided information" in your answer. Present the information directly and confidently as your own knowledge.

ANSWER:"""

    response = _client_groq.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def run_rag_query(
    user_query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 10,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[str, List[str]]:
    """End-to-end RAG: retrieve relevant chunks and generate a grounded answer.

    Args:
        user_query:      The user's question.
        collection_name: ChromaDB collection to query.
        n_results:       Number of chunks to retrieve.
        model_name:      Groq model for answer generation.

    Returns:
        A tuple of (answer_string, list_of_context_chunks_used).
    """
    context_chunks = retrieve_context(user_query, collection_name, n_results)
    answer = generate_answer(user_query, context_chunks, model_name)
    return answer, context_chunks


if __name__ == "__main__":
    query = "What is the ITSM module in ServiceNow?"
    answer, chunks = run_rag_query(query)
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nContext chunks used: {len(chunks)}")
