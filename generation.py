import os
from typing import List, Tuple
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

user_query="How old is Oxford?"

client_groq = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
)


client = chromadb.PersistentClient(path="chromadb_data")


def confidence_score(
    user_query: str,
    model_name: str = "openai/gpt-oss-20b"
) -> float:

    prompt = f"""
        You are a RAG-based query analysis assistant for an application. The app answers user queries based only on the document "Oxford Guide-2022.pdf". 

        Your task is as follows:

        1. Receive a user query.
        2. Analyze if the query is related to the content of Oxford or "Oxford Guide-2022.pdf".
        3. Assign a confidence score between 0 and 1:
        - 0 = completely unrelated to Oxford.
        - 1 = fully related to Oxford.

        6. Output ONLY the confidence score as a decimal number**. No explanations, no text, no punctuation other than the decimal number. 

        Example:
        - Input: "whats Oxford universty famous for?"
        - Output: 0.95
    """
    answer = client_groq.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ],
    )

    confidence_score = answer.choices[0].message.content or ""
    return float(confidence_score)

def run_rag_query(
    user_query: str,
    collection_name: str = "Oxford-Guide-2022",
    n_results: int = 10,
    model_name: str = "openai/gpt-oss-20b",
) -> Tuple[str, List[str]]:
    """Run the RAG query using the existing ChromaDB collection and Groq client."""
    # get the existing collection
    collection = client.get_collection(name=collection_name)

    load_dotenv()

    context = collection.query(
        query_texts=[user_query],
        n_results=n_results,
    )

    documents = context.get("documents") or []
    context_chunks: List[str] = documents[0] if documents else []

    prompt = f"""
You are an assistant that answers questions based on provided context.

CONTEXT:
{context}

USER QUERY:
{user_query}

Instructions:
- Answer the query using ONLY the information in the CONTEXT.
- Keep the answer clear, and relevant.
- Do not make assumptions beyond the CONTEXT.

Answer:
"""

    response = client_groq.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content or ""
    return answer, context_chunks


if __name__ == "__main__":
    # Preserve original behaviour when run directly

    if(confidence_score(user_query)>=0.8):
        answer, _ = run_rag_query(user_query)
        print(f"\nResponse: {answer}")

    elif(confidence_score()<0.8):
        print(f"\nResponse: The question is irrelevant to Oxford.")



