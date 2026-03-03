from router import query, tools
from tools.llm_response import llm_response
from tools.vector_db import vector_db
from tools.web_search import web_search
from LLM import call_groq
from query_decomposer import query_decomposer


def _run_tool(tool_name: str, q: str) -> str:
    """Run a single tool and return its output.
    Falls back to llm_response if the tool raises an error or returns empty.
    Non-ServiceNow queries are already rejected upstream by router.py.
    """
    try:
        if tool_name == "llm_response":
            result = llm_response(q)
        elif tool_name == "vector_db":
            result = vector_db(q)
        elif tool_name == "web_search":
            result = web_search(q)
        else:
            result = ""

        # Treat empty / error-like results as failures
        if not result or result.strip().startswith("["):
            raise ValueError(f"Tool '{tool_name}' returned an unusable result.")

        return result

    except Exception as e:
        print(f"[orchestrator] '{tool_name}' failed ({e}). Falling back to llm_response...")
        try:
            fallback = llm_response(q)
            if fallback:
                return fallback
        except Exception as fe:
            print(f"[orchestrator] llm_response fallback also failed: {fe}")

        return (
            "I was unable to retrieve information for this query at the moment. "
            "Please try again shortly."
        )


def orchestrate_query(query: str, tools: list[str]) -> str:
    context = ""

    if not tools:
        print("[orchestrator] No tools selected — query was likely off-topic.")
        return "Your question does not appear to be related to ServiceNow. Please ask a ServiceNow-related question."

    if len(tools) == 1:
        context = _run_tool(tools[0], query)

    elif len(tools) > 1:
        sub_queries = query_decomposer(query, tools)

        # Guard: zip ensures we never go out of bounds if counts differ
        if len(sub_queries) != len(tools):
            print(f"[orchestrator] Warning: decomposer returned {len(sub_queries)} sub-queries for {len(tools)} tools. Zipping to shortest.")

        sub_contexts = []
        for tool, sub_q in zip(tools, sub_queries):
            sub_contexts.append(_run_tool(tool, sub_q))

        context = "\n\n".join(sub_contexts)

    prompt = f"""You are "Now Assist", an expert AI assistant for the ServiceNow platform.
Your job is to provide accurate, helpful answers to questions about ServiceNow using only the CONTEXT provided below.

CONTEXT:
{context}

USER QUERY:
{query}

INSTRUCTIONS:
- Answer the query using ONLY the information available in the CONTEXT above.
- If the context comes from multiple sources (e.g., documentation + web), synthesize them into one clear, unified response.
- Be concise, structured, and professional.
- Use bullet points or numbered lists where helpful.
- If the context does not contain enough information to answer the query, say: "I don't have enough information to answer this based on the available sources."
- Do NOT make up facts or hallucinate details not present in the CONTEXT.

RESPONSE:"""

    response = call_groq(prompt)
    print(f"\nFinal Response:\n{response}")
    return response


orchestrate_query(query, tools)
