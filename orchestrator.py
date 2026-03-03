from router import query, tools
from tools.llm_response import llm_response 
from tools.vector_db import vector_db
from tools.web_search import web_search
from LLM import call_groq
from query_decomposer import query_decomposer

def orchestrate_query(query: str, tools: list[str]):
    context = ""

    if len(tools) == 1:
        if tools[0] == "llm_response":
            context = llm_response(query)
        elif tools[0] == "vector_db":
            context = vector_db(query)
        elif tools[0] == "web_search":
            context = web_search(query)

    elif len(tools) > 1:
        sub_queries = query_decomposer(query, tools)
        sub_contexts = []

        for i in range(len(sub_queries)):
            if tools[i] == "llm_response":
                sub_contexts.append(llm_response(sub_queries[i]))
            elif tools[i] == "vector_db":
                sub_contexts.append(vector_db(sub_queries[i]))
            elif tools[i] == "web_search":
                sub_contexts.append(web_search(sub_queries[i]))

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
