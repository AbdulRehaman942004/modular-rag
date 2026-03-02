from router import query, tools
from LLM import call_groq
import re

def query_decomposer(query:str, tools:list[str]) -> str:

    prompt = f"""
### System Role
You are a **Query Decomposer** for 'Now Assist'. You receive a complex User Query and a list of Tools that have already been selected to answer it.

### Task
1.  **Analyze** the `User Query` and break it down into distinct sub-queries.
2.  **Map** each sub-query to the specific tool in the `Assigned Tools` list that is best suited to answer it.
3.  **Order** your output to strictly match the order of the tools provided.

Assigned Tools could be any of them:
web_search
Use when: The query requires live data, historical data, current trends, statistics, or up-to-date external information (e.g., "most frequent," "current version," "market share").

vector_db
Use when: The query asks for theoretical information, definitions, concepts, or foundational knowledge specific to ServiceNow modules (e.g., "What is ITSM?", "Explain the CMDB schema"). The database contains all learning materials for basic and advanced ServiceNow understanding.

llm_response
Use when: The query asks for specific technical use cases, code debugging, script generation, or logical reasoning based on provided context (e.g., "Fix this script," "Write a business rule for X").

### Inputs
**User Query:** "{query}"
**Assigned Tools:** {tools}

### Output Rules
* **Format:** Output the sub-queries separated strictly by a comma `,`.
* **Constraint:** The first part of your response must correspond to the first tool, the second part to the second tool, and so on.
* **Content:** Do not add introductory text. Only output the separated questions.

### Examples
**Example 1**
* Query: "What is ITSM and what is the current stock price of ServiceNow?"
* Tools: ['vector_db', 'web_search']
* Response: What is ITSM?,What is the current stock price of ServiceNow?

**Example 2**
* Query: "Fix this script error and tell me who is the CEO."
* Tools: ['llm_response', 'web_search']
* Response: Fix this script error.,Who is the CEO of ServiceNow?

### YOUR RESPONSE:

    """

    response=call_groq(prompt)

    response=re.split(r'\s*,\s*', response.strip())
    return response

print(f"Query after decomposition: {query_decomposer(query,tools)}")