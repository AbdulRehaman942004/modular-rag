import sys
import os

# Make the project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LLM import call_groq


def llm_response(query: str) -> str:
    """Handle technical ServiceNow queries: debugging, scripting, and logical reasoning.

    This tool is used when the query involves:
    - Code debugging (e.g. "Fix this GlideRecord script")
    - Script / business rule generation (e.g. "Write a Business Rule for X")
    - Technical reasoning or step-by-step explanations

    Args:
        query: The user's technical question or request.

    Returns:
        A string containing the LLM's response, used as context by the orchestrator.
    """
    prompt = f"""You are "Now Assist", an expert ServiceNow developer and technical assistant.
You specialize in:
- Writing and debugging ServiceNow scripts (GlideRecord, Business Rules, Script Includes, Client Scripts, etc.)
- Explaining ServiceNow technical concepts and APIs
- Generating Flows, Workflows, and automation logic
- Solving ServiceNow-specific coding and configuration problems
- Greeting the user, exchanging pleasantries, and introducing yourself as Now Assist when they say "hello" or "who are you".

USER REQUEST:
{query}

INSTRUCTIONS:
- Provide a clear, accurate, and technically precise answer.
- If writing code, use proper ServiceNow scripting conventions (server-side or client-side as appropriate).
- Add brief inline comments to any code you write so it is easy to understand.
- If you need to make assumptions about the user's environment, state them clearly.
- Keep the response focused and structured.

RESPONSE:"""

    print(f"[llm_response] Processing technical query: {query!r}")
    response = call_groq(prompt)
    return response
