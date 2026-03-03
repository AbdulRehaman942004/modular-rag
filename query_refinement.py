from LLM import call_groq

# Actual user query — change this to test different inputs
query = "What is ITSM? and tell me if this code is syntactically correct: var gr = new GlideRecord('incident'); gr.query(); Also tell me who is the founder of service now?"


def confidence_score(query: str) -> float:
    """Return a 0.0–1.0 score indicating how relevant the query is to ServiceNow."""

    prompt = f"""
System Role:
You are a Relevance Evaluator for 'Now Assist', an AI assistant dedicated to the ServiceNow platform. Your goal is to filter queries based on their relevance to ServiceNow, ServiceNow concepts, ServiceNow modules (ITSM, HRSD, CSM, etc.), scripting (GlideRecord, Jelly), or platform administration.

Task:
Analyze the incoming user query.

Determine if the query pertains to the ServiceNow ecosystem.

Assign a confidence score between 0.0 and 1.0:

0.0 = Completely unrelated (e.g., cooking, general world history, celebrities).

1.0 = Highly specific and technical ServiceNow query (e.g., "Business Rules," "GlideSystem," "ACLs"), conversational/small-talk queries that explicitly mention ServiceNow, OR standard AI Assistant greetings (e.g., "Hello", "Hi", "Who are you?", "How are you?").

Output ONLY the decimal number.

Output Rules:
Strict format: Output only the float value (e.g., 0.95).

No Text: Do not include words like "Score:", "Confidence:", or explanations.

No Markdown: Do not use bolding or code blocks in the output.

Input: "How do I write a GlideRecord query to fetch active incidents?"
Output: 0.99

Input: "Hello there, who are you?"
Output: 1.0

Input: "What is the best pizza recipe?"
Output: 0.01

Input: "Explain the difference between Client Scripts and UI Policies."
Output: 0.98

Input: "Who is the CEO of Microsoft?"
Output: 0.10

USER INPUT: {query}

OUTPUT:

"""
    response = call_groq(prompt)

    # Guard: if the LLM returns something non-numeric, default to 0.0 (reject)
    try:
        return float(response.strip())
    except ValueError:
        print(f"[confidence_score] Warning: could not parse score '{response}', defaulting to 0.0")
        return 0.0


if __name__ == "__main__":
    print(f"Query: {query}")
    print(f"Confidence Score: {confidence_score(query)}")