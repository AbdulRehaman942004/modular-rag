from LLM import call_groq

# Actual user query — change this to test different inputs
query = "What is ITSM? and tell me if this code is syntactically correct: var gr = new GlideRecord('incident'); gr.query(); Also tell me who is the founder of service now?"


def confidence_score(query: str, chat_history: list = None) -> float:
    """Return a 0.0–1.0 score indicating how relevant the query is to ServiceNow."""

    history_str = ""
    if chat_history:
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-3:]])
    else:
        history_str = "No prior conversation context."

    prompt = f"""
System Role:
You are a Relevance Evaluator for 'Now Assist', an AI assistant dedicated to the ServiceNow platform. Your goal is to filter queries based on their relevance to ServiceNow, ServiceNow concepts, ServiceNow modules (ITSM, HRSD, CSM, etc.), scripting (GlideRecord, Jelly), or platform administration.

Task:
Analyze the incoming user query IN THE CONTEXT of the recent conversation history.

Determine if the query pertains to the ServiceNow ecosystem, OR if it is a valid conversational continuation/follow-up to a previous ServiceNow topic.

Assign a confidence score between 0.0 and 1.0:

0.0 = Completely unrelated (e.g., cooking, general world history, celebrities) and has nothing to do with recent context.

1.0 = Highly specific and technical ServiceNow query (e.g., "Business Rules," "GlideSystem," "ACLs"), standard AI Assistant greetings (e.g., "Hello", "Hi"), OR any conversational follow-up commands/questions that relate back to the recent ServiceNow chat history (e.g., "Summarize it", "Explain more", "Make it shorter").

Output ONLY the decimal number.

Output Rules:
Strict format: Output only the float value (e.g., 0.95).

No Text: Do not include words like "Score:", "Confidence:", or explanations.

No Markdown: Do not use bolding or code blocks in the output.

Input: "How do I write a GlideRecord query to fetch active incidents?"
Output: 0.99

Input: "Hello there, who are you?"
Output: 1.0

Input: "Can you summarize that in one paragraph?"
Output: 1.0

Input: "What is the best pizza recipe?"
Output: 0.01

Input: "Explain the difference between Client Scripts and UI Policies."
Output: 0.98

Input: "Who is the CEO of Microsoft?"
Output: 0.10

RECENT CONVERSATION CONTEXT:
{history_str}

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