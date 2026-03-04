from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

#for groq
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

#for chatgpt
client_chatgpt = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)



from typing import List, Dict, Optional

def call_groq(prompt: str = "", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """Call the Groq LLM via the OpenAI-compatible Chat Completions API."""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
    )
    return response.choices[0].message.content or ""

#groq_api_key
def call_groq_stream(prompt: str = "", messages: Optional[List[Dict[str, str]]] = None):
    """Yield chunks of text from the Groq LLM as they become available."""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


#chatgpt_api_key

def call_chatgpt_stream(prompt: str = "", messages: Optional[List[Dict[str, str]]] = None):
    """Yield chunks of text from the ChatGPT API as they become available."""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        
    response = client_chatgpt.chat.completions.create(
        model="gpt-4o-mini",  # Change to your preferred OpenAI model (e.g. gpt-4-turbo, gpt-3.5-turbo)
        messages=messages,
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content



