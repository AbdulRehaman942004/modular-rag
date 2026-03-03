from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
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