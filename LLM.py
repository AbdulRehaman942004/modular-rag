from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def call_groq(query:str ,prompt:str):
    response = client.responses.create(
        input= prompt,
        model="openai/gpt-oss-20b",
    )
    return response.output_text


# query = "What is the capital of France?"

# prompt = f"""
# CONTEXT:
# You are a helpful assistant that can answer questions and help with tasks.

# QUERY: {query}

# INSTRUCTIONS: You must answer the query in just one word under any circumstances.
# """

# print(call_groq(query, prompt))