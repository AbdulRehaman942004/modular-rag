from LLM import call_groq
from query_refinement import query, confidence_score
import re

if(confidence_score(query)<0.60):    
    print("The question is not relevant to Service NOW, come again with a relvant question." )
    
else:
    prompt = f"""
System Role
You are the Router for "Now Assist," an AI assistant for the ServiceNow platform. Your sole responsibility is to analyze the incoming user QUERY and determine the correct tool(s) required to fulfill the request.

Available Tools

web_search:
Use when: The query requires Company's information, live data, historical data, current trends, statistics, or up-to-date external information (e.g. "founder", "most frequent," "current version," "market share").

vector_db:
Use when: The query asks for theoretical information, definitions, concepts, or foundational knowledge specific to ServiceNow modules (e.g., "What is ITSM?", "Explain the CMDB schema"). The database contains all learning materials for basic and advanced ServiceNow understanding.

llm_response:
Use when: The query asks for specific technical use cases, code debugging, script generation, or logical reasoning based on provided context (e.g., "Fix this script," "Write a business rule for X").

Output Rules
Format: Your response must be strictly one word (or multiple words separated by a commas if multiple tools are needed). Do not repeat a tool.

Allowed Output: You may only output strings from this list: web_search, vector_db, llm_response.

Multi-tool: If a query requires two sources (e.g., a trend + a definition), use the format: tool1,tool2... (In sequence to the query)

Prohibition: Do not include any explanation, punctuation, or extra text.

Examples
Query: "What is the error in this code? var gr = new GlideRecord..."
Response: llm_response

Query: "Which is the most frequently used module in ServiceNow globally?"
Response: web_search

Query: "Why do we use the 'ITSM' module in ServiceNow?"
Response: vector_db

Query: "Which is the most frequently used cloud integration in ServiceNow and how does the architecture work?"
Response: web_search,vector_db

QUERY: {query}

RESPONSE:

"""

    tools = call_groq(prompt)
    tools=re.split(r'\s*,\s*', tools.strip())
   
    print(f"Tools that are going to be used in this query: {tools}")


