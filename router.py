from LLM import call_groq
from query_refinement import query, confidence_score
import re

if(confidence_score(query)<0.70):    
    print("The question is not relevant to Service NOW, come again with a relvant question." )
    

else:
    prompt = f"""
System Role
You are the Router for "Now Assist," an AI assistant for the ServiceNow platform. Your sole responsibility is to analyze the incoming user QUERY and determine the correct tool(s) required to fulfill the request.

Available Tools
web-search

Use when: The query requires live data, historical data, current trends, statistics, or up-to-date external information (e.g., "most frequent," "current version," "market share").

vector-db

Use when: The query asks for theoretical information, definitions, concepts, or foundational knowledge specific to ServiceNow modules (e.g., "What is ITSM?", "Explain the CMDB schema"). The database contains all learning materials for basic and advanced ServiceNow understanding.

llm-response

Use when: The query asks for specific technical use cases, code debugging, script generation, or logical reasoning based on provided context (e.g., "Fix this script," "Write a business rule for X").

Output Rules
Format: Your response must be strictly one word (or multiple words separated by a commas if multiple tools are needed).

Allowed Output: You may only output strings from this list: web-search, vector-db, llm-response.

Multi-tool: If a query requires two sources (e.g., a trend + a definition), use the format: tool1,tool2...

Prohibition: Do not include any explanation, punctuation, or extra text.

Examples
Query: "What is the error in this code? var gr = new GlideRecord..."
Response: llm-response

Query: "Which is the most frequently used module in ServiceNow globally?"
Response: web-search

Query: "Why do we use the 'ITSM' module in ServiceNow?"
Response: vector-db

Query: "Which is the most frequently used cloud integration in ServiceNow and how does the architecture work?"
Response: web-search,vector-db

QUERY: {query}

RESPONSE:

"""

    response = call_groq(prompt)
    response=[response] #converting it into a list

    def split_string_to_array(input_string):
        result_array = re.split(r'\s*,\s*', input_string.strip())
        return result_array

    if(',' in response):
        response=split_string_to_array(response)

    print(response)

