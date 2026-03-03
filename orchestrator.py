from router import query, tools
from tools.llm_response import llm_response 
from tools.vector_db import vector_db
from tools.web_search import web_search
from LLM import call_groq
from query_decomposer import query_decomposer

def orchestrate_query(query:str, tools:list[str]):
    if(len(tools)==1):    

        if tools[0]=="llm_response":
            context= llm_response()
        elif tools[0]=="vector_db":
            context = vector_db()
        elif tools[0]=="web_search":
            context = web_search()

    elif(len(tools)>1):
      query=query_decomposer(query, tools) 
      print(f"Query after decomposition: {query_decomposer(query,tools)}")


#     prompt = f""""
# CONTEXT:{context}

# QUERY: {query}

# INSTRUCTIONS:

# RESPONSE:

# # """
#     response=call_groq(prompt)
#     return response

orchestrate_query(query,tools)


