from router import query, tools
from tools.llm_response import llm_response 
from tools.vector_db import vector_db
from tools.web_search import web_search
from LLM import call_groq
from query_decomposer import query_decomposer

def orchestrate_query(query:str, tools:list[str]):
    if(len(tools)==1):    

        if tools[0]=="llm_response":
            context= llm_response(query)
        elif tools[0]=="vector_db":
            context = vector_db(query)
        elif tools[0]=="web_search":
            context = web_search(query)

    elif(len(tools)>1):
        query=query_decomposer(query, tools) 

        print(f"Query after decomposition: {query_decomposer(query,tools)}")

        for i in range(len(query)):

            if tools[i]=="llm_response":
                context[i]= llm_response(query[i])
            elif tools[i]=="vector_db":
                context[i] = vector_db(query[i])
            elif tools[i]=="web_search":
                context[i] = web_search(query[i])

        for i in range(len(query)):

            context="\n\n".join(context)
            

#     prompt = f""""
# CONTEXT:{context}

# QUERY: {query}

# INSTRUCTIONS:

# RESPONSE:

# # """
#     response=call_groq(prompt)
#     return response

orchestrate_query(query,tools)


