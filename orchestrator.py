from router import query, tools
from tools.llm_response import llm_response 
from tools.vector_db import vector_db
from tools.web_search import web_search

if(len(tools)==1):    

    if tools[0]=="llm_response":
        llm_response()
    elif tools[0]=="vector_db":
        vector_db()
    elif tools[0]=="web search":
        web_search()



    



