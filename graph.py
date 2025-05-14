# Aqui vai conter todas as lógicas entre os agents
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from tavily import TavilyClient

from schemas import *
from prompts import *

import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_S")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)


# Nós
def build_first_queries(state: ReportState):
    class QueryList(BaseModel):
        queries: List[str]

    user_input = state.user_input

    prompt = build_queries.format(user_input=user_input)
    query_llm = llm.with_structured_output(QueryList)
    result = query_llm.invoke(prompt)

    return {"queries": result.queries}


def spawn_researchers(state: ReportState):
    return [Send("single_search", query) for query in state.queries]
    

def single_search(query: str):
    tavily_client = TavilyClient()
    
    results =tavily_client.search(
        query,
        max_results=1,
        include_raw_content=False
    )
    url = results["results"][0]["url"]
    url_extraction = tavily_client.extract(url)

    if len(url_extraction["results"]) > 0:
        raw_content = url_extraction["results"][0]["raw_content"]
        prompt = resume_research.format(user_input=user_input, search_results=raw_content)

        llm_result = llm.invoke(prompt)
        query_results = QueryResult(title=results["results"][0]["title"],
                                   url=url,
                                   resume=llm_result.content)
        
    return {"queries_results": [query_results]}


def final_writer(state: ReportState):
    search_results = ""
    references = ""
    for i, result in enumerate(state.queries_results):
        search_results += f"[{i+1}]\n\n"
        search_results += f"Title {result.title}\n"
        search_results += f"URL {result.url}\n"
        search_results += f"Content {result.resume}\n\n"
        search_results += f"===================\n\n"

        references += f"[{i+1}] - [{result.title}]({result.url})\n"

    prompt = build_final_response.format(user_input=user_input,
                                         search_results=search_results)
    
    llm_result = llm.invoke(prompt)
    final_response = llm_result.content + "\n\n References:\n" + references
    print(final_response)

    return {"final_response": final_response}


#Edges
builder = StateGraph(ReportState)
builder.add_node("build_first_queries", build_first_queries)
builder.add_node("single_search", single_search)
builder.add_node("final_write", final_writer)

builder.add_edge(START, "build_first_queries")
# Nó condicional do "first query" para o "spawn_researchers" e vai mandar para o "single_search"
#  = STRING ("build_first_queries")> FUNÇÃO (spawn_researchers) > LISTA: (["single_search"])
builder.add_conditional_edges("build_first_queries",
                              spawn_researchers,
                              ["single_search"])
# Conectar cada um do single_search com o "final_write"
builder.add_edge("single_search",
                 "final_write")
# E finalizamos no Final Write
builder.add_edge("final_write", END)

graph = builder.compile()

if __name__ == "__main__":

    #from IPython.display import Image, display
    #display(Image(graph.get_graph().draw_mermaid_png))

    st.title("Perplexity Project")
    user_input = st.text_input("Qual a sua pergunta?",
                               value="How is the process of building a LLM")
    
    if st.button("Pesquisar"):
        with st.status("Gerando resposta"):
            #response = graph.invoke({"user_input": user_input})
            #st.write(response)

            for output in graph.stream({"user_input": user_input},
                                       stream_mode="debug"):
                if output["type"] == "task_result":
                    st.write(f"Running {output['payload']['name']}")
                    st.write(output)
        
        #print(output)
        response = output["payload"]["result"][0][1]
        think_str = response.split("</think>")[0]
        final_response = response.split("</think>")[0]

        with st.expander("Reflexão", expanded=False):
            st.write(think_str)
        st.write(final_response)

    #user_input = """
    #Can you explain to me how
    #is the full process of building a LLM? From scratch
    #"""
    #graph.invoke({"user_input": user_input})