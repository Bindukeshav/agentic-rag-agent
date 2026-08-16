import os
import sqlite3
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ---------- Shared setup ----------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
web_search_tool = TavilySearchResults(k=3)

# ---------- Grader ----------
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader checking if a retrieved document is relevant to a user's question.
If the document contains information that helps answer the question, respond with exactly: yes
If the document is unrelated or unhelpful, respond with exactly: no
Answer with only one word: yes or no."""),
    ("human", "Question: {question}\n\nDocument: {document}\n\nIs this document relevant?")
])
grader_chain = grade_prompt | llm

def grade_document(question: str, document: str) -> bool:
    result = grader_chain.invoke({"question": question, "document": document})
    return "yes" in result.content.strip().lower()

# ---------- Contextualizer (rewrites follow-up questions) ----------
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and a new question, rewrite the new question so it can be
understood completely on its own, with no chat history needed. If it's already standalone, return
it unchanged. Only output the rewritten question, nothing else — no explanation."""),
    ("human", "Chat history:\n{chat_history}\n\nNew question: {question}\n\nStandalone question:")
])
contextualize_chain = contextualize_prompt | llm

# ---------- 1. State ----------
class AgentState(TypedDict):
    question: str
    documents: List[str]
    web_search_needed: bool
    generation: str
    chat_history: Annotated[List[str], operator.add]

# ---------- 2. Nodes ----------
def contextualize(state: AgentState):
    print("---CONTEXTUALIZE---")
    history = state.get("chat_history", [])
    if not history:
        return {"question": state["question"]}
    history_text = "\n".join(history[-6:])
    result = contextualize_chain.invoke({"chat_history": history_text, "question": state["question"]})
    rewritten = result.content.strip()
    print(f"Rewritten question: {rewritten}")
    return {"question": rewritten}

def retrieve(state: AgentState):
    print("---RETRIEVE---")
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": [doc.page_content for doc in docs]}

def grade(state: AgentState):
    print("---GRADE---")
    relevant_docs = [d for d in state["documents"] if grade_document(state["question"], d)]
    print(f"{len(relevant_docs)} of {len(state['documents'])} docs graded relevant")
    return {"documents": relevant_docs, "web_search_needed": len(relevant_docs) == 0}

def web_search(state: AgentState):
    print("---WEB SEARCH (fallback triggered)---")
    results = web_search_tool.invoke({"query": state["question"]})
    web_content = [r["content"] for r in results]
    return {"documents": state["documents"] + web_content}

def generate(state: AgentState):
    print("---GENERATE---")
    context = "\n\n".join(state["documents"])
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the context provided. If the context doesn't fully answer it, say so honestly."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    answer = result.content
    return {
        "generation": answer,
        "chat_history": [f"Q: {state['question']}", f"A: {answer}"]
    }

# ---------- 3. Conditional edge ----------
def decide_next_step(state: AgentState):
    return "web_search" if state["web_search_needed"] else "generate"

# ---------- 4. Wire it together ----------
workflow = StateGraph(AgentState)

workflow.add_node("contextualize", contextualize)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("contextualize")
workflow.add_edge("contextualize", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_next_step,
    {"web_search": "web_search", "generate": "generate"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

# ---------- 5. CLI test ----------
if __name__ == "__main__":
    thread_id = input("Enter a thread ID (any name, e.g. 'session1'): ")
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        result = graph.invoke({"question": question}, config=config)
        print("\n--- ANSWER ---")
        print(result["generation"])