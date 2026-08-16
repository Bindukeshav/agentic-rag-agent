from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from graph import graph, vectorstore  # reuses everything you already built

app = FastAPI(title="Agentic RAG Assistant")

class ChatRequest(BaseModel):
    thread_id: str
    question: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    os.makedirs("docs", exist_ok=True)
    save_path = f"docs/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(save_path) if file.filename.endswith(".pdf") else TextLoader(save_path, encoding="utf-8")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    vectorstore.add_documents(chunks)

    return {"status": "ingested", "filename": file.filename, "chunks_added": len(chunks)}

@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = graph.invoke({"question": request.question}, config=config)
    return {"answer": result["generation"]}

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    return {
        "question": state.values.get("question"),
        "answer": state.values.get("generation")
    }