from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load your document
loader = TextLoader("docs/TERA_Presentation_Script_Final.md", encoding="utf-8")
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} document(s)")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunks")

# 3. Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Stored in Chroma")

# 4. Sanity check search
question = "Why does the doctor use hand gestures instead of touching a mouse or keyboard?"
results = vectorstore.similarity_search(question, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:300])